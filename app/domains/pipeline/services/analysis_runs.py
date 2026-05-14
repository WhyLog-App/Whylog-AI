import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from app.core.enums import RunStatus, TranscribeRunPhase
from app.domains.pipeline.schemas import (
    TranscribeAnalysisResponse,
    TranscribeAnalysisRunAccepted,
    TranscribeAnalysisRunStatus,
)

RUN_TTL = timedelta(hours=24)
MAX_RUN_RECORDS = 300


@dataclass
class _RunRecord:
    run_id: str
    status: RunStatus
    phase: TranscribeRunPhase
    meeting_id: str | None
    project_id: str | None
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    result: TranscribeAnalysisResponse | None = None


_runs: dict[str, _RunRecord] = {}
_lock = asyncio.Lock()


def _utc_now() -> datetime:
    # UTC 기준 현재 시각 반환
    return datetime.now(UTC)


def _to_iso(value: datetime | None) -> str | None:
    # datetime을 ISO8601 문자열로 변환
    return value.isoformat() if value else None


def _cleanup_runs_locked() -> None:
    # TTL/최대 개수 정책에 따라 run 저장소를 정리
    now = _utc_now()
    expiry_cutoff = now - RUN_TTL

    expired_ids = []
    for run_id, run in _runs.items():
        # 진행 중 실행은 TTL 삭제 대상에서 제외
        if run.status in {RunStatus.QUEUED, RunStatus.PROCESSING}:
            continue
        reference_time = run.finished_at or run.submitted_at
        if reference_time < expiry_cutoff:
            expired_ids.append(run_id)
    for run_id in expired_ids:
        _runs.pop(run_id, None)

    if len(_runs) <= MAX_RUN_RECORDS:
        return

    overflow = len(_runs) - MAX_RUN_RECORDS
    eviction_candidates = sorted(
        (
            item
            for item in _runs.items()
            if item[1].status not in {RunStatus.QUEUED, RunStatus.PROCESSING}
        ),
        key=lambda item: item[1].submitted_at,
    )
    for run_id, _ in eviction_candidates[:overflow]:
        _runs.pop(run_id, None)


def _to_run_status(run: _RunRecord) -> TranscribeAnalysisRunStatus:
    # 내부 저장 레코드를 외부 응답 모델로 변환
    return TranscribeAnalysisRunStatus(
        run_id=run.run_id,
        status=run.status,
        phase=run.phase,
        meeting_id=run.meeting_id,
        project_id=run.project_id,
        submitted_at=run.submitted_at.isoformat(),
        started_at=_to_iso(run.started_at),
        finished_at=_to_iso(run.finished_at),
        error=run.error,
        result=run.result.model_copy(deep=True) if run.result else None,
    )


async def create_run(
    meeting_id: str | None,
    project_id: str | None,
) -> TranscribeAnalysisRunAccepted:
    # 비동기 파이프라인 실행 레코드를 생성
    async with _lock:
        _cleanup_runs_locked()
        run_id = uuid4().hex
        _runs[run_id] = _RunRecord(
            run_id=run_id,
            status=RunStatus.QUEUED,
            phase=TranscribeRunPhase.QUEUED,
            meeting_id=meeting_id,
            project_id=project_id,
            submitted_at=_utc_now(),
        )
        return TranscribeAnalysisRunAccepted(
            run_id=run_id,
            status=RunStatus.QUEUED,
            phase=TranscribeRunPhase.QUEUED,
            meeting_id=meeting_id,
            project_id=project_id,
        )


async def mark_run_processing(run_id: str) -> None:
    # run 상태를 처리중(transcribing)으로 전환
    async with _lock:
        run = _runs.get(run_id)
        if not run:
            return
        run.status = RunStatus.PROCESSING
        run.phase = TranscribeRunPhase.TRANSCRIBING
        run.started_at = _utc_now()
        run.error = None


async def mark_run_phase(
    run_id: str,
    phase: TranscribeRunPhase,
    result: TranscribeAnalysisResponse | None = None,
) -> None:
    # 단계별 중간 결과를 저장하며 phase 갱신
    async with _lock:
        run = _runs.get(run_id)
        if not run:
            return
        run.phase = phase
        if result is not None:
            run.result = result.model_copy(deep=True)


async def mark_run_completed(run_id: str, result: TranscribeAnalysisResponse) -> None:
    # run을 최종 완료 상태로 전환
    async with _lock:
        run = _runs.get(run_id)
        if not run:
            return
        run.status = RunStatus.COMPLETED
        run.phase = TranscribeRunPhase.APPLICATIONS_READY
        run.result = result.model_copy(deep=True)
        run.error = None
        run.finished_at = _utc_now()


async def mark_run_failed(run_id: str, error: str) -> None:
    # run을 실패 상태로 전환하고 오류 메시지 저장
    async with _lock:
        run = _runs.get(run_id)
        if not run:
            return
        run.status = RunStatus.FAILED
        run.phase = TranscribeRunPhase.FAILED
        run.error = error
        run.finished_at = _utc_now()


async def get_run_status(run_id: str) -> TranscribeAnalysisRunStatus | None:
    # run 현재 상태/결과를 조회
    async with _lock:
        _cleanup_runs_locked()
        run = _runs.get(run_id)
        if not run:
            return None
        return _to_run_status(run)
