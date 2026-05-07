import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from app.core.errors import AppServiceError
from app.domains.commit.schemas import (
    CommitAnalyzeRequest,
    CommitAnalyzeRunAccepted,
    CommitAnalyzeRunPhase,
    CommitAnalyzeRunResult,
    CommitAnalyzeRunStatus,
    CommitAnalyzeRunStatusValue,
)
from app.domains.commit.services.diff_filter import filter_changed_files
from app.domains.commit.services.summarize import (
    store_commit_embedding,
    summarize_commit,
)

RUN_TTL = timedelta(hours=24)
MAX_RUN_RECORDS = 300
logger = logging.getLogger(__name__)


@dataclass
class _CommitAnalyzeRunRecord:
    run_id: str
    status: CommitAnalyzeRunStatusValue
    phase: CommitAnalyzeRunPhase
    request: CommitAnalyzeRequest
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    result: CommitAnalyzeRunResult | None = None


_runs: dict[str, _CommitAnalyzeRunRecord] = {}
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
        if run.status in {"queued", "processing"}:
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
            if item[1].status not in {"queued", "processing"}
        ),
        key=lambda item: item[1].submitted_at,
    )
    for run_id, _ in eviction_candidates[:overflow]:
        _runs.pop(run_id, None)


def _to_run_status(run: _CommitAnalyzeRunRecord) -> CommitAnalyzeRunStatus:
    # 내부 저장 레코드를 외부 응답 모델로 변환
    return CommitAnalyzeRunStatus(
        run_id=run.run_id,
        status=run.status,
        phase=run.phase,
        commit_id=run.request.commit_id,
        commit_hash=run.request.commit_hash,
        repository_id=run.request.repository_id,
        submitted_at=run.submitted_at.isoformat(),
        started_at=_to_iso(run.started_at),
        finished_at=_to_iso(run.finished_at),
        error=run.error,
        result=run.result.model_copy(deep=True) if run.result else None,
    )


async def create_commit_analyze_run(
    request: CommitAnalyzeRequest,
) -> CommitAnalyzeRunAccepted:
    # 커밋 분석 비동기 실행 레코드를 생성
    async with _lock:
        _cleanup_runs_locked()
        run_id = uuid4().hex
        _runs[run_id] = _CommitAnalyzeRunRecord(
            run_id=run_id,
            status="queued",
            phase="queued",
            request=request.model_copy(deep=True),
            submitted_at=_utc_now(),
        )
        return CommitAnalyzeRunAccepted(
            run_id=run_id,
            status="queued",
            phase="queued",
            commit_id=request.commit_id,
            commit_hash=request.commit_hash,
            repository_id=request.repository_id,
        )


async def _mark_run_processing(run_id: str) -> None:
    # run 상태를 처리중(summarizing)으로 전환
    async with _lock:
        run = _runs.get(run_id)
        if not run:
            return
        run.status = "processing"
        run.phase = "summarizing"
        run.started_at = _utc_now()
        run.error = None


async def _mark_run_phase(
    run_id: str,
    phase: CommitAnalyzeRunPhase,
    result: CommitAnalyzeRunResult | None = None,
) -> None:
    # 단계별 중간 결과를 저장하며 phase 갱신
    async with _lock:
        run = _runs.get(run_id)
        if not run:
            return
        run.phase = phase
        if result is not None:
            run.result = result.model_copy(deep=True)


async def _mark_run_completed(run_id: str, result: CommitAnalyzeRunResult) -> None:
    # run을 최종 완료 상태로 전환
    async with _lock:
        run = _runs.get(run_id)
        if not run:
            return
        run.status = "completed"
        run.phase = "embedding_ready"
        run.result = result.model_copy(deep=True)
        run.error = None
        run.finished_at = _utc_now()


async def _mark_run_failed(run_id: str, error: str) -> None:
    # run을 실패 상태로 전환하고 오류 메시지 저장
    async with _lock:
        run = _runs.get(run_id)
        if not run:
            return
        run.status = "failed"
        run.phase = "failed"
        run.error = error
        run.finished_at = _utc_now()


async def get_commit_analyze_run_status(
    run_id: str,
) -> CommitAnalyzeRunStatus | None:
    # run 현재 상태/결과를 조회
    async with _lock:
        _cleanup_runs_locked()
        run = _runs.get(run_id)
        if not run:
            return None
        return _to_run_status(run)


async def run_commit_analyze_pipeline(run_id: str) -> None:
    # 커밋 요약부터 임베딩 저장까지 단계별(phase)로 실행
    async with _lock:
        run = _runs.get(run_id)
        if not run:
            return
        request = run.request.model_copy(deep=True)

    try:
        filtered_files = filter_changed_files(request.changed_file_list)
        if not filtered_files:
            raise AppServiceError(
                "분석할 수 있는 변경 파일이 없습니다.", status_code=400
            )

        await _mark_run_processing(run_id)
        summary = await summarize_commit(request.message, filtered_files)
        result = CommitAnalyzeRunResult(
            commit_id=request.commit_id,
            commit_hash=request.commit_hash,
            repository_id=request.repository_id,
            summary=summary,
            embedding_ready=False,
        )
        await _mark_run_phase(
            run_id=run_id,
            phase="summary_ready",
            result=result,
        )

        await _mark_run_phase(
            run_id=run_id,
            phase="embedding",
            result=result,
        )
        await store_commit_embedding(
            commit_hash=request.commit_hash,
            repository_id=request.repository_id,
            message=request.message,
            changed_file_list=filtered_files,
            commit_id=request.commit_id,
        )

        completed_result = result.model_copy(update={"embedding_ready": True})
        await _mark_run_completed(run_id, completed_result)
    except AppServiceError as e:
        await _mark_run_failed(run_id, f"{e.status_code}: {e.message}")
    except Exception as e:
        logger.exception(
            "commit analyze run failed",
            extra={"run_id": run_id},
        )
        await _mark_run_failed(run_id, f"unexpected_error: {e}")
