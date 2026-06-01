import json
import logging
import os
import re

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from app.core.enums import TimelineStep
from app.core.errors import AppServiceError
from app.core.gemini import generate_content_with_retry
from app.domains.meeting_analysis.schemas import (
    Application,
    MeetingAnalysis,
    MeetingAnalysisResult,
)
from app.domains.transcribe.schemas import TranscribeSegment

logger = logging.getLogger(__name__)
AMBIGUOUS_SHORT_UTTERANCES = {
    "네",
    "예",
    "응",
    "음",
    "어",
    "맞아요",
    "맞습니다",
    "아니요",
    "확인",
    "오케이",
    "ok",
    "okay",
}
GENERIC_MEETING_TITLE_KEYWORDS = {
    "whylog",
    "프로젝트",
    "회의",
    "미팅",
    "기능",
    "점검",
    "테스트",
    "확인",
}
GENERIC_MEETING_TITLES = {
    "",
    "데이터 없음",
    "회의",
    "미팅",
    "회의록",
    "Whylog 회의",
    "Whylog 프로젝트 회의",
    "Whylog 프로젝트 기능 점검 회의",
    "Whylog meeting",
}
REASON_NOUN_ENDING_PATTERNS = (
    (re.compile(r"(.+?)(?:이|가)\s+저하된다\.?$"), r"\1 저하"),
    (re.compile(r"(.+?)(?:이|가)\s+증가한다\.?$"), r"\1 증가"),
    (re.compile(r"(.+?)(?:이|가)\s+감소한다\.?$"), r"\1 감소"),
    (re.compile(r"(.+?)(?:이|가)\s+개선된다\.?$"), r"\1 개선"),
    (re.compile(r"(.+?)(?:이|가)\s+확보된다\.?$"), r"\1 확보"),
    (re.compile(r"(.+?)(?:이|가)\s+필요하다\.?$"), r"\1 필요성"),
    (re.compile(r"(.+?)할\s+필요가\s+있다\.?$"), r"\1 필요성"),
    (re.compile(r"(.+?)해야\s+한다\.?$"), r"\1 필요성"),
    (re.compile(r"(.+?)해야\s+함\.?$"), r"\1 필요성"),
    (re.compile(r"(.+?)(?:을|를)\s+방지한다\.?$"), r"\1 방지"),
    (re.compile(r"(.+?)(?:을|를)\s+개선한다\.?$"), r"\1 개선"),
    (re.compile(r"(.+?)(?:을|를)\s+확보한다\.?$"), r"\1 확보"),
    (re.compile(r"(.+?)(?:을|를)\s+보장한다\.?$"), r"\1 보장"),
    (re.compile(r"(.+?)(?:을|를)\s+줄인다\.?$"), r"\1 감소"),
    (re.compile(r"(.+?)(?:을|를)\s+감소시킨다\.?$"), r"\1 감소"),
)
NOUN_ENDING_REASON_RULE = [
    "[근거 출력 형식 - 반드시 준수]",
    "1. application_reasons의 모든 항목은 명사형 어미로 끝낸다.",
    '2. 허용 예: "사용자 경험 저하 방지", "운영 리스크 감소",',
    '   "응답 속도 개선 필요성", "데이터 정합성 확보"',
    '3. 금지 예: "사용자 경험이 저하된다", "운영 리스크가 감소한다",',
    '   "응답 속도 개선이 필요하다", "정합성을 확보해야 한다"',
    "4. '~다', '~한다', '~됩니다', '~합니다', '~필요하다', '~해야 한다'로",
    "   끝나는 서술형 문장은 절대 출력하지 않는다.",
    "5. 전체 문장을 설명하지 말고 짧은 근거 라벨처럼 작성한다.",
]
CONCRETE_APPLICATION_RULE = [
    "[적용사항 구체화 규칙 - 반드시 준수]",
    "1. application_title은 주제명이 아니라 실제 실행/반영 단위로 작성한다.",
    '2. 금지 예: "전사 품질 개선", "회의 분석 고도화", "데모 준비"',
    '3. 허용 예: "WebSocket 발화로그 기준으로 STT 화자 보정 적용",',
    '   "전시회 데모 시나리오를 5분 이내 흐름으로 재구성"',
    "4. timeline의 적용합의 content도 구체적인 실행 문장으로 작성한다.",
    "5. 적용합의 content는 무엇을 어떻게 반영할지 드러나야 한다.",
    '6. "개선하기로 함", "논의하기로 함", "준비하기로 함"처럼',
    "   대상/방법이 빠진 두루뭉술한 표현은 절대 사용하지 않는다.",
]
TIMELINE_EVIDENCE_ALIGNMENT_RULE = [
    "[타임라인 근거 정렬 규칙 - 반드시 준수]",
    "1. timeline은 application_title/application_reasons의 핵심 작업 의도를",
    "   직접 뒷받침하는 발화만 포함한다.",
    "2. 같은 화면, API, 도메인명이 겹쳐도 작업 의도가 다르면",
    "   같은 application의 timeline에 넣지 않는다.",
    "3. '어디에 적용하는가'보다 '무엇을 바꾸기로 했는가'를 우선한다.",
    '4. 예: "커밋 상세 페이지 로딩 스피너 컴포넌트 적용"에는',
    '   "응답 필드명 통일" 발화를 포함하지 않는다.',
    '5. 예: "커밋 상세 API 응답 필드명 통일"에는',
    '   "로딩 스피너 적용" 발화를 포함하지 않는다.',
    "6. 공통 범위 키워드만 겹치는 발화는 other_mentions로 보내거나 제외한다.",
]

APPLICATION_POLICY_PROMPT = "\n".join(
    [
        "당신은 Whylog 프로젝트의 수석 분석가입니다.",
        "아래 STT 데이터를 바탕으로 'Whylog 3대 운영 정책'을",
        "엄격히 준수하여 분석을 수행하세요.",
        "",
        "------------------------------------------------------------",
        "[정책 1: 회의 종료 시 적용사항 생성 정책]",
        "1. 명확한 선택지 수렴, 기술/구조/정책/프로세스 변경 합의,",
        '   "~하기로 한다"는 명시적 결론이 있을 때만 적용사항을 생성한다.',
        "2. 단순 의견 교환, 결론 없는 토론, 감정 표현, 농담/잡담은",
        "   절대 생성하지 않는다.",
        "3. 실제 반영/실행 단위가 다르면 별도 적용사항으로 분리하고,",
        "   모호한 경우 생성하지 말고 보류 처리한다.",
        "",
        "[정책 2: 적용사항 근거 정책]",
        "1. 기술적/비용적/운영적 이유, 리스크 판단,",
        "   성능/확장성/안정성 관련 판단을 포함한다.",
        "2. 감정적 반응, 농담, 단순 동의는 제외한다.",
        "3. 동일 의미 발언은 병합하고 구어체는 명사형 근거로 정제한다.",
        "4. 반드시 '1 근거 = 1 문장' 원칙을 준수한다.",
        "   (리스트의 각 항목은 한 문장이어야 함)",
        *NOUN_ENDING_REASON_RULE,
        "",
        "[정책 3: 적용사항 타임라인 정책]",
        "1. 이슈 제기 -> 대안 논의 -> 적용 합의 시점 순으로 구성한다.",
        "2. 적용사항과 직접 관련된 흐름만 표시하고 단순 발언은 제외한다.",
        "3. 반드시 실제 발화(Utterance) 원문과 member_id를 포함한다.",
        "   member_id는 STT 데이터의 member_id 값을 사용하고, 없으면 null로 둔다.",
        "4. timeline의 content는 간략한 한 문장으로 작성한다.",
        *CONCRETE_APPLICATION_RULE,
        *TIMELINE_EVIDENCE_ALIGNMENT_RULE,
        "------------------------------------------------------------",
        "",
        "[출력 JSON 구조]:",
        "{",
        '  "overall_analysis": {',
        '    "meeting_info": { "title": "...", "purpose": "...", "duration": "..." },',
        '    "topics": ["논의된 모든 주제 리스트"],',
        '    "core_context": ["프로젝트 배경 및 제약 사항"],',
        '    "application_titles": ["생성된 모든 적용사항 타이틀 리스트"],',
        '    "application_reasons": [',
        '      "모든 적용사항의 근거를 통합하여 정렬한 리스트"',
        "    ]",
        "  },",
        '  "applications": [',
        "    {",
        '      "application_title": "커밋과 연결될 적용사항 명칭",',
        '      "application_reasons": ["해당 적용사항의 근거 (1근거=1문장)"],',
        '      "timeline": [',
        "        {",
        '          "timestamp": "...",',
        (
            f'          "step": "{TimelineStep.ISSUE.value}/'
            f'{TimelineStep.DISCUSSION.value}/{TimelineStep.AGREEMENT.value}",'
        ),
        '          "member_id": 1,',
        '          "content": "간략 요약 한 문장",',
        '          "utterance": "실제 발화 원문"',
        "        }",
        "      ]",
        "    }",
        "  ],",
        '  "other_mentions": ["적용사항으로 확정되지 않은 기술적 제언 및 미래 과제"]',
        "}",
    ]
)

SUMMARY_ONLY_PROMPT = "\n".join(
    [
        "당신은 Whylog 프로젝트의 수석 분석가입니다.",
        "아래 STT 데이터를 바탕으로 회의 요약 정보만 구조화하세요.",
        "",
        "[핵심 규칙]",
        "1. 실제 회의 내용 기반으로만 작성하고 추측하지 않는다.",
        "2. application_titles에는 실제 합의된 적용사항만 넣는다.",
        "3. application_reasons는 근거를 문장 단위로 정리하되",
        "   각 항목은 명사형 어미로 끝낸다.",
        "4. meeting_info.title은 회의 내용의 핵심 논의 대상/이슈/합의를",
        "   12~35자 내외의 구체적인 한국어 제목으로 작성한다.",
        "5. 'Whylog 프로젝트 회의', 'Whylog meeting', '기능 점검 회의',",
        "   '데이터 없음'처럼 서비스명이나 일반 명사만 있는 제목은 금지한다.",
        "6. title에는 가능하면 핵심 기능명, 오류명, 정책명, 연동 대상처럼",
        "   검색 가능한 고유 단어를 포함한다.",
        *NOUN_ENDING_REASON_RULE,
        "",
        "[출력 JSON 구조]",
        "{",
        '  "overall_analysis": {',
        '    "meeting_info": { "title": "...", "purpose": "...", "duration": "..." },',
        '    "topics": ["논의된 모든 주제 리스트"],',
        '    "core_context": ["프로젝트 배경 및 제약 사항"],',
        '    "application_titles": ["생성된 모든 적용사항 타이틀 리스트"],',
        '    "application_reasons": [',
        '      "모든 적용사항의 근거를 통합하여 정렬한 리스트"',
        "    ]",
        "  }",
        "}",
    ]
)

APPLICATIONS_ONLY_PROMPT = "\n".join(
    [
        "당신은 Whylog 프로젝트의 수석 분석가입니다.",
        "아래 STT 데이터를 바탕으로 적용사항 목록만 구조화하세요.",
        "",
        "[정책 1: 적용사항 생성]",
        "1. 명확한 결론이 있는 합의만 applications에 포함한다.",
        "2. 결론 없는 토론/잡담은 제외한다.",
        "3. 실제 반영/실행 단위가 다르면 별도 적용사항으로 분리한다.",
        "",
        "[정책 2: 근거]",
        "1. 기술/비용/운영/리스크/성능 관련 근거를 우선한다.",
        "2. 1 근거 = 1 문장 원칙을 지킨다.",
        *NOUN_ENDING_REASON_RULE,
        "",
        "[정책 3: 타임라인]",
        (
            f"1. {TimelineStep.ISSUE.value} -> {TimelineStep.DISCUSSION.value} "
            f"-> {TimelineStep.AGREEMENT.value} 순서를 따른다."
        ),
        "2. 실제 발화 원문과 member_id를 포함한다.",
        "3. member_id는 STT 데이터의 member_id 값을 사용하고, 없으면 null로 둔다.",
        "4. content는 간략한 한 문장으로 작성한다.",
        *CONCRETE_APPLICATION_RULE,
        *TIMELINE_EVIDENCE_ALIGNMENT_RULE,
        "",
        "[출력 JSON 구조]",
        "{",
        '  "applications": [',
        "    {",
        '      "application_title": "커밋과 연결될 적용사항 명칭",',
        '      "application_reasons": ["해당 적용사항의 근거 (1근거=1문장)"],',
        '      "timeline": [',
        "        {",
        '          "timestamp": "...",',
        (
            f'          "step": "{TimelineStep.ISSUE.value}/'
            f'{TimelineStep.DISCUSSION.value}/{TimelineStep.AGREEMENT.value}",'
        ),
        '          "member_id": 1,',
        '          "content": "간략 요약 한 문장",',
        '          "utterance": "실제 발화 원문"',
        "        }",
        "      ]",
        "    }",
        "  ],",
        '  "other_mentions": ["적용사항으로 확정되지 않은 기술적 제언 및 미래 과제"]',
        "}",
    ]
)


class _SummaryOnlyResponse(BaseModel):
    overall_analysis: MeetingAnalysis = Field(default_factory=MeetingAnalysis)


class _ApplicationsOnlyResponse(BaseModel):
    applications: list[Application] = Field(default_factory=list)
    other_mentions: list[str] = Field(default_factory=list)


def _clean_json_response(result_text: str) -> str:
    # 모델 응답에서 코드블록 마크다운을 제거
    cleaned = re.sub(r"```json|```", "", result_text or "", flags=re.IGNORECASE)
    return cleaned.strip()


def _build_prompt(stt_data: list[dict]) -> str:
    # 전체 회의 분석 결과 추출용 프롬프트 생성
    stt_json = json.dumps(stt_data, ensure_ascii=False)
    return (
        f"{APPLICATION_POLICY_PROMPT}\n\n"
        f"[STT 데이터]: {stt_json}\n\n"
        "반드시 JSON만 반환하세요. 마크다운 코드블록은 사용하지 마세요."
    )


def _build_summary_prompt(stt_data: list[dict]) -> str:
    # 회의 요약(overall_analysis) 전용 프롬프트 생성
    stt_json = json.dumps(stt_data, ensure_ascii=False)
    return (
        f"{SUMMARY_ONLY_PROMPT}\n\n"
        f"[STT 데이터]: {stt_json}\n\n"
        "반드시 JSON만 반환하세요. 마크다운 코드블록은 사용하지 마세요."
    )


def _build_applications_prompt(stt_data: list[dict]) -> str:
    # applications/other_mentions 전용 프롬프트 생성
    stt_json = json.dumps(stt_data, ensure_ascii=False)
    return (
        f"{APPLICATIONS_ONLY_PROMPT}\n\n"
        f"[STT 데이터]: {stt_json}\n\n"
        "반드시 JSON만 반환하세요. 마크다운 코드블록은 사용하지 마세요."
    )


async def _request_gemini_json(
    prompt: str,
    failure_context: str,
) -> dict:
    # Gemini 호출 + JSON 응답 파싱을 공통 처리
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise AppServiceError(
            "GEMINI_API_KEY가 설정되지 않았습니다.",
            status_code=500,
        )

    client = genai.Client(api_key=api_key)
    try:
        response = await generate_content_with_retry(
            client,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
            timeout=90.0,
            operation_name=failure_context,
        )
    except Exception as e:
        raise AppServiceError(
            f"{failure_context} 요청 실패: {e}",
            status_code=502,
        ) from e

    cleaned_json = _clean_json_response(response.text or "")
    if not cleaned_json:
        raise AppServiceError(
            f"{failure_context} 응답이 비어 있습니다.",
            status_code=502,
        )

    try:
        parsed = json.loads(cleaned_json)
    except Exception as e:
        raise AppServiceError(
            f"{failure_context} JSON 파싱 실패: {e}",
            status_code=502,
        ) from e

    if not isinstance(parsed, dict):
        raise AppServiceError(
            f"{failure_context} 응답 형식이 객체(JSON object)가 아닙니다.",
            status_code=502,
        )
    return parsed


def _parse_timestamp_to_seconds(value: str) -> int | None:
    # HH:MM:SS 문자열을 초 단위 정수로 변환
    parts = (value or "").strip().split(":")
    if len(parts) == 2:
        parts = ["0", *parts]
    if len(parts) != 3:
        return None
    try:
        hours, minutes, seconds = (int(part) for part in parts)
    except ValueError:
        return None
    return hours * 3600 + minutes * 60 + seconds


def _normalize_text(value: str) -> str:
    # 공백 정규화로 비교 안정성 확보
    return " ".join((value or "").split())


def _normalize_reason_to_noun_ending(value: str) -> str:
    # 자주 나오는 서술형 근거 종결을 명사형 근거 라벨로 보정
    normalized = _normalize_text(value).rstrip(".")
    if not normalized:
        return ""

    for pattern, replacement in REASON_NOUN_ENDING_PATTERNS:
        converted = pattern.sub(replacement, normalized).strip()
        if converted != normalized:
            return converted.rstrip(".")

    return normalized


def _normalize_application_reason_outputs(result: MeetingAnalysisResult) -> None:
    # applications와 overall의 근거 출력을 명사형 어미 기준으로 정리
    for application in result.applications:
        application.application_reasons = [
            normalized
            for reason in application.application_reasons
            if (normalized := _normalize_reason_to_noun_ending(reason))
        ]
    result.overall_analysis.application_reasons = [
        normalized
        for reason in result.overall_analysis.application_reasons
        if (normalized := _normalize_reason_to_noun_ending(reason))
    ]


def _normalize_overall_reason_outputs(overall_analysis: MeetingAnalysis) -> None:
    # summary_ready 단계의 근거도 동일한 명사형 형식으로 정리
    overall_analysis.application_reasons = [
        normalized
        for reason in overall_analysis.application_reasons
        if (normalized := _normalize_reason_to_noun_ending(reason))
    ]


def _is_generic_meeting_title(value: str) -> bool:
    # 서비스명/일반 명사만 있는 회의 제목인지 판단
    normalized = _normalize_text(value)
    if not normalized:
        return True
    if normalized in GENERIC_MEETING_TITLES:
        return True

    lowered = normalized.lower()
    if lowered in {title.lower() for title in GENERIC_MEETING_TITLES}:
        return True

    tokens = set(re.findall(r"[A-Za-z0-9가-힣_]+", lowered))
    if not tokens:
        return True

    generic_tokens = {token.lower() for token in GENERIC_MEETING_TITLE_KEYWORDS}
    return tokens.issubset(generic_tokens)


def _build_title_from_items(items: list[str]) -> str:
    # topics/application_titles에서 회의 제목 후보를 생성
    normalized_items = []
    seen: set[str] = set()
    for item in items:
        normalized = _normalize_text(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        normalized_items.append(normalized)

    if not normalized_items:
        return ""

    if len(normalized_items) == 1:
        base = normalized_items[0]
    else:
        base = f"{normalized_items[0]} 및 {normalized_items[1]}"

    suffixes = ("회의", "논의", "점검")
    if base.endswith(suffixes):
        return base
    return f"{base} 회의"


def _refine_meeting_title(overall_analysis: MeetingAnalysis) -> None:
    # LLM이 일반 제목을 반환하면 topics/application_titles 기반으로 보정
    current_title = overall_analysis.meeting_info.title
    if not _is_generic_meeting_title(current_title):
        return

    topic_title = _build_title_from_items(overall_analysis.topics)
    if topic_title:
        overall_analysis.meeting_info.title = topic_title
        return

    application_title = _build_title_from_items(overall_analysis.application_titles)
    if application_title:
        overall_analysis.meeting_info.title = application_title


def _is_ambiguous_utterance(value: str) -> bool:
    # 화자 추정에 쓰기 어려운 짧은/모호 발화 여부 판단
    normalized = _normalize_text(value).lower()
    if not normalized:
        return True
    if normalized in AMBIGUOUS_SHORT_UTTERANCES:
        return True
    return len(normalized) <= 3


def _match_score(utterance: str, segment_text: str) -> int:
    # utterance와 segment 텍스트의 유사도 점수 계산
    if not utterance or not segment_text:
        return 0
    if utterance == segment_text:
        return 4
    if utterance in segment_text:
        return 3
    if segment_text in utterance:
        return 2
    utterance_tokens = set(utterance.split())
    segment_tokens = set(segment_text.split())
    if utterance_tokens and segment_tokens and utterance_tokens & segment_tokens:
        return 1
    return 0


def _infer_timeline_member_id(
    utterance: str,
    timestamp: str,
    segments: list[TranscribeSegment],
) -> int | None:
    # timestamp + utterance 유사도로 timeline 발화의 member_id를 추정
    normalized_utterance = _normalize_text(utterance)
    target_seconds = _parse_timestamp_to_seconds(timestamp)
    timestamp_candidates: list[TranscribeSegment] = []
    if target_seconds is not None:
        for segment in segments:
            if segment.member_id is None:
                continue
            start_seconds = _parse_timestamp_to_seconds(segment.start_time)
            end_seconds = _parse_timestamp_to_seconds(segment.end_time)
            if start_seconds is None or end_seconds is None:
                continue
            if start_seconds <= target_seconds <= end_seconds:
                timestamp_candidates.append(segment)

        if len(timestamp_candidates) == 1:
            return timestamp_candidates[0].member_id

        if len(timestamp_candidates) > 1 and not _is_ambiguous_utterance(
            normalized_utterance
        ):
            scored = sorted(
                (
                    (
                        _match_score(
                            normalized_utterance,
                            _normalize_text(segment.text),
                        ),
                        segment,
                    )
                    for segment in timestamp_candidates
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            if scored and scored[0][0] > 0:
                return scored[0][1].member_id

    if _is_ambiguous_utterance(normalized_utterance):
        return None

    global_scored = sorted(
        (
            (
                _match_score(
                    normalized_utterance,
                    _normalize_text(segment.text),
                ),
                segment,
            )
            for segment in segments
            if segment.member_id is not None
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    if global_scored and global_scored[0][0] >= 3:
        return global_scored[0][1].member_id
    return None


def _normalize_timeline_member_ids(
    result: MeetingAnalysisResult,
    segments: list[TranscribeSegment],
) -> None:
    # LLM이 생성한 member_id를 전사 세그먼트 기준으로 검증/보정
    valid_member_ids = {
        segment.member_id for segment in segments if segment.member_id is not None
    }
    corrected_count = 0
    unresolved_count = 0

    for application in result.applications:
        for item in application.timeline:
            if item.member_id in valid_member_ids:
                continue

            inferred = None
            if valid_member_ids:
                inferred = _infer_timeline_member_id(
                    utterance=item.utterance,
                    timestamp=item.timestamp,
                    segments=segments,
                )
            if inferred is not None:
                item.member_id = inferred
                corrected_count += 1
            else:
                # 추정이 불가능하면 임의 member_id를 만들지 않고 None으로 둔다.
                item.member_id = None
                unresolved_count += 1

    if corrected_count > 0:
        logger.warning(
            "Normalized %s invalid timeline member_id values.",
            corrected_count,
        )
    if unresolved_count > 0:
        logger.warning(
            "Could not infer %s timeline member_id values.",
            unresolved_count,
        )


def _synchronize_overall_with_applications(result: MeetingAnalysisResult) -> None:
    # overall_analysis를 applications 기준으로 재정렬/동기화
    _normalize_application_reason_outputs(result)
    titles = [
        application.application_title.strip()
        for application in result.applications
        if application.application_title and application.application_title.strip()
    ]
    reasons: list[str] = []
    seen_reasons: set[str] = set()
    for application in result.applications:
        for reason in application.application_reasons:
            normalized_reason = " ".join((reason or "").split())
            if not normalized_reason:
                continue
            if normalized_reason in seen_reasons:
                continue
            seen_reasons.add(normalized_reason)
            reasons.append(normalized_reason)

    result.overall_analysis.application_titles = titles
    result.overall_analysis.application_reasons = reasons
    _refine_meeting_title(result.overall_analysis)


def build_analysis_result(
    overall_analysis: MeetingAnalysis,
    applications_result: MeetingAnalysisResult,
) -> MeetingAnalysisResult:
    # 요약 결과와 적용사항 결과를 하나의 최종 DTO로 결합
    result = MeetingAnalysisResult(
        overall_analysis=overall_analysis,
        applications=applications_result.applications,
        other_mentions=applications_result.other_mentions,
    )
    _synchronize_overall_with_applications(result)
    return result


async def extract_meeting_analysis(
    segments: list[TranscribeSegment],
) -> MeetingAnalysisResult:
    # 요약 + 적용사항 추출을 순차 실행해 최종 결과 반환
    overall_analysis = await extract_overall_analysis(segments)
    applications_result = await extract_applications_only(segments)
    return build_analysis_result(overall_analysis, applications_result)


async def extract_overall_analysis(
    segments: list[TranscribeSegment],
) -> MeetingAnalysis:
    # 회의 요약(overall_analysis)만 추출
    stt_data = [segment.model_dump(mode="json") for segment in segments]
    prompt = _build_summary_prompt(stt_data)
    parsed = await _request_gemini_json(prompt, "Gemini 회의 요약 추출")

    try:
        result = _SummaryOnlyResponse.model_validate(parsed)
        _normalize_overall_reason_outputs(result.overall_analysis)
        _refine_meeting_title(result.overall_analysis)
        return result.overall_analysis
    except Exception as e:
        raise AppServiceError(
            f"Gemini 회의 요약 JSON 파싱/검증 실패: {e}",
            status_code=502,
        ) from e


async def extract_applications_only(
    segments: list[TranscribeSegment],
) -> MeetingAnalysisResult:
    # applications/other_mentions만 추출 후 member_id 보정
    stt_data = [segment.model_dump(mode="json") for segment in segments]
    prompt = _build_applications_prompt(stt_data)
    parsed = await _request_gemini_json(prompt, "Gemini 적용사항 추출")

    try:
        applications_result = _ApplicationsOnlyResponse.model_validate(parsed)
        result = MeetingAnalysisResult(
            applications=applications_result.applications,
            other_mentions=applications_result.other_mentions,
        )
        _normalize_application_reason_outputs(result)
        _normalize_timeline_member_ids(result, segments)
        return result
    except Exception as e:
        raise AppServiceError(
            f"Gemini 적용사항 JSON 파싱/검증 실패: {e}",
            status_code=502,
        ) from e
