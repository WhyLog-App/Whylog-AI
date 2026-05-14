from enum import StrEnum


class RunStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TranscribeRunPhase(StrEnum):
    QUEUED = "queued"
    TRANSCRIBING = "transcribing"
    TRANSCRIPT_READY = "transcript_ready"
    SUMMARY_READY = "summary_ready"
    APPLICATIONS_READY = "applications_ready"
    FAILED = "failed"


class CommitAnalyzeRunPhase(StrEnum):
    QUEUED = "queued"
    SUMMARIZING = "summarizing"
    SUMMARY_READY = "summary_ready"
    EMBEDDING = "embedding"
    EMBEDDING_READY = "embedding_ready"
    FAILED = "failed"


class TimelineStep(StrEnum):
    ISSUE = "이슈제기"
    DISCUSSION = "대안논의"
    AGREEMENT = "적용합의"


class MatchStatus(StrEnum):
    APPLIED = "APPLIED"
    PARTIAL = "PARTIAL"
    UNAPPLIED = "UNAPPLIED"


class CommitChangeDirection(StrEnum):
    ADD = "add"
    REMOVE = "remove"
    MODIFY = "modify"
    MIGRATE = "migrate"


class CommitType(StrEnum):
    FEAT = "feat"
    FIX = "fix"
    DOCS = "docs"
    REFACTOR = "refactor"
    TEST = "test"
    BUILD = "build"
    CHORE = "chore"
