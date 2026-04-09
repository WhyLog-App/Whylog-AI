import logging
import re
from pathlib import PurePosixPath

from app.domains.commit.schemas import ChangedFile

logger = logging.getLogger(__name__)

# 의존성 lock 파일 — 자동 생성되며 의도 분석에 노이즈
LOCK_FILES = {
    "uv.lock",
    "poetry.lock",
    "Pipfile.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Cargo.lock",
    "Gemfile.lock",
    "composer.lock",
    "go.sum",
    "mix.lock",
    "flake.lock",
    "Podfile.lock",
}

# 자동 생성·도구 스크립트 — 프로젝트 의도와 무관
GENERATED_FILES = {
    ".DS_Store",
    "Thumbs.db",
    "mvnw",
    "mvnw.cmd",
    "gradlew",
    "gradlew.bat",
    "HELP.md",
    ".eslintcache",
}

# 자동 생성·빌드 산출물·바이너리 확장자
NOISE_EXTENSIONS = {
    ".min.js",
    ".min.css",
    ".map",
    ".pyc",
    ".pyo",
    ".class",
    ".o",
    ".so",
    ".dll",
    ".exe",
    ".dylib",
    ".jar",
    ".war",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".svg",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".mp3",
    ".mp4",
    ".wav",
    ".log",
    ".tsbuildinfo",
    ".hprof",
    ".keystore",
    ".jks",
}

# 자동 생성·빌드·캐시 디렉토리
NOISE_DIR_PARTS = {
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".nuxt",
    ".cache",
    ".venv",
    "venv",
    "target",
    "vendor",
    ".gradle",
    ".idea",
    ".vscode",
    "coverage",
    ".nyc_output",
    ".turbo",
    ".parcel-cache",
    "Pods",
    ".expo",
    ".expo-shared",
}

# diff 본문이 바이너리임을 나타내는 패턴
BINARY_DIFF_PATTERN = re.compile(r"Binary files? .* differ", re.IGNORECASE)


def _is_noise(file_name: str, changed_code: str) -> bool:
    path = PurePosixPath(file_name)

    if path.name in LOCK_FILES or path.name in GENERATED_FILES:
        return True

    name_lower = path.name.lower()
    for ext in NOISE_EXTENSIONS:
        if name_lower.endswith(ext):
            return True

    if any(part in NOISE_DIR_PARTS for part in path.parts):
        return True

    if BINARY_DIFF_PATTERN.search(changed_code):
        return True

    return False


def filter_changed_files(changed_file_list: list[ChangedFile]) -> list[ChangedFile]:
    """LLM 분석에 노이즈가 되는 파일(lock, 바이너리, 자동 생성물)을 제거."""
    filtered: list[ChangedFile] = []
    skipped: list[str] = []
    for f in changed_file_list:
        if _is_noise(f.file_name, f.changed_code):
            skipped.append(f.file_name)
        else:
            filtered.append(f)
    if skipped:
        logger.info(
            "diff 필터링: %d개 파일 제외 (%s)", len(skipped), ", ".join(skipped)
        )
    return filtered
