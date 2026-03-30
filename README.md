# FastAPI Project

## 설명
WhyLog의 FastAPI 레포지토리입니다

## 사전 준비

[uv](https://docs.astral.sh/uv/getting-started/installation/) 설치가 필요합니다.

```bash
# Mac / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

> **uv란?** pip + venv를 대체하는 패키지 매니저입니다. 기존에는 가상환경 생성 → 활성화 → pip install을 따로 해야 했지만, uv는 이를 자동으로 처리합니다. 또한 `uv.lock` 파일로 팀원 모두가 동일한 패키지 버전을 사용할 수 있습니다.

## 로컬 환경 설정

```bash
# 의존성 설치 (가상환경 생성 + 패키지 설치 자동)
uv sync
```

| 기존 (pip) | uv |
|---|---|
| `python3 -m venv .venv` | (자동) |
| `source .venv/bin/activate` | (자동) |
| `pip install --upgrade pip` | (자동) |
| `pip install -r requirements.txt` | `uv sync` |

## 서버 실행

```bash
uv run uvicorn main:app --reload
```

`uv run`은 가상환경을 직접 활성화하지 않아도 자동으로 인식합니다.

서버 실행 후 `http://127.0.0.1:8000` 또는 `http://127.0.0.1:8000/health`로 확인할 수 있습니다.

## pre-commit 설정 (최초 1회)

커밋 전 자동으로 코드 린트/포맷을 실행합니다.

```bash
uv tool install pre-commit
pre-commit install
```

이후 `git commit` 시 자동으로 실행됩니다. 별도로 수동 실행하려면:

```bash
pre-commit run --all-files
```

## 패키지 추가

```bash
uv add 패키지명           # 프로덕션 의존성
uv add --dev 패키지명     # 개발 의존성 (lint, test 등)
```

## Docker 이미지 빌드

```bash
docker build -t fastapi-app .
docker run --rm -p 8000:8000 fastapi-app
```

이미지를 빌드 후, 컨테이너를 실행합니다.
컨테이너 실행 후 `http://localhost:8000`로 확인할 수 있습니다.

## DockerHub 푸시 스크립트

```bash
chmod +x build_and_push.sh
./build_and_push.sh
```

위 스크립트는 `whylog/whylog-fastapi:latest` 로 이미지를 빌드하고 DockerHub로 푸시합니다.
푸시 전에 `docker login` 이 되어 있어야 합니다.
