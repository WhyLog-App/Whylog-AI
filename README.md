# FastAPI Project

## 설명
WhyLog의 FastAPI 레포지토리입니다

## 로컬 가상환경 설정

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --reload
```

서버 실행 후 `http://127.0.0.1:8000` 또는 `http://127.0.0.1:8000/health`로 확인할 수 있습니다.

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
