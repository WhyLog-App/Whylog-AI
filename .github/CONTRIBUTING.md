# 네이밍 규칙 및 협업 가이드

## 브랜치 명

```
태그/깃허브닉네임-기능#이슈번호
```

| 예시 |
|---|
| `feat/wantkdd-deepgram-transcribe#3` |
| `fix/wantkdd-api-key-loading#7` |

---

## 커밋 메시지

```
[태그/#이슈번호] - 메시지
```

| 예시 |
|---|
| `[feat/#3] - Deepgram 음성 전사 API 구현` |
| `[fix/#7] - API 키 로딩 순서 오류 수정` |

### 커밋 유형

| 태그 | 설명 |
|---|---|
| `feat` | 새로운 기능 추가 또는 기존 기능 개선 |
| `fix` | 버그 수정 |
| `refactor` | 코드 리팩토링 (기능 변화 없이 구조 개선) |
| `doc` | 문서 작업 (README 등) |
| `test` | 테스트 코드 추가 또는 수정 |
| `perform` | 성능 개선 |
| `style` | 코드 스타일 변경 (포맷, 들여쓰기 등) – 기능 변화 없음 |
| `comment` | 주석 수정, 추가 |
| `merge` | 브랜치 병합 |
| `deps` | 패키지 의존성 추가 · 변경 · 삭제 |
| `chore` | 기타 개발 세팅 등 잡다한 것 |

---

## 코드 네이밍 규칙

| 대상 | 규칙 | 예시 |
|---|---|---|
| 파일명 | snake_case | `transcribe.py`, `audio_utils.py` |
| 폴더명 | snake_case | `routers/`, `services/` |
| 함수 · 변수 | snake_case | `transcribe_audio`, `api_key` |
| 클래스 | PascalCase | `TranscribeRequest` |
| 상수 | UPPER_SNAKE_CASE | `DEEPGRAM_URL`, `CONTENT_TYPE_MAP` |
| API 경로 | kebab-case | `/api/transcribe-audio` |

---

## 브랜치 전략

```
main        ← 배포 브랜치
└── develop ← 통합 브랜치
    └── feat/... fix/... 등 작업 브랜치
```

- 작업은 항상 `develop` 기반으로 브랜치를 따서 진행
- PR은 `develop`으로 올림
- `main` 머지는 배포 시점에만
