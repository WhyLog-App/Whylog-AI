# Commit Matching Evaluation

## 목적

회의 적용사항과 커밋의 실제 정답 관계가 운영 DB에 직접 저장되어 있지 않기 때문에,
실제 커밋을 기준으로 만든 평가용 적용사항을 별도 golden dataset으로 관리한다.

이 평가셋은 다음 작업에 사용한다.

- `/api/commit/match` 결과가 기대 커밋을 상위 K개 안에 포함하는지 확인
- 70점 이상 false positive 후보가 생기는지 확인
- 아직 구현되지 않은 적용사항이 빈 추천 목록으로 반환되는지 확인
- 점수식, threshold, keyword/context 정책 변경 전후의 품질 비교

## 평가셋 위치

```bash
tests/fixtures/commit_matching_golden_cases.json
```

각 case는 실제 Whylog-AI 커밋 해시를 기준으로 구성한다.

- `application_title`: 시연용/평가용 가짜 회의 적용사항 제목
- `application_reasons`: 적용사항 근거
- `expected_commit_hashes`: 반드시 추천되어야 하는 실제 커밋 해시
- `accepted_commit_hashes`: 정답은 아니지만 false positive로 세지 않을 허용 커밋 해시
- `distractor_commit_hashes`: 헷갈리지만 정답이 아닌 커밋 해시
- `should_match`: 추천이 있어야 하는지 여부
- `tags`: 분석용 태그

## 평가 실행

먼저 실제 또는 로컬 `/api/commit/match` 응답을 JSON 파일로 저장한다.
응답은 FastAPI 공통 응답 wrapper가 있는 형태와 result 본문만 있는 형태를 모두 지원한다.

```bash
uv run python scripts/evaluate_commit_matching.py \
  --cases tests/fixtures/commit_matching_golden_cases.json \
  --response /path/to/commit-match-response.json \
  --top-k 5
```

JSON 요약이 필요하면 다음 옵션을 사용한다.

```bash
uv run python scripts/evaluate_commit_matching.py \
  --response /path/to/commit-match-response.json \
  --json
```

CI나 회귀 검증에서 실패 시 non-zero exit code가 필요하면 다음 옵션을 추가한다.

```bash
uv run python scripts/evaluate_commit_matching.py \
  --response /path/to/commit-match-response.json \
  --fail-on-failure
```

정답 커밋이 포함되어도 70점 이상 오탐 커밋이 함께 추천되면 실패로 보고 싶을 때는
다음 옵션을 함께 사용한다.

```bash
uv run python scripts/evaluate_commit_matching.py \
  --response /path/to/commit-match-response.json \
  --fail-on-false-positive \
  --fail-on-failure
```

기본 high-confidence 기준은 70점이며, 필요하면 조정할 수 있다.

```bash
uv run python scripts/evaluate_commit_matching.py \
  --response /path/to/commit-match-response.json \
  --confidence-threshold 75
```

## 지표

- `recall_at_k`: 정답 커밋이 상위 K개 추천 안에 포함된 비율
- `precision_at_k`: 추천된 커밋 중 정답 또는 허용 커밋 비율
- `mean_reciprocal_rank`: 정답 커밋이 몇 번째에 나왔는지 반영한 순위 지표
- `no_match_accuracy`: 추천이 없어야 하는 적용사항에서 빈 추천을 반환한 비율
- `false_positive_count`: 정답 또는 허용 커밋이 아닌 추천 개수
- `high_confidence_false_positive_count`: confidence threshold 이상인 오탐 추천 개수
- `distractor_hit_count`: hard negative로 지정한 distractor 커밋이 추천된 개수

`accepted_commit_hashes`는 precision 계산과 false positive 판정에서만 사용한다.
해당 커밋만 추천되고 `expected_commit_hashes`가 빠진 경우에는 해당 case를 실패로 본다.

`distractor_commit_hashes`는 정답이 아니지만 헷갈리기 쉬운 커밋을 명시하는 필드다.
추천 결과에 distractor가 포함되면 `distractor_hit_hashes`와 `distractor_hit_count`로
별도 집계된다.

기본 pass 기준은 다음과 같다.

- `should_match=true`: expected commit이 top-k 안에 있으면 pass
- `should_match=false`: confidence threshold 이상 추천이 없으면 pass
- `--fail-on-false-positive`: match case에서도 threshold 이상 오탐이 있으면 fail

## 고도화 흐름

1. 실제 커밋을 기준으로 golden case를 추가한다.
2. 현재 `/api/commit/match` 결과를 저장하고 평가 스크립트로 baseline을 기록한다.
3. 일반어 제거, intent keyword 보정, LLM rerank 같은 정책을 한 번에 하나씩 적용한다.
4. 같은 response fixture 또는 같은 운영 데이터로 지표를 비교한다.
5. 시연용 데이터는 easy case, hard negative, no-match case를 모두 포함하도록 구성한다.
