# 초(float)를 HH:MM:SS 문자열로 변환
def format_time(seconds: float) -> str:
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# 연속된 같은 화자의 발화를 하나로 합침 (e.g. A-A-B-A → A-B-A)
def merge_consecutive_speaker_segments(segments: list[dict]) -> list[dict]:
    if not segments:
        return []

    merged = [segments[0].copy()]

    for cur in segments[1:]:
        prev = merged[-1]
        if str(prev["speaker"]) == str(cur["speaker"]):
            # 같은 화자면 텍스트 이어 붙이고 종료 시간 갱신
            prev["end"] = cur["end"]
            prev_text = (prev["text"] or "").strip()
            cur_text = (cur["text"] or "").strip()
            prev["text"] = f"{prev_text} {cur_text}".strip()
        else:
            merged.append(cur.copy())

    # 텍스트가 없는 세그먼트 제거
    return [seg for seg in merged if (seg.get("text") or "").strip()]
