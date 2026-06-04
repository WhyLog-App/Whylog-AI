from app.domains.transcribe.services.deepgram import (
    _extract_raw_segments,
    _response_diagnostics,
)


def test_extract_raw_segments_prefers_utterances():
    payload = {
        "results": {
            "utterances": [
                {
                    "speaker": 1,
                    "start": 1.0,
                    "end": 3.0,
                    "transcript": "회의 분석을 시작합니다",
                }
            ],
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "무시되어야 하는 전체 transcript",
                            "words": [
                                {
                                    "speaker": 0,
                                    "start": 1.0,
                                    "end": 1.2,
                                    "word": "무시",
                                }
                            ],
                        }
                    ]
                }
            ],
        }
    }

    segments = _extract_raw_segments(payload)

    assert segments == [
        {
            "speaker": 1,
            "start": 1.0,
            "end": 3.0,
            "text": "회의 분석을 시작합니다",
        }
    ]


def test_extract_raw_segments_falls_back_to_words_when_utterances_are_empty():
    payload = {
        "results": {
            "utterances": [],
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "회의 분석 결과를 확인합니다",
                            "words": [
                                {
                                    "speaker": 0,
                                    "start": 0.0,
                                    "end": 0.3,
                                    "punctuated_word": "회의",
                                },
                                {
                                    "speaker": 0,
                                    "start": 0.3,
                                    "end": 0.6,
                                    "punctuated_word": "분석",
                                },
                                {
                                    "speaker": 1,
                                    "start": 1.0,
                                    "end": 1.4,
                                    "word": "확인합니다",
                                },
                            ],
                        }
                    ]
                }
            ],
        }
    }

    segments = _extract_raw_segments(payload)

    assert segments == [
        {
            "speaker": 0,
            "start": 0.0,
            "end": 0.6,
            "text": "회의 분석",
        },
        {
            "speaker": 1,
            "start": 1.0,
            "end": 1.4,
            "text": "확인합니다",
        },
    ]


def test_extract_raw_segments_falls_back_to_channel_transcript():
    payload = {
        "results": {
            "utterances": [],
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "발화 단위는 없지만 전체 전사는 존재합니다",
                            "words": [],
                        }
                    ]
                }
            ],
        }
    }

    segments = _extract_raw_segments(payload)

    assert segments == [
        {
            "speaker": 0,
            "start": 0.0,
            "end": 0.0,
            "text": "발화 단위는 없지만 전체 전사는 존재합니다",
        }
    ]


def test_response_diagnostics_counts_deepgram_result_shapes():
    payload = {
        "results": {
            "utterances": [{"transcript": "하나"}],
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "하나 둘",
                            "words": [{"word": "하나"}, {"word": "둘"}],
                        }
                    ]
                }
            ],
        }
    }

    assert _response_diagnostics(payload) == {
        "utterance_count": 1,
        "channel_count": 1,
        "transcript_chars": 4,
        "word_count": 2,
    }
