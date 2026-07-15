from qwen_hotword.modeling.ctc_tap import (
    qwen3_asr_audio_output_length,
    qwen3_asr_audio_output_lengths,
    tensor_shape,
    tensor_values,
    validate_packed_ctc_tap,
)


class FakeTensor:
    def __init__(self, values: list[int], shape: tuple[int, ...]) -> None:
        self.values = values
        self.shape = shape

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def tolist(self) -> list[int]:
        return self.values


def test_qwen_audio_output_length_matches_100_frame_chunks() -> None:
    assert qwen3_asr_audio_output_length(0) == 0
    assert qwen3_asr_audio_output_length(50) == 7
    assert qwen3_asr_audio_output_length(100) == 13
    assert qwen3_asr_audio_output_length(101) == 14
    assert qwen3_asr_audio_output_length(200) == 26


def test_qwen_audio_output_lengths_preserve_batch_items() -> None:
    assert qwen3_asr_audio_output_lengths([100, 200]) == [13, 26]


def test_tensor_helpers_accept_tensor_like_values() -> None:
    tensor = FakeTensor([100, 200], (2,))
    assert tensor_shape(tensor) == [2]
    assert tensor_values(tensor) == [100, 200]


def test_validate_packed_ctc_tap_accepts_expected_shape() -> None:
    output_lengths, errors = validate_packed_ctc_tap(
        feature_lengths=[100, 200],
        tap_shape=[39, 1024],
    )
    assert output_lengths == [13, 26]
    assert errors == []


def test_validate_packed_ctc_tap_reports_shape_mismatches() -> None:
    _, errors = validate_packed_ctc_tap(
        feature_lengths=[100, 200],
        tap_shape=[40, 2048],
    )
    assert errors == [
        "packed row count mismatch: actual=40, expected=39",
        "hidden size mismatch: actual=2048, expected=1024",
    ]
