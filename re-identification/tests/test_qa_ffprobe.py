from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from cowtrack.config import ContractError
from cowtrack.qa.ffprobe import QaMp4Metadata, validate_qa_mp4


def _valid_stream() -> dict[str, Any]:
    return {
        "index": 0,
        "codec_type": "video",
        "codec_name": "h264",
        "width": 1920,
        "height": 1080,
        "pix_fmt": "yuv420p",
        "avg_frame_rate": "30000/1001",
        "nb_frames": "181",
    }


class FakeRunner:
    def __init__(
        self,
        payload: object,
        *,
        returncode: int = 0,
        stderr: str = "",
    ) -> None:
        self.payload = payload
        self.returncode = returncode
        self.stderr = stderr
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, command: list[str], **kwargs: Any):
        self.calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            self.returncode,
            stdout=json.dumps(self.payload),
            stderr=self.stderr,
        )


def _placeholder(tmp_path: Path) -> Path:
    path = tmp_path / "review.mp4"
    path.write_bytes(b"not read by the injected runner")
    return path


def test_validate_qa_mp4_accepts_exact_contract_and_never_uses_shell(
    tmp_path: Path,
) -> None:
    path = _placeholder(tmp_path)
    runner = FakeRunner({"streams": [_valid_stream()]})

    metadata = validate_qa_mp4(
        path,
        expected_frame_rate=Fraction(30000, 1001),
        expected_frame_count=181,
        ffprobe_binary="custom-ffprobe",
        run=runner,
    )

    assert metadata == QaMp4Metadata(
        path=path,
        stream_index=0,
        codec_name="h264",
        width=1920,
        height=1080,
        average_frame_rate=Fraction(30000, 1001),
        frame_count=181,
    )
    assert len(runner.calls) == 1
    command, kwargs = runner.calls[0]
    assert command[0] == "custom-ffprobe"
    assert command[-1] == str(path)
    assert "-show_streams" in command
    assert kwargs["shell"] is False
    assert kwargs["check"] is False
    assert kwargs["text"] is True


@pytest.mark.parametrize("codec_type", ["audio", "subtitle", "data"])
def test_validate_qa_mp4_rejects_forbidden_stream_types(
    tmp_path: Path, codec_type: str
) -> None:
    stream = {"index": 1, "codec_type": codec_type, "codec_name": "irrelevant"}
    runner = FakeRunner({"streams": [_valid_stream(), stream]})

    with pytest.raises(ContractError, match="forbidden stream types"):
        validate_qa_mp4(
            _placeholder(tmp_path),
            expected_frame_rate=Fraction(30000, 1001),
            expected_frame_count=181,
            run=runner,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("codec_name", "hevc", "codec must be h264"),
        ("width", 1280, "geometry must be 1920x1080"),
        ("height", 720, "geometry must be 1920x1080"),
        ("pix_fmt", "yuv444p", "pixel format must be yuv420p"),
        ("avg_frame_rate", "30/1", "avg_frame_rate mismatch"),
        ("nb_frames", "180", "nb_frames mismatch"),
        ("nb_frames", "N/A", "nb_frames must be a positive integer"),
    ],
)
def test_validate_qa_mp4_rejects_video_contract_mismatch(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    stream = _valid_stream()
    stream[field] = value

    with pytest.raises(ContractError, match=message):
        validate_qa_mp4(
            _placeholder(tmp_path),
            expected_frame_rate=Fraction(30000, 1001),
            expected_frame_count=181,
            run=FakeRunner({"streams": [stream]}),
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"tags": {"rotate": "0"}},
        {"rotation": 0},
        {"side_data_list": [{"side_data_type": "Display Matrix"}]},
        {"side_data_list": [{"displaymatrix": "00000000"}]},
    ],
)
def test_validate_qa_mp4_rejects_all_rotation_or_display_matrix_forms(
    tmp_path: Path, metadata: dict[str, Any]
) -> None:
    stream = _valid_stream()
    stream.update(metadata)

    with pytest.raises(ContractError, match="rotation/display matrix"):
        validate_qa_mp4(
            _placeholder(tmp_path),
            expected_frame_rate=Fraction(30000, 1001),
            expected_frame_count=181,
            run=FakeRunner({"streams": [stream]}),
        )


@pytest.mark.parametrize("streams", [[], [_valid_stream(), _valid_stream()]])
def test_validate_qa_mp4_requires_exactly_one_stream(
    tmp_path: Path, streams: list[dict[str, Any]]
) -> None:
    with pytest.raises(ContractError, match="exactly one stream"):
        validate_qa_mp4(
            _placeholder(tmp_path),
            expected_frame_rate=Fraction(30000, 1001),
            expected_frame_count=181,
            run=FakeRunner({"streams": streams}),
        )


def test_validate_qa_mp4_reports_ffprobe_failure(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match=r"ffprobe failed \(2\).+decode error"):
        validate_qa_mp4(
            _placeholder(tmp_path),
            expected_frame_rate=Fraction(30000, 1001),
            expected_frame_count=181,
            run=FakeRunner({}, returncode=2, stderr="decode error"),
        )


def test_validate_qa_mp4_rejects_invalid_json(tmp_path: Path) -> None:
    def invalid_json_runner(command: list[str], **kwargs: Any):
        return subprocess.CompletedProcess(command, 0, stdout="not JSON", stderr="")

    with pytest.raises(ContractError, match="invalid ffprobe JSON"):
        validate_qa_mp4(
            _placeholder(tmp_path),
            expected_frame_rate=Fraction(30000, 1001),
            expected_frame_count=181,
            run=invalid_json_runner,
        )


def test_validate_qa_mp4_validates_expected_values_before_running(
    tmp_path: Path,
) -> None:
    runner = FakeRunner({"streams": [_valid_stream()]})
    with pytest.raises(ContractError, match="positive Fraction"):
        validate_qa_mp4(
            _placeholder(tmp_path),
            expected_frame_rate=30,  # type: ignore[arg-type]
            expected_frame_count=181,
            run=runner,
        )
    assert runner.calls == []
