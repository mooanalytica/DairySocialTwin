from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cowtrack.config import ContractError
from cowtrack.qa.nvenc import NvencVideoWriter


class _PartialStdin:
    def __init__(self, max_write: int) -> None:
        self.max_write = max_write
        self.payload = bytearray()
        self.closed = False

    def write(self, value: memoryview) -> int:
        if self.closed:
            raise BrokenPipeError("closed")
        count = min(len(value), self.max_write)
        self.payload.extend(value[:count])
        return count

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    def __init__(
        self,
        command: list[str],
        *,
        max_write: int,
        returncode: int,
        stderr: bytes,
    ) -> None:
        self.command = command
        self.stdin = _PartialStdin(max_write)
        self.stderr = io.BytesIO(stderr)
        self.returncode: int | None = None
        self._final_returncode = returncode
        self._output_path = Path(command[-1])

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.returncode = self._final_returncode
        if self.returncode == 0:
            self._output_path.write_bytes(b"fake mp4")
        return self.returncode

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


class _Factory:
    def __init__(
        self,
        *,
        max_write: int = 5,
        returncode: int = 0,
        stderr: bytes = b"",
    ) -> None:
        self.max_write = max_write
        self.returncode = returncode
        self.stderr = stderr
        self.calls: list[tuple[list[str], dict[str, Any], _FakeProcess]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> _FakeProcess:
        process = _FakeProcess(
            command,
            max_write=self.max_write,
            returncode=self.returncode,
            stderr=self.stderr,
        )
        self.calls.append((command, kwargs, process))
        return process


def test_nvenc_writer_handles_partial_pipe_writes_and_publishes_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    factory = _Factory(max_write=5)
    output = tmp_path / "review.mp4"
    frame = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)

    with NvencVideoWriter(
        output,
        width=2,
        height=2,
        fps=30,
        expected_frame_count=1,
        popen_factory=factory,
    ) as writer:
        writer.write(frame)

    assert output.read_bytes() == b"fake mp4"
    assert writer.published is True
    assert bytes(factory.calls[0][2].stdin.payload) == frame.tobytes()
    command, kwargs, _ = factory.calls[0]
    assert "h264_nvenc" in command
    assert command[command.index("-gpu") + 1] == "0"
    assert "libx264" not in command
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "1"


def test_nvenc_writer_refuses_wrong_gpu_mask_before_spawning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    factory = _Factory()

    with pytest.raises(ContractError, match="exactly equal to '1'"):
        NvencVideoWriter(
            tmp_path / "review.mp4",
            width=2,
            height=2,
            fps=30,
            expected_frame_count=1,
            popen_factory=factory,
        )
    assert factory.calls == []


def test_nvenc_writer_refuses_underfilled_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    output = tmp_path / "review.mp4"
    writer = NvencVideoWriter(
        output,
        width=2,
        height=2,
        fps=30,
        expected_frame_count=2,
        popen_factory=_Factory(),
    )
    writer.write(np.zeros((2, 2, 3), dtype=np.uint8))

    with pytest.raises(ContractError, match="unexpected frame count"):
        writer.close()
    assert not output.exists()


def test_nvenc_writer_surfaces_encoder_failure_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    output = tmp_path / "review.mp4"
    writer = NvencVideoWriter(
        output,
        width=2,
        height=2,
        fps=30,
        expected_frame_count=1,
        popen_factory=_Factory(returncode=2, stderr=b"NVENC unavailable"),
    )
    writer.write(np.zeros((2, 2, 3), dtype=np.uint8))

    with pytest.raises(ContractError, match="NVENC encoding failed.+NVENC unavailable"):
        writer.close()
    assert not output.exists()
