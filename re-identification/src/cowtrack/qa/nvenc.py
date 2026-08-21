from __future__ import annotations

import math
import os
import subprocess
import threading
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Callable, IO, Self

import numpy as np

from cowtrack.config import ContractError


PopenFactory = Callable[..., subprocess.Popen[bytes]]


class NvencVideoWriter:
    """Strict raw-BGR to H.264/NVENC MP4 writer.

    The physical GPU is selected outside this class.  The fixed QA contract
    requires ``CUDA_VISIBLE_DEVICES=1`` and addresses that one exposed device
    as logical GPU zero.  This class deliberately has no software-encoder
    fallback.
    """

    _PRESETS = frozenset(f"p{number}" for number in range(1, 8))

    def __init__(
        self,
        output_path: str | os.PathLike[str],
        *,
        width: int,
        height: int,
        fps: int | float | Fraction,
        expected_frame_count: int,
        ffmpeg_binary: str = "ffmpeg",
        preset: str = "p4",
        cq: int = 21,
        logical_gpu: int = 0,
        pixel_format: str = "yuv420p",
        popen_factory: PopenFactory = subprocess.Popen,
    ) -> None:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            raise ContractError(
                "NVENC QA rendering requires CUDA_VISIBLE_DEVICES exactly equal to '1'"
            )

        self.width = self._positive_even_dimension(width, "width")
        self.height = self._positive_even_dimension(height, "height")
        self.fps, fps_argument = self._validated_fps(fps)
        if (
            isinstance(expected_frame_count, bool)
            or not isinstance(expected_frame_count, int)
            or expected_frame_count <= 0
        ):
            raise ContractError("NVENC expected_frame_count must be a positive integer")
        self.expected_frame_count = expected_frame_count
        if not isinstance(ffmpeg_binary, str) or not ffmpeg_binary.strip():
            raise ContractError("ffmpeg_binary must be a non-blank string")
        if "\x00" in ffmpeg_binary:
            raise ContractError("ffmpeg_binary cannot contain a NUL byte")
        if not isinstance(preset, str) or preset not in self._PRESETS:
            raise ContractError("NVENC preset must be one of p1 through p7")
        if isinstance(cq, bool) or not isinstance(cq, int) or not 0 <= cq <= 51:
            raise ContractError("NVENC cq must be an integer in [0, 51]")
        if (
            isinstance(logical_gpu, bool)
            or not isinstance(logical_gpu, int)
            or logical_gpu != 0
        ):
            raise ContractError(
                "NVENC logical_gpu must be 0 when CUDA_VISIBLE_DEVICES='1'"
            )
        if pixel_format != "yuv420p":
            raise ContractError("NVENC QA pixel_format must be exactly 'yuv420p'")
        if not callable(popen_factory):
            raise ContractError("popen_factory must be callable")

        self.output_path = Path(output_path)
        if self.output_path.suffix.lower() != ".mp4":
            raise ContractError("NVENC QA output_path must have an .mp4 suffix")
        if not self.output_path.name or self.output_path.name in {".", ".."}:
            raise ContractError("NVENC QA output_path must name a file")
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ContractError(
                f"cannot create NVENC output directory {self.output_path.parent}: {exc}"
            ) from exc

        token = uuid.uuid4().hex
        self.temporary_path = self.output_path.with_name(
            f".{self.output_path.stem}.{token}.part.mp4"
        )
        self.preset = preset
        self.cq = cq
        self.logical_gpu = logical_gpu
        self.pixel_format = pixel_format
        self._frame_count = 0
        self._closed = False
        self._published = False
        self._stderr_tail = bytearray()
        self._stderr_error: Exception | None = None
        self._stderr_thread: threading.Thread | None = None

        self.command = (
            ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            fps_argument,
            "-i",
            "pipe:0",
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-c:v",
            "h264_nvenc",
            "-gpu",
            str(self.logical_gpu),
            "-preset",
            self.preset,
            "-tune",
            "hq",
            "-rc:v",
            "vbr",
            "-cq:v",
            str(self.cq),
            "-b:v",
            "0",
            "-pix_fmt",
            self.pixel_format,
            "-movflags",
            "+faststart",
            "-frames:v",
            str(self.expected_frame_count),
            "-f",
            "mp4",
            str(self.temporary_path),
        )

        child_environment = os.environ.copy()
        try:
            self._process = popen_factory(
                list(self.command),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0,
                env=child_environment,
            )
        except Exception as exc:
            self._remove_temporary_file()
            raise ContractError(f"failed to start FFmpeg NVENC encoder: {exc}") from exc

        self._stdin: IO[bytes] | None = self._process.stdin
        self._stderr: IO[bytes] | None = self._process.stderr
        if self._stdin is None or self._stderr is None:
            self._terminate_process()
            self._closed = True
            self._remove_temporary_file()
            raise ContractError("FFmpeg NVENC process did not provide stdin/stderr pipes")
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name=f"nvenc-stderr-{token[:8]}",
            daemon=True,
        )
        self._stderr_thread.start()

    @staticmethod
    def _positive_even_dimension(value: int, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ContractError(f"NVENC {label} must be a positive integer")
        if value % 2 != 0:
            raise ContractError(f"NVENC {label} must be even for yuv420p output")
        return value

    @staticmethod
    def _validated_fps(value: int | float | Fraction) -> tuple[float, str]:
        if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
            raise ContractError("NVENC fps must be a finite positive number")
        if isinstance(value, Fraction):
            if value <= 0:
                raise ContractError("NVENC fps must be a finite positive number")
            return float(value), f"{value.numerator}/{value.denominator}"
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ContractError("NVENC fps must be a finite positive number") from exc
        if not math.isfinite(numeric) or numeric <= 0:
            raise ContractError("NVENC fps must be a finite positive number")
        return numeric, format(numeric, ".12g")

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def published(self) -> bool:
        return self._published

    def __enter__(self) -> Self:
        if self._closed:
            raise ContractError("cannot enter a closed NVENC writer")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if exc_type is None:
            self.close()
        else:
            self.abort()
        return False

    def write(self, frame: np.ndarray) -> None:
        if self._closed:
            raise ContractError("cannot write to a closed NVENC writer")
        if self._frame_count >= self.expected_frame_count:
            raise ContractError("refusing to write more than expected_frame_count frames")
        if not isinstance(frame, np.ndarray):
            raise ContractError("NVENC frame must be a numpy.ndarray")
        expected_shape = (self.height, self.width, 3)
        if frame.shape != expected_shape:
            raise ContractError(
                f"NVENC frame shape must be {expected_shape}, got {frame.shape}"
            )
        if frame.dtype != np.uint8:
            raise ContractError(
                f"NVENC frame dtype must be uint8, got {frame.dtype}"
            )
        if not frame.flags.c_contiguous:
            raise ContractError("NVENC frame must be C-contiguous")
        if self._stdin is None:
            raise ContractError("FFmpeg NVENC stdin pipe is unavailable")

        payload = memoryview(frame).cast("B")
        try:
            total_written = 0
            while total_written < frame.nbytes:
                written = self._stdin.write(payload[total_written:])
                if written is None or written <= 0:
                    raise OSError(
                        "FFmpeg pipe stopped accepting a raw frame at "
                        f"{total_written}/{frame.nbytes} bytes"
                    )
                total_written += written
        except Exception as exc:
            return_code, stderr_text = self._finish_failed_process()
            detail = self._stderr_detail(stderr_text)
            raise ContractError(
                "FFmpeg NVENC rejected a raw frame "
                f"(returncode={return_code}): {detail}"
            ) from exc
        self._frame_count += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        close_error: Exception | None = None
        if self._stdin is not None:
            try:
                self._stdin.close()
            except Exception as exc:
                close_error = exc
            self._stdin = None

        return_code, wait_error = self._wait_process()
        stderr_text, stderr_error = self._read_stderr()
        if close_error is not None or stderr_error is not None or wait_error is not None:
            cleanup_error = self._remove_temporary_file()
            reasons = [
                str(error)
                for error in (close_error, stderr_error, wait_error)
                if error is not None
            ]
            if cleanup_error is not None:
                reasons.append(f"temporary-file cleanup failed: {cleanup_error}")
            raise ContractError(
                "failed while closing FFmpeg NVENC encoder: " + "; ".join(reasons)
            )
        if return_code != 0:
            cleanup_error = self._remove_temporary_file()
            detail = self._stderr_detail(stderr_text)
            if cleanup_error is not None:
                detail += f"; temporary-file cleanup failed: {cleanup_error}"
            raise ContractError(
                f"FFmpeg NVENC encoding failed ({return_code}): {detail}"
            )
        if self._frame_count != self.expected_frame_count:
            cleanup_error = self._remove_temporary_file()
            suffix = (
                ""
                if cleanup_error is None
                else f"; temporary-file cleanup failed: {cleanup_error}"
            )
            raise ContractError(
                "refusing to publish NVENC video with unexpected frame count: "
                f"{self._frame_count} != {self.expected_frame_count}{suffix}"
            )

        try:
            size = self.temporary_path.stat().st_size
        except OSError as exc:
            self._remove_temporary_file()
            raise ContractError(
                f"FFmpeg NVENC did not create its temporary MP4: {exc}"
            ) from exc
        if size <= 0:
            cleanup_error = self._remove_temporary_file()
            suffix = (
                ""
                if cleanup_error is None
                else f"; temporary-file cleanup failed: {cleanup_error}"
            )
            raise ContractError(f"FFmpeg NVENC produced an empty MP4{suffix}")

        try:
            os.replace(self.temporary_path, self.output_path)
        except OSError as exc:
            cleanup_error = self._remove_temporary_file()
            suffix = (
                ""
                if cleanup_error is None
                else f"; temporary-file cleanup failed: {cleanup_error}"
            )
            raise ContractError(
                f"cannot atomically publish NVENC video {self.output_path}: {exc}{suffix}"
            ) from exc
        self._published = True

    def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._stdin is not None:
            try:
                self._stdin.close()
            except Exception:
                pass
            self._stdin = None
        self._terminate_process()
        self._read_stderr()
        self._remove_temporary_file()

    def _finish_failed_process(self) -> tuple[int | None, str]:
        self._closed = True
        if self._stdin is not None:
            try:
                self._stdin.close()
            except Exception:
                pass
            self._stdin = None
        return_code, _ = self._wait_process()
        stderr_text, _ = self._read_stderr()
        self._remove_temporary_file()
        return return_code, stderr_text

    def _drain_stderr(self) -> None:
        stream = self._stderr
        if stream is None:
            return
        try:
            while True:
                payload = stream.read(8_192)
                if not payload:
                    break
                if isinstance(payload, str):
                    encoded = payload.encode("utf-8", errors="replace")
                else:
                    encoded = bytes(payload)
                self._stderr_tail.extend(encoded)
                if len(self._stderr_tail) > 65_536:
                    del self._stderr_tail[:-65_536]
        except Exception as exc:
            self._stderr_error = exc
        finally:
            try:
                stream.close()
            except Exception as exc:
                if self._stderr_error is None:
                    self._stderr_error = exc

    def _read_stderr(self) -> tuple[str, Exception | None]:
        thread = self._stderr_thread
        if thread is not None:
            thread.join()
        self._stderr_thread = None
        self._stderr = None
        return (
            bytes(self._stderr_tail).decode("utf-8", errors="replace"),
            self._stderr_error,
        )

    def _wait_process(self) -> tuple[int | None, Exception | None]:
        try:
            return self._process.wait(), None
        except Exception as exc:
            self._terminate_process()
            return getattr(self._process, "returncode", None), exc

    def _terminate_process(self) -> None:
        try:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
        except Exception:
            try:
                self._process.kill()
                self._process.wait()
            except Exception:
                pass

    def _remove_temporary_file(self) -> OSError | None:
        try:
            self.temporary_path.unlink(missing_ok=True)
        except OSError as exc:
            return exc
        return None

    @staticmethod
    def _stderr_detail(stderr_text: str) -> str:
        detail = stderr_text.strip()
        if not detail:
            return "FFmpeg produced no stderr diagnostics"
        if len(detail) > 8_000:
            return "..." + detail[-8_000:]
        return detail


__all__ = ["NvencVideoWriter", "PopenFactory"]
