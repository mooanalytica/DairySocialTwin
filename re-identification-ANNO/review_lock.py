"""Cross-process lock shared by the review server and corrected exporter."""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path
from types import TracebackType


class ReviewLockUnavailable(RuntimeError):
    """Another process currently owns the review/export lock."""


class ReviewFileLock:
    """Exclusive advisory file lock released automatically on process exit."""

    def __init__(self, path: Path, *, blocking: bool) -> None:
        self.path = path
        self.blocking = blocking
        self._handle = None

    def __enter__(self) -> "ReviewFileLock":
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        handle = None
        try:
            descriptor = os.open(self.path, flags, 0o600)
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                os.close(descriptor)
                raise OSError(f"review lock is not a regular file: {self.path}")
            handle = os.fdopen(descriptor, "r+b", buffering=0)
            if metadata.st_size == 0:
                handle.write(b"\0")
            handle.seek(0)
            self._acquire(handle)
            self._handle = handle
            return self
        except ReviewLockUnavailable:
            if handle is not None:
                handle.close()
            raise
        except OSError as exc:
            if handle is not None:
                handle.close()
            raise OSError(f"cannot acquire review lock {self.path}: {exc}") from exc

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        handle = self._handle
        self._handle = None
        if handle is not None:
            try:
                self._release(handle)
            finally:
                handle.close()
        return False

    def _acquire(self, handle: object) -> None:
        if os.name == "nt":  # pragma: no cover - exercised on Windows only
            import msvcrt

            mode = msvcrt.LK_LOCK if self.blocking else msvcrt.LK_NBLCK
            try:
                msvcrt.locking(handle.fileno(), mode, 1)  # type: ignore[attr-defined]
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise ReviewLockUnavailable(
                        f"review/export lock is already held: {self.path}"
                    ) from exc
                raise
            return

        import fcntl

        operation = fcntl.LOCK_EX
        if not self.blocking:
            operation |= fcntl.LOCK_NB
        try:
            fcntl.flock(handle.fileno(), operation)  # type: ignore[attr-defined]
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise ReviewLockUnavailable(
                    f"review/export lock is already held: {self.path}"
                ) from exc
            raise

    @staticmethod
    def _release(handle: object) -> None:
        if os.name == "nt":  # pragma: no cover - exercised on Windows only
            import msvcrt

            handle.seek(0)  # type: ignore[attr-defined]
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
            return

        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
