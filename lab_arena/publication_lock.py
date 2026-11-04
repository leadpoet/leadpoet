"""Exclude duplicate publication work on the canonical single gateway host.

All publisher processes must use the same local directory. This lock does not
coordinate separate hosts. Keep its files outside release slots and never
unlink them: every contender must open the same persistent inode. Process exit
closes the descriptor and releases ownership without a lease or cleanup job.
"""

from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
import stat
from typing import Iterator, Optional
from urllib.parse import urlsplit

from lab_arena.runtime_host import _open_directory_tree


class PublicationLockError(RuntimeError):
    pass


def production_lock_directory(endpoint: str, mode: str) -> Path:
    """Use the service user's stable home, shared across API and driver slots.

    The public database endpoint and mode separate independent deployments.
    Credentials never contribute to the namespace or appear in filenames.
    """
    parsed = urlsplit(endpoint)
    # The transport validates the endpoint before production calls this helper.
    # Remove non-public URL components here as well, without retaining them.
    port = parsed.port
    if port == {"https": 443, "http": 80}.get(parsed.scheme.lower()):
        port = None
    identity = (parsed.scheme.lower(), parsed.hostname, port,
                parsed.path.rstrip("/"), mode)
    namespace = hashlib.sha256(repr(identity).encode("utf-8")).hexdigest()
    return Path.home() / ".config" / "leadpoet" / "arena-publication-locks" / namespace


def _open_lock(directory: Path, round_id: str) -> int:
    if (not directory.is_absolute() or ".." in directory.parts
            or directory == Path("/")):
        raise PublicationLockError("publication lock directory is unsafe")
    directory_fd = None
    descriptor = None
    try:
        directory_fd = _open_directory_tree(directory, create=True)
        details = os.fstat(directory_fd)
        if details.st_uid != os.geteuid() or stat.S_IMODE(details.st_mode) & 0o077:
            raise PublicationLockError("publication lock directory is unsafe")
        filename = hashlib.sha256(round_id.encode("utf-8")).hexdigest() + ".lock"
        descriptor = os.open(
            filename, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
            0o600, dir_fd=directory_fd,
        )
        details = os.fstat(descriptor)
        current = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        if (not stat.S_ISREG(details.st_mode) or details.st_nlink != 1
                or details.st_uid != os.geteuid()
                or stat.S_IMODE(details.st_mode) != 0o600
                or (details.st_dev, details.st_ino) != (current.st_dev, current.st_ino)):
            raise PublicationLockError("publication lock file is unsafe")
        owned = descriptor
        descriptor = None
        return owned
    except OSError:
        raise PublicationLockError("publication lock is unavailable") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory_fd is not None:
            os.close(directory_fd)


@contextmanager
def round_lock(directory: Optional[Path], round_id: str) -> Iterator[bool]:
    # Programmatic test services have no shared production filesystem. They may
    # inject an isolated directory to exercise exclusion with real processes.
    if directory is None:
        yield True
        return
    descriptor = _open_lock(Path(directory), round_id)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            acquired = False
        except OSError:
            raise PublicationLockError("publication lock is unavailable") from None
        else:
            acquired = True
        yield acquired
    finally:
        os.close(descriptor)
