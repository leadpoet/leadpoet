"""One publication assessment per round on a single gateway host.

The file stays in a stable host directory across release changes. Kernel flock
releases on process exit; the lock file itself must never be removed or replaced.
This does not coordinate publishers on different hosts.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import secrets
import stat
from typing import Iterator

from lab_arena.runtime_host import _open_directory_tree


class PublicationLockError(RuntimeError):
    """The local lock cannot safely protect publication."""


def _lock_file(directory: Path, round_id: str) -> int:
    directory = Path(directory)
    if not directory.is_absolute() or ".." in directory.parts or directory == Path("/"):
        raise PublicationLockError("publication lock directory is invalid")
    if not isinstance(round_id, str) or not round_id:
        raise PublicationLockError("publication round is invalid")
    name = "arena-publication-" + hashlib.sha256(round_id.encode("utf-8")).hexdigest() + ".lock"
    try:
        parent = _open_directory_tree(directory, create=False)
        descriptor = None
        try:
            parent_stat = os.fstat(parent)
            if (parent_stat.st_mode & 0o077) or (
                parent_stat.st_uid != os.geteuid() and os.geteuid() != 0
            ):
                raise PublicationLockError("publication lock directory is unsafe")
            flags = os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC
            try:
                descriptor = os.open(name, flags, dir_fd=parent)
            except FileNotFoundError:
                # Publish the inode only after root has assigned it to the
                # directory owner. A second process can never see a root-only
                # final lock file. A losing creator opens the winning inode.
                temporary = ".arena-publication-" + secrets.token_hex(16)
                descriptor = os.open(temporary, flags | os.O_CREAT | os.O_EXCL,
                                     0o600, dir_fd=parent)
                try:
                    if os.geteuid() == 0 and parent_stat.st_uid != 0:
                        os.fchown(descriptor, parent_stat.st_uid, parent_stat.st_gid)
                    try:
                        os.link(temporary, name, src_dir_fd=parent,
                                dst_dir_fd=parent, follow_symlinks=False)
                    except FileExistsError:
                        os.close(descriptor)
                        descriptor = None
                        descriptor = os.open(name, flags, dir_fd=parent)
                finally:
                    os.unlink(temporary, dir_fd=parent)
            details = os.fstat(descriptor)
            # A killed creator can leave its private temporary hard link after
            # publishing the final name. Both names still refer to this one
            # inode, so a later publisher must be able to take the final lock.
            if (not stat.S_ISREG(details.st_mode) or details.st_nlink < 1
                or details.st_uid != parent_stat.st_uid
                or stat.S_IMODE(details.st_mode) != 0o600):
                raise PublicationLockError("publication lock file is unsafe")
            acquired = descriptor
            descriptor = None
            return acquired
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent)
    except (OSError, ValueError) as exc:
        raise PublicationLockError("publication lock unavailable") from exc


@contextmanager
def publication_round_lock(directory: Path, round_id: str) -> Iterator[bool]:
    """Try the per-round host lock without waiting; close it on every exit."""
    try:
        import fcntl
    except ImportError as exc:
        raise PublicationLockError("publication lock unavailable") from exc
    descriptor = _lock_file(directory, round_id)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            acquired = False
        except OSError as exc:
            raise PublicationLockError("publication lock unavailable") from exc
        else:
            acquired = True
        yield acquired
    finally:
        os.close(descriptor)
