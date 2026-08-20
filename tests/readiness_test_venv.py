from __future__ import annotations

import importlib.metadata
from pathlib import Path
import shutil
import subprocess
import sys
import venv


_READINESS_DISTRIBUTIONS = ("cbor2", "cryptography", "cffi", "pycparser")


def build_dependency_complete_readiness_venv(root: Path) -> Path:
    """Build the isolated, local-only verifier environment used by restart tests."""

    venv.EnvBuilder(with_pip=False, symlinks=True).create(root)
    python = root / "bin" / "python"
    site_packages = (
        root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    for name in _READINESS_DISTRIBUTIONS:
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            if name in {"cbor2", "cryptography"}:
                raise
            continue
        for item in distribution.files or ():
            relative = Path(item)
            if relative.is_absolute() or ".." in relative.parts:
                continue
            source = Path(distribution.locate_file(item))
            if not source.is_file():
                continue
            destination = site_packages / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    subprocess.run(
        [
            str(python),
            "-I",
            "-S",
            "-c",
            (
                "import sys; sys.path.append(sys.argv[1]); "
                "import cbor2, cryptography; "
                "from cryptography import x509; "
                "from cryptography.hazmat.primitives.asymmetric import ec"
            ),
            str(site_packages),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return python
