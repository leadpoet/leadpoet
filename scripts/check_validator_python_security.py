"""Reject an unpatched interpreter before canonical validator activation."""

from pathlib import Path
import re
import sys


def stable_version(value: str) -> tuple[int, ...]:
    if not re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?", value):
        raise ValueError("a stable package version is required")
    parts = tuple(int(part) for part in value.split("."))
    return parts + (0,) * (3 - len(parts))


def check_security() -> None:
    import certifi
    import pip
    from pip._vendor import certifi as pip_certifi

    if stable_version(certifi.__version__) < (2026, 10, 19):
        raise ValueError("runtime certifi is unpatched")
    if stable_version(pip.__version__) < (26, 2, 2):
        raise ValueError("pip is unpatched")
    if stable_version(pip_certifi.__version__) < (2026, 10, 19):
        raise ValueError("pip's private certifi is unpatched")
    bundle = Path(pip_certifi.where()).resolve(strict=True)
    if not bundle.is_file() or not bundle.is_relative_to(
        Path(pip.__file__).resolve(strict=True).parent
    ):
        raise ValueError("pip must use its patched private CA bundle")


def main() -> int:
    try:
        check_security()
    except Exception:
        # Do not print imported-package exceptions or environment contents.
        print(
            "Validator Python trust preflight failed: require stable pip>=26.2.2 "
            "and certifi>=2026.10.19, including pip's private CA bundle. "
            "Prepare a compatible environment or set VALIDATOR_PYTHON_BIN; "
            "the running service has not been stopped.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
