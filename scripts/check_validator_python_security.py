"""Reject an unpatched interpreter before canonical validator activation."""

import hashlib
from pathlib import Path
import re
import ssl
import sys


COMPROMISED_ROOTS = {
    "86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b",
    "d7a7a0fb5d7e2731d771e9484ebcdef71d5f0c3e0a2948782bc83ee0ea699ef4",
}


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
    for ca_file in (certifi.where(), bundle):
        # Explicit CA files keep this check separate from OS and custom trust.
        context = ssl.create_default_context(cafile=str(ca_file))
        roots = {hashlib.sha256(cert).hexdigest()
                 for cert in context.get_ca_certs(binary_form=True)}
        if roots & COMPROMISED_ROOTS:
            raise ValueError("a certifi bundle retains a compromised root")


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
