#!/usr/bin/env python3
"""Verify gateway.config loads NUL-separated fallback env files."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    with TemporaryDirectory() as tmp:
        env_path = Path(tmp) / "gw.environ"
        env_path.write_bytes(
            b"\0".join(
                [
                    b"SUPABASE_URL=https://example.supabase.co",
                    b"SUPABASE_SERVICE_ROLE_KEY=service-role",
                    b"RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_1=http://hosted-1.example:8000",
                    b"RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_1=http://scoring-1.example:9000",
                    b"BASH_FUNC_which%%=ignored",
                    b"# ignored comment",
                ]
            )
        )

        original = {key: os.environ.get(key) for key in (
            "GATEWAY_ENV_FILE",
            "SUPABASE_URL",
            "SUPABASE_SERVICE_ROLE_KEY",
            "RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_1",
            "RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_1",
            "BASH_FUNC_which%%",
        )}
        try:
            for key in original:
                os.environ.pop(key, None)
            os.environ["GATEWAY_ENV_FILE"] = str(env_path)
            os.environ["SUPABASE_URL"] = "already-set"
            import gateway.config as config
            importlib.reload(config)

            assert os.environ["SUPABASE_URL"] == "already-set"
            assert os.environ["RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_1"] == "http://hosted-1.example:8000"
            assert os.environ["RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_1"] == "http://scoring-1.example:9000"
            assert "BASH_FUNC_which%%" not in os.environ
        finally:
            for key, value in original.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    print("gateway env file loader verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
