#!/usr/bin/env python3
"""Reject warm/persistent/cache-broker runner features before source execution."""

import os
from urllib.parse import urlparse


def verify_isolation(env):
    for name in (
        "RUNS_ON_POOL_NAME", "RUNS_ON_POOL_STANDBY_TYPE",
        "RUNS_ON_STICKYDISK_DIR", "RUNS_ON_STICKYDISK_NAME",
        "RUNS_ON_STICKYDISK_READY_FILE", "RUNS_ON_STICKYDISK_UNAVAILABLE_FILE",
        "RUNS_ON_EFS_ID",
    ):
        if env.get(name):
            raise ValueError(f"persistent or warm runner feature is forbidden: {name}")
    for name in ("ACTIONS_CACHE_URL", "ACTIONS_RESULTS_URL"):
        if env.get(name):
            endpoint = urlparse(env[name])
            host = endpoint.hostname or ""
            if endpoint.scheme != "https" or not host.endswith(".actions.githubusercontent.com"):
                raise ValueError(f"non-GitHub cache broker is forbidden: {name}")


if __name__ == "__main__":
    verify_isolation(os.environ)
    print("No warm pool, persistent cache disk, EFS or external cache broker detected")
