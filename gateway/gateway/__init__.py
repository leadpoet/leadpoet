"""Compatibility package for flat EC2 gateway checkouts.

Production gateway deploys often copy the contents of ``gateway/`` to
``/home/ec2-user/gateway``. In that layout, commands launched from the gateway
root still import modules as ``gateway.*``. This shim makes those imports
resolve to the sibling packages in the flat checkout.
"""

from __future__ import annotations

from pathlib import Path

_SHIM_DIR = Path(__file__).resolve().parent
_FLAT_GATEWAY_ROOT = _SHIM_DIR.parent

__path__ = [str(_FLAT_GATEWAY_ROOT), str(_SHIM_DIR)]

__version__ = "1.0.0"
__author__ = "LeadPoet Team"
