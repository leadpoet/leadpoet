"""Frozen qualification fork for Research Lab Engine v0.

The production qualification model is network-heavy. Engine v0 needs a real
``qualify(icp)`` module boundary, but it must be deterministic and cheap while
calibrating loop depth. This fork therefore reads frozen L1 fixtures and emits
CompanyOutput-shaped dictionaries through three strategies:

* ``reference``: conservative reference behavior.
* ``source_routing_v0``: a structurally distinct source-routing patch.
* ``overbroad_v0``: a deliberately noisy patch used to exercise guardrails.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


MAX_LEADS_PER_ICP = 5
_FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "research_lab"
    / "fixtures"
    / "research_loop_v0_fixtures.json"
)


def qualify(icp: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Default lab fork entry point: conservative reference behavior."""
    return qualify_reference(icp)


def qualify_reference(icp: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _qualify_with_strategy(icp, "reference")


def qualify_source_routing_v0(icp: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _qualify_with_strategy(icp, "source_routing_v0")


def qualify_overbroad_v0(icp: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _qualify_with_strategy(icp, "overbroad_v0")


def _qualify_with_strategy(icp: Dict[str, Any], strategy: str) -> List[Dict[str, Any]]:
    if not isinstance(icp, dict):
        return []
    fixture = _fixture_for_icp(icp)
    if fixture is None:
        return []

    candidates = [
        candidate
        for candidate in fixture["candidate_pool"]
        if strategy in candidate.get("strategies", [])
    ]
    candidates.sort(
        key=lambda candidate: (
            -float(candidate.get("strategy_score", {}).get(strategy, 0.0)),
            candidate["company_name"],
        )
    )
    return [_to_company_output(candidate) for candidate in candidates[:MAX_LEADS_PER_ICP]]


def _fixture_for_icp(icp: Dict[str, Any]) -> Dict[str, Any] | None:
    icp_id = str(icp.get("icp_id") or icp.get("fixture_id") or "")
    with _FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        fixture_set = json.load(handle)
    for fixture in fixture_set["fixtures"]:
        if fixture["fixture_id"] == icp_id or fixture["icp"].get("icp_id") == icp_id:
            return fixture
    return None


def _to_company_output(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "company_name": candidate["company_name"],
        "website": candidate["website"],
        "description": candidate["description"],
        "intent_signals": [
            {
                "source": candidate["source"],
                "url": candidate["source_url"],
                "date": candidate["date"],
                "snippet": candidate["snippet"],
                "description": candidate["description"],
            }
        ],
    }
