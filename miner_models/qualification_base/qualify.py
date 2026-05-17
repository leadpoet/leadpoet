"""
Reference qualification miner — minimal starting point for the
Lead Qualification Agent competition.

This file is illustrative.  It is NOT a winning submission, and is
unlikely to ever earn a champion score.  It exists so that new miners
can see exactly what the subnet expects from a ``qualify(icp)`` entry
point and roll their own from a known-good skeleton.

WHAT TO PUT WHERE
-----------------
* Place ``qualify.py`` in a directory.  At submission time the miner
  CLI (``neurons/miner.py``) will tar that directory with
  ``arcname='model'``, upload it to S3, and the validator's TEE
  sandbox will import it as ``from model import qualify``.
* Add any dependencies in ``requirements.txt`` next to this file.
* Code must run in an isolated container with NO unrestricted network
  access.  Outbound HTTP is only allowed via the validator's API
  proxy URL injected at runtime as ``QUALIFICATION_PROXY_URL`` env
  var (talks to gateway-approved upstreams; counted toward your
  per-lead cost budget).

THE CONTRACT
------------
``qualify(icp: dict) -> dict``

The validator passes ICP as a JSON-serialized dict and JSON-decodes
your return value.  As of May 2026 the competition is single-path:
miners must return a ``CompanyOutput`` dict matching the schema in
``gateway/qualification/models.py``.

There is no person / role / email / seniority dimension.  Surfacing
contacts cleanly requires Apify / LinkedIn scraping, which we do not
want baked into the base miner.  Fulfillment miners can layer their
own contact enrichment on top of a license-clean base model.

Pydantic ``extra = 'forbid'`` is set on ``CompanyOutput``: any extra
key (e.g. a ``full_name`` or ``email`` field) gives an instant 0 for
that ICP.  Build to the schema exactly.

SCORING
-------
For each ICP, the validator runs ``score_company`` in
``qualification/scoring/lead_scorer.py``:

  * Hard gate: company-existence check (HTTP fetch of
    ``company_website`` — must return 2xx/3xx, must mention
    ``company_name``, must not be a parked / for-sale domain).
    Failure → score 0.
  * Company-mode ICP-fit LLM (0-40): industry + product + structural
    + intent-class fit.
  * Per-signal intent verification with time decay (0-60).
  * Cost variability penalty (-5 if run cost exceeds 2x average).
  * Max total = 100.

Fabricated intent signals (signals whose URL doesn't contain the
claim, hardcoded dates, dup domains) zero the entire score.  See
``qualification/scoring/verification_helpers.py`` for the exact rules.
"""

from __future__ import annotations

from typing import Any, Dict


def qualify(icp: Dict[str, Any]) -> Dict[str, Any]:
    """Validator entry point.

    The validator-side TEE sandbox imports this as
    ``from model import qualify`` and calls ``qualify(icp_dict)``,
    where ``icp_dict`` is ``ICPPrompt.model_dump(mode='json')``.
    Return one ``CompanyOutput``-shaped dict.

    A real implementation would search the open web (news, job boards,
    company websites, GitHub, LinkedIn company pages) for companies
    that match ``icp`` and have at least one verifiable intent signal.

    This stub returns a known-fake company so it scores 0; replace
    everything between the REPLACE markers below.
    """
    icp = icp or {}
    return {
        # --- REPLACE: pick a real company that matches the ICP -------
        "company_name": "ExampleCo",
        "company_website": "https://exampleco.com",
        "company_linkedin": "",
        # ------------------------------------------------------------

        # Mirror ICP classification fields as best you can.
        "industry": icp.get("industry") or "Unknown",
        "sub_industry": icp.get("sub_industry") or "",
        "employee_count": icp.get("employee_count") or "51-200",
        "company_stage": icp.get("company_stage") or "",
        "country": icp.get("country") or icp.get("geography") or "United States",
        "state": "",
        "description": "",

        # --- REPLACE: at least one VERIFIABLE intent signal ---------
        # ``url`` must point at a page where the signal can be re-read,
        # ``snippet`` must literally appear on that page (otherwise the
        # validator's intent_verification module will mark the signal
        # fabricated and zero your score for this ICP).
        "intent_signals": [
            {
                "source": "news",
                "description": "Placeholder signal — replace with real evidence",
                "url": "https://example.com/never-going-to-verify",
                "date": None,
                "snippet": "This signal is intentionally fake and will score 0.",
            }
        ],
        # ------------------------------------------------------------
    }
