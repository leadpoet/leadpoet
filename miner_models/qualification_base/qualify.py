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
your return value.  The schemas live in
``gateway/qualification/models.py``:

  * ``ICPPrompt``      — what you receive
  * ``LeadOutput``     — what you return when ``icp['mode'] == 'lead'``
                         (legacy / default)
  * ``CompanyOutput``  — what you return when ``icp['mode'] == 'company'``

Pydantic ``extra = 'forbid'`` is set on both output models: any extra
key (e.g. an email field on a CompanyOutput) gives an instant 0 for
that ICP.  Build to the schema exactly.

SCORING
-------
For the same ICP, validators run:

* Lead-mode (``score_lead`` in ``qualification/scoring/lead_scorer.py``):
  ICP fit LLM (0-20), decision-maker LLM (0-20), per-signal intent
  verification with time decay (0-60), cost penalty.  Max 100.

* Company-mode (``score_company`` in the same file):
  Company-existence check (HTTP fetch — must pass), company-mode ICP
  fit LLM (0-40), per-signal intent verification (0-60), cost penalty.
  Max 100.

In both modes, fabricated intent signals (signals whose URL doesn't
contain the claim, hardcoded dates, dup domains) zero the entire
score.  See ``qualification/scoring/intent_verification.py`` for the
exact rules.
"""

from __future__ import annotations

from typing import Any, Dict


def _lead_mode_stub(icp: Dict[str, Any]) -> Dict[str, Any]:
    """Return a trivially valid LeadOutput skeleton.

    This is NOT a real lead — it's the shape only.  A real implementation
    here would query the published leads table
    (``test_leads_for_miners``) intelligently using
    ``icp['industry']``, ``icp['target_roles']``, etc., and attach
    verifiable intent signals from public sources (LinkedIn, news,
    job boards).  Score 0 expected from this stub.
    """
    return {
        "lead_id": 0,
        "business": "ExampleCo",
        "company_linkedin": "https://www.linkedin.com/company/exampleco",
        "company_website": "https://exampleco.com",
        "employee_count": icp.get("employee_count") or "51-200",
        "industry": icp.get("industry") or "Unknown",
        "sub_industry": icp.get("sub_industry") or "Unknown",
        "country": icp.get("country") or icp.get("geography") or "United States",
        "city": "San Francisco",
        "state": "California",
        "role": (icp.get("target_roles") or ["VP Sales"])[0],
        "role_type": "Sales",
        "seniority": icp.get("target_seniority") or "VP",
        "intent_signals": [
            {
                "source": "news",
                "description": "Placeholder signal — replace with real evidence",
                "url": "https://example.com/never-going-to-verify",
                "date": None,
                "snippet": "This signal is intentionally fake and will score 0.",
            }
        ],
    }


def _company_mode_stub(icp: Dict[str, Any]) -> Dict[str, Any]:
    """Return a trivially valid CompanyOutput skeleton.

    A real implementation would search the open web (news, job boards,
    company websites) for companies that match ``icp`` and have at
    least one verifiable intent signal.  The model competition is
    explicitly NOT about contact enrichment — leave that to
    fulfillment miners.

    This stub returns a known-fake company so it scores 0; replace
    everything between the comments below.
    """
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


def qualify(icp: Dict[str, Any]) -> Dict[str, Any]:
    """Validator entry point.

    The validator-side TEE sandbox imports this as
    ``from model import qualify`` and calls ``qualify(icp_dict)``,
    where ``icp_dict`` is ``ICPPrompt.model_dump(mode='json')``.
    Return one output dict matching the schema selected by
    ``icp['mode']`` ("lead" — default — or "company").
    """
    mode = (icp or {}).get("mode") or "lead"
    if mode == "company":
        return _company_mode_stub(icp)
    return _lead_mode_stub(icp)
