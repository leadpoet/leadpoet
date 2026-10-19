"""A current judge image sends each round's frozen model choices."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from lab_arena import scoring


@pytest.mark.parametrize("historical", [False, True])
def test_frozen_model_policy_is_bound_before_import_and_sent_on_the_wire(historical):
    models = dict(scoring.DEFAULT_JUDGE_MODELS)
    if historical:
        models.update({
            "intent_details_review": "anthropic/claude-sonnet-4.5",
            "role_batch_check": "google/gemini-2.5-flash",
        })
    policy = scoring.build_scorer_policy(judge_models=models)
    environment = dict(os.environ)
    for name in (*scoring.CREDENTIAL_ENV_NAMES, "ARENA_INTENT_EVIDENCE_MODEL",
                 "ARENA_INTENT_DETAILS_REVIEW_MODEL", "ARENA_ROLE_BATCH_MODEL"):
        environment.pop(name, None)
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_AND_WIRE_PROBE, json.dumps(policy)],
        cwd=Path(__file__).resolve().parents[2], env=environment,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("name", ["ARENA_INTENT_DETAILS_REVIEW_MODEL", "ARENA_ROLE_BATCH_MODEL"])
def test_frozen_model_policy_rejects_conflicting_environment_before_mutation(name):
    environment = {name: "custom/conflicting-model"}
    credentials = {key: "test-only-placeholder" for key in scoring.CREDENTIAL_ENV_NAMES}
    with pytest.raises(scoring.ScorerPolicyConflict):
        scoring.apply_policy_to_environment(
            scoring.build_scorer_policy(), environ=environment, credentials=credentials,
        )
    assert environment == {name: "custom/conflicting-model"}


_IMPORT_AND_WIRE_PROBE = r'''
import asyncio
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock
from lab_arena import scoring

policy = json.loads(sys.argv[1])
models = policy["judge_models"]
scoring.apply_policy_to_environment(
    policy, environ=os.environ,
    credentials={name: "test-only-placeholder" for name in scoring.CREDENTIAL_ENV_NAMES},
)
from qualification.scoring import intent_details, role_batch_check, verification_helpers
from qualification.scoring import company_evidence_investigator as investigator
assert intent_details.REVIEW_MODEL == models["intent_details_review"]
assert role_batch_check.MODEL == investigator.INVESTIGATOR_MODEL == models["role_batch_check"]
requests = []

class Client:
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        pass
    async def post(self, url, *, json, **kwargs):
        requests.append(json)
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": '[{"id": 1, "match": true}]'}}]})

verification_helpers.httpx.AsyncClient = Client
asyncio.run(verification_helpers.openrouter_chat(
    "review", model=intent_details.REVIEW_MODEL, max_retries=0,
))
assert requests[-1]["model"] == models["intent_details_review"]
if models["intent_details_review"] == "anthropic/claude-sonnet-5.5":
    assert not {"temperature", "top_p", "top_k"} & requests[-1].keys()
else:
    assert requests[-1]["temperature"] == 0.3
asyncio.run(role_batch_check._judge_chunk(
    Client(), "test-only-placeholder", ["VP Sales"], [{"id": 1, "role": "VP Sales"}],
))
assert requests[-1]["model"] == models["role_batch_check"]
assert requests[-1]["temperature"] == 0

async def post_json(_session, _url, *, headers, payload):
    requests.append(payload)
    return 503, {}
investigator._post_json = post_json
investigator._search_web = AsyncMock(return_value={"results": []})
investigator._fetch_page = AsyncMock(return_value={})
asyncio.run(investigator.investigate_company_evidence(
    company_locator={"name": "Acme", "website": "https://acme.example"},
    targets=("stage",), requested_stage="Series B",
))
assert requests[-1]["model"] == models["role_batch_check"]
assert requests[-1]["max_tokens"] == investigator.REASONING_MAX_TOKENS
assert requests[-1]["temperature"] == 0.0
'''
