from __future__ import annotations

import json

from gateway.tee.prepare_gateway_envelopes_v2 import (
    prepare_gateway_envelopes_v2,
)
from gateway.utils.tee_kms_provision_v2 import validate_provider_envelope


class KMS:
    def __init__(self):
        self.requests = []

    def encrypt(self, **request):
        self.requests.append(request)
        return {
            "KeyId": "arn:aws:kms:us-east-1:123:key/gateway-v2",
            "CiphertextBlob": ("ciphertext:%03d" % len(self.requests)).encode(),
        }


def _environment():
    return {
        "RESEARCH_LAB_HOSTED_RUNS_ENABLED": "true",
        "RESEARCH_LAB_EVALUATION_BUNDLES_ENABLED": "true",
        "RESEARCH_LAB_V2_OPENROUTER_API_KEY": "openrouter-secret",
        "RESEARCH_LAB_V2_EXA_API_KEY": "exa-secret",
        "RESEARCH_LAB_V2_SCRAPINGDOG_API_KEY": "scrapingdog-secret",
        "RESEARCH_LAB_V2_DEEPLINE_API_KEY": "deepline-secret",
        "SUPABASE_SERVICE_ROLE_KEY": "supabase-secret",
        "TRUELIST_API_KEY": "truelist-secret",
        "RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_1": "https://hosted-1",
        "RESEARCH_LAB_AUTO_RESEARCH_WEBSHARE_PROXY_2": "https://hosted-2",
        "RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_1": "https://scoring-1",
        "RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_2": "https://scoring-2",
        "RESEARCH_LAB_QUALIFICATION_WEBSHARE_PROXY_3": "https://scoring-3",
    }


def test_prepares_complete_dynamic_gateway_envelope_set(tmp_path):
    kms = KMS()
    output = tmp_path / "v2"
    result = prepare_gateway_envelopes_v2(
        environment=_environment(),
        kms_key_id="alias/gateway-v2",
        deploy_commit="1" * 40,
        output_dir=output,
        kms_client=kms,
    )

    assert result["hosted_worker_count"] == 2
    assert result["scoring_worker_count"] == 3
    assert (output / "autoresearch_proxy_01.json").is_file()
    assert (output / "scoring_proxy_02.json").is_file()
    assert result["required_count_environment"] == {
        "RESEARCH_LAB_HOSTED_WORKER_PROCESS_COUNT": "2",
        "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT": "3",
    }
    assert "SUPABASE_SERVICE_ROLE_KEY" not in result["plaintext_environment_names_to_remove"]
    assert "TRUELIST_API_KEY" not in result["plaintext_environment_names_to_remove"]
    documents = [
        json.loads(path.read_text())
        for path in output.glob("*.json")
        if path.name != "gateway-v2-env-transition.json"
    ]
    assert len(documents) == 7 + 5 + 5
    assert all(validate_provider_envelope(document) for document in documents)
    assert not any(
        secret in json.dumps(documents)
        for secret in (
            "openrouter-secret",
            "exa-secret",
            "scrapingdog-secret",
            "https://hosted-1",
        )
    )
