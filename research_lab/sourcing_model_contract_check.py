"""Sourcing-model consumer-contract conformance checks.

Historical rollback contracts are snapshotted byte-for-byte under
``research_lab/``. New signed releases are admitted by the stable
``research-lab-consumer-api:v1`` policy instead of by ``contract_id``: the
release ID may advance while the consumer-owned schema, callable/import
surface, constants, critical bindings, and runtime protocols remain
compatible. Exact reviewed legacy source identities remain available only for
retained rollback releases.

``verify_source_tree_contract`` and
``source_tree_compatibility_admission_v1`` validate a model source tree using
``ast`` and canonical bytes only — no imports or execution of untrusted model
code. Intended call sites:

* the candidate build path, so an autoresearch code-edit that would break the
  frozen adapter surface fails fast at build time instead of producing an
  image the benchmark cannot invoke (flag-gated, see code_build);
* local/CI checks against a model checkout.

Pure stdlib. A structurally valid new contract still fails closed unless its
signed source satisfies the complete consumer policy; JSON shape or a receipt
alone never authorizes unknown semantics.
"""

from __future__ import annotations

import ast
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import threading
from typing import Any, Dict, List, Mapping

CONTRACT_PATH = Path(__file__).with_name("sourcing_model_contract.json")
PARITY_FIXTURE_PATH = Path(__file__).with_name(
    "sourcing_model_parity_fixtures.json"
)
CONTRACT_V7_PATH = Path(__file__).with_name("sourcing_model_contract_v7.json")
PARITY_FIXTURE_V7_PATH = Path(__file__).with_name(
    "sourcing_model_parity_fixtures_v7.json"
)
CONTRACT_V11_PATH = Path(__file__).with_name("sourcing_model_contract_v11.json")
PARITY_FIXTURE_V11_PATH = Path(__file__).with_name(
    "sourcing_model_parity_fixtures_v11.json"
)
CONTRACT_V12_PATH = Path(__file__).with_name("sourcing_model_contract_v12.json")
PARITY_FIXTURE_V12_PATH = Path(__file__).with_name(
    "sourcing_model_parity_fixtures_v12.json"
)
CONTRACT_V13_PATH = Path(__file__).with_name("sourcing_model_contract_v13.json")
PARITY_FIXTURE_V13_PATH = Path(__file__).with_name(
    "sourcing_model_parity_fixtures_v13.json"
)
CONTRACT_V26_PATH = Path(__file__).with_name("sourcing_model_contract_v26.json")
PARITY_FIXTURE_V26_PATH = Path(__file__).with_name(
    "sourcing_model_parity_fixtures_v26.json"
)
CONTRACT_V46_PATH = Path(__file__).with_name("sourcing_model_contract_v46.json")
PARITY_FIXTURE_V46_PATH = Path(__file__).with_name(
    "sourcing_model_parity_fixtures_v46.json"
)
CONTRACT_V47_PATH = Path(__file__).with_name("sourcing_model_contract_v47.json")
PARITY_FIXTURE_V47_PATH = Path(__file__).with_name(
    "sourcing_model_parity_fixtures_v47.json"
)
CONTRACT_V52_PATH = Path(__file__).with_name("sourcing_model_contract_v52.json")
PARITY_FIXTURE_V52_PATH = Path(__file__).with_name(
    "sourcing_model_parity_fixtures_v52.json"
)
CONTRACT_V55_PATH = Path(__file__).with_name("sourcing_model_contract_v55.json")
PARITY_FIXTURE_V55_PATH = Path(__file__).with_name(
    "sourcing_model_parity_fixtures_v55.json"
)
CONTRACT_V55_E55_PATH = Path(__file__).with_name(
    "sourcing_model_contract_v55_e55.json"
)
PARITY_FIXTURE_V55_E55_PATH = Path(__file__).with_name(
    "sourcing_model_parity_fixtures_v55_e55.json"
)
CONTRACT_V68_PATH = Path(__file__).with_name("sourcing_model_contract_v68.json")
PARITY_FIXTURE_V28_PATH = Path(__file__).with_name(
    "sourcing_model_parity_fixtures_v28.json"
)
ADDITIVE_DISPATCH_CUSTODY_V3_CONTRACT_ID = (
    "leadpoet-sourcing-wrapper-contract-v68"
)
ADDITIVE_DISPATCH_CUSTODY_V3_CONTRACT_SHA256 = (
    "sha256:960c3ce752f1b978ebe0eb67be5db432fc4698b2a328d1f40ef6b0c52d38f7ab"
)
ADDITIVE_DISPATCH_CUSTODY_V3_PARITY_SHA256 = (
    "sha256:deef2e842dc70dd5e1e10f19693237e5714527a17b814c05e4f5bb47fd16e003"
)
ADDITIVE_DISPATCH_CUSTODY_V3_METADATA_SHA256 = (
    "sha256:bcbd1d629d1328d43a56e2b3585d776d0b9e8b6c8b9af465aff915d3788239db"
)
ADDITIVE_DISPATCH_CUSTODY_V3_ROUTING_COMPILER_VERSION = "routing-compiler-v5"
SEMANTIC_COMPATIBILITY_POLICY_V1_PATH = Path(__file__).with_name(
    "sourcing_model_semantic_compatibility_v1.json"
)
SEMANTIC_COMPATIBILITY_POLICY_SCHEMA_V1 = (
    "leadpoet.sourcing-model-compatibility-policy.v1"
)
SEMANTIC_COMPATIBILITY_CONSUMER_API_V1 = "research-lab-consumer-api:v1"
SEMANTIC_COMPATIBILITY_RECEIPT_SCHEMA_V1 = (
    "leadpoet.sourcing-model-compatibility-admission.v1"
)
SEMANTIC_COMPATIBILITY_RUNTIME_INVARIANTS_SCHEMA_V1 = (
    "leadpoet.sourcing-model-runtime-invariants.v1"
)
SEMANTIC_COMPATIBILITY_ACCEPTED_DECISION = "accepted"
QUALIFICATION_PROTOCOL_COMPATIBILITY_RECEIPT_SCHEMA_V2 = (
    "leadpoet.sourcing-model-qualification-compatibility-admission.v2"
)
QUALIFICATION_PROTOCOL_CONSUMER_API_V2 = (
    "research-lab-qualification-consumer-api:v2"
)
QUALIFICATION_PROTOCOL_ADMISSION_MODE_V2 = "qualification_protocol_v2"
QUALIFICATION_SCORING_ADAPTER_VERSION_V1 = (
    "qualification-company-scorer:v1"
)
QUALIFICATION_SCORING_ADAPTER_VERSION_V2 = (
    "qualification-company-scorer:v2"
)
QUALIFICATION_SUPPORTED_SCORING_ADAPTER_VERSIONS = frozenset(
    {
        QUALIFICATION_SCORING_ADAPTER_VERSION_V1,
        QUALIFICATION_SCORING_ADAPTER_VERSION_V2,
    }
)
QUALIFICATION_OUTCOME_CONTRACT_V2_PATH = Path(__file__).with_name(
    "sourcing_model_qualification_outcome_v2.json"
)
_QUALIFICATION_OUTCOME_CONTRACT_POLICY_V2 = json.loads(
    QUALIFICATION_OUTCOME_CONTRACT_V2_PATH.read_text(encoding="utf-8")
)
_QUALIFICATION_OUTCOME_CONTRACT_SHA256_V2 = (
    "e618f73abddd2ddef88fa62d09fd7ad90b3ca7c69b97da7444424abdd8e9c0fa"
)
if (
    not isinstance(_QUALIFICATION_OUTCOME_CONTRACT_POLICY_V2, Mapping)
    or hashlib.sha256(
        json.dumps(
            dict(_QUALIFICATION_OUTCOME_CONTRACT_POLICY_V2),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    != _QUALIFICATION_OUTCOME_CONTRACT_SHA256_V2
):
    raise RuntimeError("qualification outcome contract differs from protected policy")
QUALIFICATION_PROTOCOL_POLICY_SHA256_V2 = (
    f"sha256:{_QUALIFICATION_OUTCOME_CONTRACT_SHA256_V2}"
)
QUALIFICATION_PROTOCOL_ENTRYPOINT_V2 = str(
    _QUALIFICATION_OUTCOME_CONTRACT_POLICY_V2["entrypoint"]
)
QUALIFICATION_PROTOCOL_REQUIRED_ENTRYPOINTS_V2 = {
    "adapter_metadata": (),
    QUALIFICATION_PROTOCOL_ENTRYPOINT_V2: ("icp", "context"),
}
SEMANTIC_COMPATIBILITY_RECEIPT_FIELDS_V1 = frozenset(
    {
        "schema_version",
        "consumer_api_version",
        "decision",
        "admission_mode",
        "policy_hash",
        "source_tree_hash",
        "manifest_hash",
        "image_digest",
        "contract_id",
        "contract_schema_major",
        "contract_hash",
        "parity_hash",
        "bindings",
        "receipt_hash",
    }
)
_SEMANTIC_COMPATIBILITY_CACHE_SIZE = 256
_SEMANTIC_COMPATIBILITY_CACHE: "OrderedDict[tuple[str, str, str, str, str, str, str], Dict[str, Any]]" = OrderedDict()
_SEMANTIC_COMPATIBILITY_CACHE_LOCK = threading.Lock()
REVIEWED_CONSUMER_SNAPSHOT_SPECS = (
    {
        "contract_id": "leadpoet-sourcing-wrapper-contract-v55",
        "contract_path": CONTRACT_V55_PATH,
        "contract_sha256": (
            "sha256:02fcbcd84b2c887d0f6ba1515fba280267fc5d2571876f990acc865f4a038d2a"
        ),
        "parity_path": PARITY_FIXTURE_V55_PATH,
        "parity_sha256": (
            "sha256:fe0e1faff8e45b432459dda2d5f5bf131aef2b5f60935d48395a814c7ed59573"
        ),
        "required_source_constants": {
            "sourcing_model/runtime_capabilities.py": {
                "CAPABILITY_CONTRACT_VERSION": (
                    "sourcing-model-runtime-capabilities:v3"
                ),
            },
        },
        "release_identities": (
            {
                "source_tree_hash": (
                    "sha256:a34a9158480dff89a53a7a5a3df27325239b7b64f476cb6f48c593520eca3858"
                ),
                "git_commit_sha": "0be5905ee24a4d8bb3ec6f316af3e8891f763919",
                "manifest_hash": (
                    "sha256:2f4876077475c7c33135ec9b727e010d7e7845e3591d9edc9101e045dcaa8c01"
                ),
                "image_digest": (
                    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
                    "sourcing-model@sha256:ed7c03e744ba9bccd13e7608a6eabf7bcdd828dabaf02e09efd2774d7187d6a5"
                ),
            },
            {
                "source_tree_hash": (
                    "sha256:2690deb3a6b9c8952e4ecd153458cfee1b0cebbd4edb79eb13129c3e96e673d5"
                ),
                "git_commit_sha": "cf6630732f7f8f16150d9dd3908dcd7f91ae7667",
                "manifest_hash": (
                    "sha256:518022b4667471f866ef4cd66b1756f6d79ebe1757e44c9194ddd7687635eddd"
                ),
                "image_digest": (
                    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
                    "sourcing-model@sha256:a6cba9be0ff862883d9a7f33eccbb0212aee89d4949c4da10e14b0d5b0c21165"
                ),
            },
        ),
        "positional_exact_signatures": True,
        "variadic_parameters": {
            "sourcing_model/corporate_filing_contract.py:"
            "build_corporate_filing_envelope": {
                "vararg": None,
                "kwarg": "payload",
            },
        },
    },
    {
        "contract_id": "leadpoet-sourcing-wrapper-contract-v52",
        "contract_path": CONTRACT_V52_PATH,
        "contract_sha256": (
            "sha256:2454c60c1e2614feef912aa6ea471307657dac7d418bdb3bdab5b105ddbb5932"
        ),
        "parity_path": PARITY_FIXTURE_V52_PATH,
        "parity_sha256": (
            "sha256:7d18b358f7f6dcf1b58a175af43288a1db244c08af6fc5295116dbfe51976332"
        ),
        "required_source_constants": {
            "sourcing_model/runtime_capabilities.py": {
                "CAPABILITY_CONTRACT_VERSION": (
                    "sourcing-model-runtime-capabilities:v3"
                ),
            },
        },
        "release_identities": (
            {
                "source_tree_hash": (
                    "sha256:603c4569fa35d6a66ee60596a44e37841aab1c6d794c3109349c1d6b7a5bcd85"
                ),
                "git_commit_sha": "6ed6289626b7e81c745daff97feabd237aa4ccee",
                "manifest_hash": (
                    "sha256:e75c820acf1e2d1348aab3d34b85c3ae578fe8043d5ef97b28817a8b234bd3c0"
                ),
                "image_digest": (
                    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
                    "sourcing-model@sha256:5539652d47471773ca94221373ce01b5e610715177155a12ae8026da48e2ae52"
                ),
            },
            {
                "source_tree_hash": (
                    "sha256:946fe12e38efa08c08631c864591bdf99c0538e6c450bdf4c33fbba3e167a969"
                ),
                "git_commit_sha": "ec5c0e7c7314e123c9fdafff63d2b809cb254cfd",
                "manifest_hash": (
                    "sha256:ee0a1ad40a12d33dabd4d7fb68d4b9507cbfcbf2fe276a69e4a05cb82dc93f52"
                ),
                "image_digest": (
                    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
                    "sourcing-model@sha256:164cdf6b5c8c37af61d61c2d4b5c0a22fc248014be8289c0db2f242602459607"
                ),
            },
        ),
        "positional_exact_signatures": True,
        "variadic_parameters": {
            "sourcing_model/corporate_filing_contract.py:"
            "build_corporate_filing_envelope": {
                "vararg": None,
                "kwarg": "payload",
            },
        },
    },
    {
        "contract_id": "leadpoet-sourcing-wrapper-contract-v47",
        "contract_path": CONTRACT_V47_PATH,
        "contract_sha256": (
            "sha256:a6a388b731a11628a95491995fa87b80fc679a0576f9442d2b494dc9f450cb15"
        ),
        "parity_path": PARITY_FIXTURE_V47_PATH,
        "parity_sha256": (
            "sha256:da1bd8df2abb99bd617795613f09b3ce116079d46bc5de7cfa1dc23b77265619"
        ),
        "positional_exact_signatures": True,
        "variadic_parameters": {
            "sourcing_model/corporate_filing_contract.py:"
            "build_corporate_filing_envelope": {
                "vararg": None,
                "kwarg": "payload",
            },
        },
    },
    {
        "contract_id": "leadpoet-sourcing-wrapper-contract-v46",
        "contract_path": CONTRACT_V46_PATH,
        "contract_sha256": (
            "sha256:3e7bffd37cb3b33821717a66a65447e3862b2c87830685a8de46189d3bbd5ef6"
        ),
        "parity_path": PARITY_FIXTURE_V46_PATH,
        "parity_sha256": (
            "sha256:4ddc10fb52d9101c3a0981f954ef86abdae8d664c5020a4f88cacaeb30dc5422"
        ),
        # This signed source already satisfies semantic-v1 and intentionally
        # receives no legacy exemption.
        "release_identities": (),
        "positional_exact_signatures": True,
        "variadic_parameters": {
            "sourcing_model/corporate_filing_contract.py:"
            "build_corporate_filing_envelope": {
                "vararg": None,
                "kwarg": "payload",
            },
        },
    },
    {
        "contract_id": "leadpoet-sourcing-wrapper-contract-v26",
        "contract_path": CONTRACT_V26_PATH,
        "contract_sha256": (
            "sha256:fb20751ddbc068d754913f5a6aea35d2330572acd267dd0e3a2906ff5c221a83"
        ),
        "parity_path": PARITY_FIXTURE_V26_PATH,
        "parity_sha256": (
            "sha256:28fd84abd9a0af578590c0744744a0e817624a5effe37f5449916b40e8557675"
        ),
        "release_identities": (
            {
                "source_tree_hash": "sha256:7ff9f905bddb8911b247fddbe1ae12c4913a3093187276f21a67b1b94356a667",
                "git_commit_sha": "101206e2ef1e6ab57e01c2ebd0417823371370c2",
                "manifest_hash": "sha256:7043483ed7c1770a78a87643c29b184b7d7a5519ce452bbd75886db14f39d1ea",
                "image_digest": (
                    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
                    "sourcing-model@sha256:972ab6db779a788fa1e6825d33ff89a5a4cfd6569798a2257c1d5d051ec10176"
                ),
            },
            {
                "source_tree_hash": "sha256:666391b590779b0db0d716e34ffee00d628e955b7cf037cef28753d69267a8f8",
                "git_commit_sha": "2ee46b22b01c9895eab66a4a1e2e2c7ec652b98d",
                "manifest_hash": "sha256:cfa3e8520aaabee52237b4c3c79cc19c40a4d58baeef08f56b0864e6dbacf7fd",
                "image_digest": (
                    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
                    "sourcing-model@sha256:abc62f3571493e45231d55e8fa9cbb007058f3e8fcf2c53bd364756b5d2fbc22"
                ),
            },
            {
                "source_tree_hash": "sha256:d00b1d77ec6529a0f1ff40bd79187a32253ad29d3832eb9f7e330442e95e136b",
                "git_commit_sha": "c93ccc26fde54e2a4c6ebfdcb5426c7ad20f28b5",
                "manifest_hash": "sha256:418b6751d5498c87bc26e93a90b9ce815cbb37100ef079212917a05d72329731",
                "image_digest": (
                    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
                    "sourcing-model@sha256:33bf746e75cd2c426a6eb4a197ffa4ef4ca236937d4c9eeb2f427e353469febd"
                ),
            },
        ),
    },
    {
        "contract_id": "leadpoet-sourcing-wrapper-contract-v13",
        "contract_path": CONTRACT_V13_PATH,
        "contract_sha256": (
            "sha256:9ab93592ae1e969bd08e50c73708513968b601c2b95e8d661a67cdcd3674f5da"
        ),
        "parity_path": PARITY_FIXTURE_V13_PATH,
        "parity_sha256": (
            "sha256:22638a5804681b3305606844359e6e69112937c21bda1cd34bb5edde93cdc7f0"
        ),
        "release_identities": (
            {
                "source_tree_hash": "sha256:2a9ed2b6986bb46c940607e69f381d19f23188f872fbc971d92cc74dab80f4ed",
                "git_commit_sha": "5b28d3d998cf341d6f4544f404510cd98914bb50",
                "manifest_hash": "sha256:7ca780371750f9fa5d6075d8ec9b5888079db7c4fc7b39fc5c919820b5499a10",
                "image_digest": (
                    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
                    "sourcing-model@sha256:47804dbafc2833902552aa0a1572edff5d82e0b3d9385d5e50a7ed621137a4d8"
                ),
            },
            {
                "source_tree_hash": "sha256:2440f4d9cb507bfc2e91954997060bf4e2008f2ba347aaf3c2b9c6fb38816ce3",
                "git_commit_sha": "cc52fa923fa3225f550e095048b6bf817b1ae778",
                "manifest_hash": "sha256:8eea2512f1022a98661f93392830c39cb01d470413cde31b0f7de750035a2be8",
                "image_digest": (
                    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
                    "sourcing-model@sha256:5cb345b0657da51cfa171093ab3fda24b0cc8cebfad88e746e734e65701c6a31"
                ),
            },
        ),
    },
    {
        "contract_id": "leadpoet-sourcing-wrapper-contract-v12",
        "contract_path": CONTRACT_V12_PATH,
        "contract_sha256": (
            "sha256:d681d2100a570c1e22447e3ac8bba53806ce01ae1f4cdad6aeba8eb8b6abaff3"
        ),
        "parity_path": PARITY_FIXTURE_V12_PATH,
        "parity_sha256": (
            "sha256:82b2cbd1cf9cf346b144d0d5cee8ec8d9ca4c02d97a52da2914313a1a5718dea"
        ),
        "release_identities": (
            {
                "source_tree_hash": "sha256:bd616b9389440cb65cf547f3ec1842e0c471dea583b70be4ef04adc300641ea9",
                "git_commit_sha": "8dc2bab6df8ae2a769623028d1367964bf1d65bb",
                "manifest_hash": "sha256:938fe32a67051728f7b6cf73e31ea63db08f198f8b39f044b79b53f0b616be69",
                "image_digest": (
                    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
                    "sourcing-model@sha256:ce009b6d6ba425fb34b1ddb8a5afd575e1b5834d18c19a7f93cdce41338e6a56"
                ),
            },
        ),
    },
    {
        "contract_id": "leadpoet-sourcing-wrapper-contract-v11",
        "contract_path": CONTRACT_V11_PATH,
        "contract_sha256": (
            "sha256:2cd4d09b99db1f0ac523c3e57f361afb7c7ff1413392bd9aa5dfcee9efb81c01"
        ),
        "parity_path": PARITY_FIXTURE_V11_PATH,
        "parity_sha256": (
            "sha256:8b0d23b1664b5539e790c988afcb558c2aa4cf0ff925af0f7dbe2f9bc900fce4"
        ),
        # Exact source identity from the signed v11 release manifest. Older
        # releases predate semantic-v1's critical surface, so contract/parity
        # bytes alone must never select their legacy admission profile.
        "release_identities": (
            {
                "source_tree_hash": "sha256:641adf30cfb197276da018702688ab3378f69ec2e7f71b2245963f537c35aab3",
                "git_commit_sha": "74a29a984938e1a443bdd0d2eed2f41f737be1e6",
                "manifest_hash": "sha256:84b7f21f843c46a551e346693b2079bfee63fbc62f3a0e8db00339bb57d932e8",
                "image_digest": (
                    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
                    "sourcing-model@sha256:cff5ce5f9e95e749ea242cd8377d4479f017c68e78eac49fe7bb957997f09eff"
                ),
            },
        ),
    },
    {
        "contract_id": "leadpoet-sourcing-wrapper-contract-v8",
        "contract_path": CONTRACT_PATH,
        "contract_sha256": (
            "sha256:080e7b199c3e1d27ae080e497b541b560a2e12d383a709d453e7a2dd320b8dfc"
        ),
        "parity_path": PARITY_FIXTURE_PATH,
        "parity_sha256": (
            "sha256:5527186b45294135639619d99bfcf076ec98035670f68843244ccd18fc3f80fe"
        ),
        "release_identities": (
            {
                "source_tree_hash": "sha256:879ace5e05383dcfebf877d60d80f7e179017a7c487741990e896c1d63caed28",
                "git_commit_sha": "2d90daa8347daec34e8e7966eb6d208f47f52df2",
                "manifest_hash": "sha256:3f92e56236f4c5f583ca0b3f8cf6c2b42bcf41a7a06c3ec584a8d6b8ceee6caa",
                "image_digest": (
                    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
                    "sourcing-model@sha256:dd5710d30d589b657f4bd593d4d015bbfe47374283a862bbb2aef57455c3de4a"
                ),
            },
        ),
    },
    {
        "contract_id": "leadpoet-sourcing-wrapper-contract-v7",
        "contract_path": CONTRACT_V7_PATH,
        "contract_sha256": (
            "sha256:f2fea5a16de1dd1fafb1fa5259b161cd0dd8059fddaf30d8e9982d3eec391d10"
        ),
        "parity_path": PARITY_FIXTURE_V7_PATH,
        "parity_sha256": (
            "sha256:c39c48335a4877c091e6ca264f3f9411dbecd4992c09e9c77bdb789479076d3a"
        ),
        "release_identities": (
            {
                "source_tree_hash": "sha256:54ccdacb8200c750426d815c0c7d8e379096be5514d9aa6868550a40d05d0533",
                "git_commit_sha": "4dfd54ed1a3142dbfcdad6a3b2988c5136e4f50e",
                "manifest_hash": "sha256:c978a61d661f6281620ebf7c7775c52bea92254593a08eca2199a62791439092",
                "image_digest": (
                    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
                    "sourcing-model@sha256:4947b2972548cb8382636633acbaf9ecd22148da17f7ea664f237723433805f6"
                ),
            },
        ),
    },
)

# The model publisher retained the v55 contract id while adding the measured
# intent-source evidence surface. Keep both byte-exact v55 revisions so the
# active release and the older rollback releases remain independently bound.
REVIEWED_CONSUMER_ALTERNATE_SNAPSHOT_SPECS = (
    {
        "contract_id": "leadpoet-sourcing-wrapper-contract-v55",
        "contract_path": CONTRACT_V55_E55_PATH,
        "contract_sha256": (
            "sha256:b89eda998cf8cf3d9ee80c4ccd2bd4e10e37d6e4bdd7be80e2dc70492d2c0ffd"
        ),
        "parity_path": PARITY_FIXTURE_V55_E55_PATH,
        "parity_sha256": (
            "sha256:b75f79a8b7c3eb72c24b14ceab7c84442e394dd8c738a627dbbb22ed4bf4271a"
        ),
        "required_source_constants": {
            "sourcing_model/runtime_capabilities.py": {
                "CAPABILITY_CONTRACT_VERSION": (
                    "sourcing-model-runtime-capabilities:v3"
                ),
            },
        },
        "release_identities": (
            {
                "source_tree_hash": (
                    "sha256:491d6e76adf629b60d913062005191673f962db3cd5cd77223a68cf6262ac60f"
                ),
                "git_commit_sha": "e55e57f2be0ddadcc6b9c92c18b932dc2c354d21",
                "manifest_hash": (
                    "sha256:af68f0fbd29c77f9ffe686dcbddbc1e5dd1cab6c8725c7c9669de367bd592928"
                ),
                "image_digest": (
                    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
                    "sourcing-model@sha256:f1ae9bc0ba2cd55450e4c1b1bbdb0030514dbf5afd380f29a09d5e95bdb0ade5"
                ),
            },
        ),
        "positional_exact_signatures": True,
        "variadic_parameters": {
            "sourcing_model/corporate_filing_contract.py:"
            "build_corporate_filing_envelope": {
                "vararg": None,
                "kwarg": "payload",
            },
        },
    },
)


def load_wrapper_contract(path: Path | None = None) -> Dict[str, Any]:
    """Load and shape-check the reviewed model-owned contract snapshot."""
    document = json.loads(Path(path or CONTRACT_PATH).read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError(
            "Unsupported sourcing wrapper contract schema_version: "
            f"{document.get('schema_version')!r}"
        )
    for key in (
        "contract_id",
        "canonical_path",
        "parity_fixture_path",
        "required_files",
        "functions",
    ):
        if key not in document:
            raise ValueError(f"wrapper contract missing required key {key!r}")
    return document


def _reviewed_consumer_snapshot_from_spec(
    spec: Mapping[str, Any],
) -> Dict[str, Any]:
    contract_path = Path(spec["contract_path"])
    parity_path = Path(spec["parity_path"])
    contract_sha256 = _snapshot_sha256(contract_path)
    parity_sha256 = _snapshot_sha256(parity_path)
    if contract_sha256 != spec["contract_sha256"]:
        raise ValueError(
            f"reviewed sourcing contract hash differs: {contract_path.name}"
        )
    if parity_sha256 != spec["parity_sha256"]:
        raise ValueError(
            f"reviewed sourcing parity hash differs: {parity_path.name}"
        )
    document = load_wrapper_contract(contract_path)
    contract_id = str(spec["contract_id"])
    if document["contract_id"] != contract_id:
        raise ValueError(
            f"reviewed sourcing contract id differs: {contract_path.name}"
        )
    if not parity_path.is_file():
        raise ValueError(
            f"reviewed sourcing parity snapshot is missing: {parity_path.name}"
        )
    release_identities: list[Dict[str, str]] = []
    for value in spec.get("release_identities") or ():
        release = {str(key): str(item) for key, item in dict(value).items()}
        if set(release) != {
            "source_tree_hash",
            "git_commit_sha",
            "manifest_hash",
            "image_digest",
        }:
            raise ValueError(
                f"reviewed legacy release identity is malformed: {contract_id}"
            )
        if (
            re.fullmatch(r"sha256:[0-9a-f]{64}", release["source_tree_hash"])
            is None
            or re.fullmatch(r"[0-9a-f]{40}", release["git_commit_sha"])
            is None
            or re.fullmatch(r"sha256:[0-9a-f]{64}", release["manifest_hash"])
            is None
            or re.fullmatch(
                r"[^\s]+@sha256:[0-9a-f]{64}", release["image_digest"]
            )
            is None
        ):
            raise ValueError(
                f"reviewed legacy release identity is invalid: {contract_id}"
            )
        release_identities.append(release)
    if len({item["source_tree_hash"] for item in release_identities}) != len(
        release_identities
    ):
        raise ValueError(
            f"duplicate reviewed legacy source identity: {contract_id}"
        )
    return {
        "contract": document,
        "contract_path": contract_path,
        "contract_sha256": contract_sha256,
        "parity_path": parity_path,
        "parity_sha256": parity_sha256,
        "release_identities": tuple(release_identities),
        "positional_exact_signatures": bool(
            spec.get("positional_exact_signatures", False)
        ),
        "variadic_parameters": dict(spec.get("variadic_parameters") or {}),
        "required_source_constants": {
            str(relative): {
                str(name): value
                for name, value in dict(expected_values).items()
            }
            for relative, expected_values in dict(
                spec.get("required_source_constants") or {}
            ).items()
        },
    }


def reviewed_consumer_snapshots() -> Dict[str, Dict[str, Any]]:
    """Return the primary exact contract/parity pair for each contract id."""

    snapshots: Dict[str, Dict[str, Any]] = {}
    for spec in REVIEWED_CONSUMER_SNAPSHOT_SPECS:
        snapshot = _reviewed_consumer_snapshot_from_spec(spec)
        contract_id = str(snapshot["contract"]["contract_id"])
        if contract_id in snapshots:
            raise ValueError(
                f"duplicate reviewed sourcing wrapper contract id: {contract_id}"
            )
        snapshots[contract_id] = snapshot
    return snapshots


def reviewed_consumer_profiles() -> tuple[Dict[str, Any], ...]:
    """Return every byte-exact reviewed profile, including same-id revisions."""

    primary = reviewed_consumer_snapshots()
    profiles = list(primary.values())
    seen_pairs = {
        (snapshot["contract_sha256"], snapshot["parity_sha256"])
        for snapshot in profiles
    }
    for spec in REVIEWED_CONSUMER_ALTERNATE_SNAPSHOT_SPECS:
        snapshot = _reviewed_consumer_snapshot_from_spec(spec)
        contract_id = str(snapshot["contract"]["contract_id"])
        if contract_id not in primary:
            raise ValueError(
                f"alternate reviewed contract id has no primary: {contract_id}"
            )
        pair = (snapshot["contract_sha256"], snapshot["parity_sha256"])
        if pair in seen_pairs:
            raise ValueError(f"duplicate reviewed sourcing profile: {contract_id}")
        seen_pairs.add(pair)
        profiles.append(snapshot)
    return tuple(profiles)


def _snapshot_sha256(path: Path) -> str:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"reviewed sourcing snapshot is unreadable: {path.name}") from exc
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _excluded_compatibility_source_path_v1(relative: str) -> bool:
    """Mirror the immutable private-artifact tree exclusion contract."""

    parts = relative.split("/")
    if any(
        part
        in {
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".venv",
            "venv",
        }
        for part in parts
    ):
        return True
    if relative.endswith((".pyc", ".pyo", ".env", ".pem", ".key")):
        return True
    return relative == ".env" or relative.startswith(".env.")


def compute_compatibility_source_tree_hash_v1(root: Path) -> str:
    """Compute the canonical private-artifact identity used by admission.

    Admission owns this computation instead of trusting a caller-provided
    digest. ``private_runtime.compute_private_source_tree_hash`` delegates to
    this function so signing, extraction, cache lookup, and compatibility
    decisions cannot silently drift onto different tree-hash algorithms.
    """

    resolved = Path(root).expanduser().resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError("sourcing compatibility source root is unavailable")
    digest_inputs: list[tuple[str, str]] = []
    try:
        for file_path in sorted(resolved.rglob("*")):
            if not file_path.is_file():
                continue
            relative = file_path.relative_to(resolved).as_posix()
            if _excluded_compatibility_source_path_v1(relative):
                continue
            digest_inputs.append((relative, _snapshot_sha256(file_path)))
    except OSError as exc:
        raise ValueError(
            "sourcing compatibility source tree is unreadable"
        ) from exc
    return _sha256_json(digest_inputs)


def _resolve_reviewed_consumer_contract_pair(
    root: Path,
) -> Dict[str, Any] | None:
    """Resolve exact reviewed contract/parity bytes without admitting source."""

    root = Path(root)
    matches: list[Dict[str, Any]] = []
    for snapshot in reviewed_consumer_profiles():
        document = snapshot["contract"]
        candidate_contract_path = root / str(document["canonical_path"])
        candidate_parity_path = root / str(document["parity_fixture_path"])
        try:
            if (
                candidate_contract_path.is_file()
                and candidate_parity_path.is_file()
                and candidate_contract_path.read_bytes()
                == Path(snapshot["contract_path"]).read_bytes()
                and candidate_parity_path.read_bytes()
                == Path(snapshot["parity_path"]).read_bytes()
            ):
                matches.append(snapshot)
        except OSError:
            continue
    return matches[0] if len(matches) == 1 else None


def _reviewed_consumer_snapshot_for_source_hash(
    root: Path,
    *,
    source_tree_hash: str,
    manifest: Mapping[str, Any] | None = None,
) -> Dict[str, Any] | None:
    """Select legacy admission only for an exact signed source identity."""

    snapshot = _resolve_reviewed_consumer_contract_pair(root)
    if snapshot is None:
        return None
    matching = _reviewed_legacy_release_identities(
        snapshot,
        source_tree_hash=source_tree_hash,
    )
    if not matching:
        return None
    manifest_document = dict(manifest or {})
    if not manifest_document:
        return snapshot
    manifest_identity = {
        field: str(manifest_document.get(field) or "")
        for field in (
            "model_artifact_hash",
            "git_commit_sha",
            "manifest_hash",
            "image_digest",
        )
    }
    for release in matching:
        if manifest_identity == {
            "model_artifact_hash": str(release["source_tree_hash"]),
            "git_commit_sha": str(release["git_commit_sha"]),
            "manifest_hash": str(release["manifest_hash"]),
            "image_digest": str(release["image_digest"]),
        }:
            return snapshot
    # A profiled historical source identity is reserved. Once a signed
    # manifest is supplied it must be the exact reviewed release tuple; a
    # hybrid must never escape into the semantic fallback path.
    raise ValueError(
        "reviewed legacy source manifest identity differs from signed release"
    )


def _reviewed_legacy_release_identities(
    snapshot: Mapping[str, Any],
    *,
    source_tree_hash: str,
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        release
        for release in snapshot["release_identities"]
        if str(release["source_tree_hash"]) == str(source_tree_hash)
    )


def validate_reviewed_legacy_release_manifest_identity_v1(
    compatibility_receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    """Reject rebuilt/hybrid manifests that claim a legacy admission mode."""

    receipt = dict(compatibility_receipt)
    if receipt.get("admission_mode") != "legacy_exact":
        return
    matching_snapshots = tuple(
        snapshot
        for snapshot in reviewed_consumer_profiles()
        if str(snapshot["contract_sha256"]) == str(receipt.get("contract_hash"))
        and str(snapshot["parity_sha256"]) == str(receipt.get("parity_hash"))
    )
    if (
        len(matching_snapshots) != 1
        or str(receipt.get("contract_id") or "")
        != str(matching_snapshots[0]["contract"]["contract_id"])
    ):
        raise ValueError(
            "legacy compatibility receipt differs from its reviewed signed release"
        )
    document = dict(manifest)
    manifest_identity = {
        field: str(document.get(field) or "")
        for field in (
            "model_artifact_hash",
            "git_commit_sha",
            "manifest_hash",
            "image_digest",
        )
    }
    matching_releases = _reviewed_legacy_release_identities(
        matching_snapshots[0],
        source_tree_hash=str(receipt.get("source_tree_hash") or ""),
    )
    if matching_releases and not document:
        return
    for release in matching_releases:
        if manifest_identity == {
            "model_artifact_hash": str(release["source_tree_hash"]),
            "git_commit_sha": str(release["git_commit_sha"]),
            "manifest_hash": str(release["manifest_hash"]),
            "image_digest": str(release["image_digest"]),
        }:
            return
    raise ValueError(
        "legacy compatibility receipt differs from its reviewed signed release"
    )


def resolve_reviewed_consumer_snapshot(root: Path) -> Dict[str, Any] | None:
    """Resolve a tree only when its pair and signed source are both reviewed."""

    root = Path(root)
    return _reviewed_consumer_snapshot_for_source_hash(
        root,
        source_tree_hash=compute_compatibility_source_tree_hash_v1(root),
    )


def _function_signature(node: ast.AST) -> Dict[str, Any]:
    args = getattr(node, "args", None)
    if args is None:
        return {
            "params": [],
            "all_params": [],
            "positional_only": [],
            "vararg": None,
            "kwarg": None,
            "required_positional": 0,
            "required_keyword_only": [],
        }
    positional = list(args.posonlyargs + args.args)
    required_positional = max(0, len(positional) - len(args.defaults))
    required_keyword_only = [
        item.arg
        for item, default in zip(args.kwonlyargs, args.kw_defaults)
        if default is None
    ]
    return {
        "params": [item.arg for item in positional],
        "all_params": [
            item.arg for item in [*positional, *args.kwonlyargs]
        ],
        "positional_only": [item.arg for item in args.posonlyargs],
        "vararg": args.vararg.arg if args.vararg is not None else None,
        "kwarg": args.kwarg.arg if args.kwarg is not None else None,
        "required_positional": required_positional,
        "required_keyword_only": required_keyword_only,
    }


def _int_constant(value: ast.AST | None) -> int | None:
    if (
        isinstance(value, ast.Constant)
        and isinstance(value.value, int)
        and not isinstance(value.value, bool)
    ):
        return value.value
    return None


def _module_symbols(tree: ast.Module) -> Dict[str, Any]:
    """Top-level function param-lists and integer constant assignments.

    Constants follow last-assignment-wins module semantics: a plain or
    annotated assignment of an integer literal records the value, and any
    later top-level rebinding of the same name to a non-literal (call,
    expression, augmented assignment) discards it — the value the runtime
    would see is no longer statically verifiable, which downstream reads as
    a missing-constant violation rather than silently trusting an earlier
    literal.
    """
    functions: Dict[str, Dict[str, Any]] = {}
    constants: Dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = {
                **_function_signature(node),
                "is_async": isinstance(node, ast.AsyncFunctionDef),
            }
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                value = _int_constant(node.value)
                if value is not None:
                    constants[target.id] = value
                    continue
                constants.pop(target.id, None)
                # Simple top-level alias (``qualify = _qualify_impl``) is a
                # runtime-valid rebinding — carry the aliased function's
                # surface instead of reporting it missing. Anything else
                # rebinding a function name makes it unverifiable.
                if isinstance(node.value, ast.Name) and node.value.id in functions:
                    functions[target.id] = dict(functions[node.value.id])
                else:
                    functions.pop(target.id, None)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.value is not None:
                value = _int_constant(node.value)
                if value is not None:
                    constants[node.target.id] = value
                else:
                    constants.pop(node.target.id, None)
                    functions.pop(node.target.id, None)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            constants.pop(node.target.id, None)
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants.pop(target.id, None)
                    functions.pop(target.id, None)
    return {"functions": functions, "constants": constants}


def _module_bound_imports(tree: ast.Module) -> set[str]:
    """Return dotted modules bound by plain, unaliased ``import a.b``.

    The site wrapper reaches ``clients.urllib.request.urlopen``. That chain is
    available after ``import urllib.request`` but not after
    ``from urllib.request import urlopen`` or
    ``import urllib.request as request``. Match the release bundle's contract
    check exactly.
    """
    bound: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Import):
            continue
        for alias in node.names:
            if alias.asname:
                continue
            parts = alias.name.split(".")
            for index in range(1, len(parts) + 1):
                bound.add(".".join(parts[:index]))
    return bound


def _module_scope_bindings(node: ast.AST) -> set[str]:
    """Return names rebound while evaluating ``node`` at module scope.

    Function and class bodies have their own scopes and are intentionally not
    traversed. Their names, decorators, bases, defaults, and annotations are
    evaluated at module scope and therefore still count.
    """

    bound: set[str] = set()

    class BindingVisitor(ast.NodeVisitor):
        def visit_Name(self, item: ast.Name) -> None:  # noqa: N802
            if isinstance(item.ctx, (ast.Store, ast.Del)):
                bound.add(item.id)

        def _visit_function_binding(
            self, item: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            bound.add(item.name)
            for decorator in item.decorator_list:
                self.visit(decorator)
            for default in item.args.defaults:
                self.visit(default)
            for default in item.args.kw_defaults:
                if default is not None:
                    self.visit(default)
            if item.returns is not None:
                self.visit(item.returns)

        def visit_FunctionDef(self, item: ast.FunctionDef) -> None:  # noqa: N802
            self._visit_function_binding(item)

        def visit_AsyncFunctionDef(  # noqa: N802
            self, item: ast.AsyncFunctionDef
        ) -> None:
            self._visit_function_binding(item)

        def visit_ClassDef(self, item: ast.ClassDef) -> None:  # noqa: N802
            bound.add(item.name)
            for decorator in item.decorator_list:
                self.visit(decorator)
            for base in item.bases:
                self.visit(base)
            for keyword in item.keywords:
                self.visit(keyword.value)

        def visit_Lambda(self, item: ast.Lambda) -> None:  # noqa: N802
            for default in item.args.defaults:
                self.visit(default)
            for default in item.args.kw_defaults:
                if default is not None:
                    self.visit(default)

        def visit_Import(self, item: ast.Import) -> None:  # noqa: N802
            for alias in item.names:
                bound.add(alias.asname or alias.name.split(".", 1)[0])

        def visit_ImportFrom(self, item: ast.ImportFrom) -> None:  # noqa: N802
            for alias in item.names:
                if alias.name != "*":
                    bound.add(alias.asname or alias.name)

    BindingVisitor().visit(node)
    return bound


def _binding_nodes(tree: ast.Module, name: str) -> list[ast.AST]:
    return [node for node in tree.body if name in _module_scope_bindings(node)]


def _required_import_matches_v1(
    node: ast.AST,
    requirement: Mapping[str, Any],
) -> bool:
    """Match one exact consumer-required import and its runtime binding."""

    binding = str(requirement.get("binding") or "")
    kind = str(requirement.get("kind") or "")
    if kind == "import" and isinstance(node, ast.Import):
        aliases = [
            alias
            for alias in node.names
            if (alias.asname or alias.name.split(".", 1)[0]) == binding
        ]
        expected_module = str(requirement.get("module") or "")
        expected_alias = (
            None if binding == expected_module.split(".", 1)[0] else binding
        )
        return (
            len(aliases) == 1
            and aliases[0].name == expected_module
            and aliases[0].asname == expected_alias
        )
    if kind == "from" and isinstance(node, ast.ImportFrom):
        aliases = [
            alias
            for alias in node.names
            if (alias.asname or alias.name) == binding
        ]
        expected_name = str(requirement.get("name") or "")
        expected_alias = None if binding == expected_name else binding
        return (
            len(aliases) == 1
            and int(node.level or 0) == int(requirement.get("level") or 0)
            and str(node.module or "") == str(requirement.get("module") or "")
            and aliases[0].name == expected_name
            and aliases[0].asname == expected_alias
        )
    return False


def _required_import_binding_valid_v1(
    tree: ast.Module,
    requirement: Mapping[str, Any],
) -> bool:
    binding = str(requirement.get("binding") or "")
    nodes = _binding_nodes(tree, binding)
    expected_module = str(requirement.get("module") or "")
    if (
        requirement.get("kind") == "import"
        and binding == expected_module.split(".", 1)[0]
    ):
        aliases = [
            alias
            for node in nodes
            if isinstance(node, ast.Import)
            for alias in node.names
            if (alias.asname or alias.name.split(".", 1)[0]) == binding
        ]
        return bool(
            aliases
            and all(isinstance(node, ast.Import) for node in nodes)
            and all(
                alias.asname is None
                and alias.name.split(".", 1)[0] == binding
                for alias in aliases
            )
            and any(alias.name == expected_module for alias in aliases)
        )
    return len(nodes) == 1 and _required_import_matches_v1(
        nodes[0], requirement
    )


def _inert_definition_expression_v1(node: ast.AST | None) -> bool:
    if node is None:
        return True
    forbidden = (
        ast.Await,
        ast.Call,
        ast.DictComp,
        ast.GeneratorExp,
        ast.Lambda,
        ast.ListComp,
        ast.NamedExpr,
        ast.SetComp,
        ast.Yield,
        ast.YieldFrom,
    )
    return not any(isinstance(item, forbidden) for item in ast.walk(node))


def _inert_additive_function_v1(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    """Whether an unconsumed function is inert while its module imports."""

    if (
        node.decorator_list
        or (node.name.startswith("__") and node.name.endswith("__"))
    ):
        return False
    defaults = [
        *node.args.defaults,
        *(item for item in node.args.kw_defaults if item is not None),
    ]
    try:
        for default in defaults:
            ast.literal_eval(default)
    except (TypeError, ValueError):
        return False
    annotations = [
        *(item.annotation for item in node.args.posonlyargs),
        *(item.annotation for item in node.args.args),
        *(item.annotation for item in node.args.kwonlyargs),
        node.args.vararg.annotation if node.args.vararg is not None else None,
        node.args.kwarg.annotation if node.args.kwarg is not None else None,
        node.returns,
    ]
    return all(_inert_definition_expression_v1(item) for item in annotations)


def _inert_additive_literal_v1(node: ast.AST) -> bool:
    value: ast.AST | None = None
    annotation: ast.AST | None = None
    name = ""
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ):
        name = node.targets[0].id
        value = node.value
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        name = node.target.id
        value = node.value
        annotation = node.annotation
    if (
        value is None
        or (name.startswith("__") and name.endswith("__"))
        or not _inert_definition_expression_v1(annotation)
    ):
        return False
    try:
        ast.literal_eval(value)
    except (TypeError, ValueError):
        return False
    return True


def _function_local_names_v1(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
) -> set[str]:
    names = {
        argument.arg
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        )
    }
    if node.args.vararg is not None:
        names.add(node.args.vararg.arg)
    if node.args.kwarg is not None:
        names.add(node.args.kwarg.arg)
    globals_: set[str] = set()
    nonlocals: set[str] = set()

    class LocalVisitor(ast.NodeVisitor):
        def visit_Name(self, item: ast.Name) -> None:  # noqa: N802
            if isinstance(item.ctx, (ast.Store, ast.Del)):
                names.add(item.id)

        def visit_Global(self, item: ast.Global) -> None:  # noqa: N802
            globals_.update(item.names)

        def visit_Nonlocal(self, item: ast.Nonlocal) -> None:  # noqa: N802
            nonlocals.update(item.names)

        def visit_Import(self, item: ast.Import) -> None:  # noqa: N802
            names.update(
                alias.asname or alias.name.split(".", 1)[0]
                for alias in item.names
            )

        def visit_ImportFrom(self, item: ast.ImportFrom) -> None:  # noqa: N802
            names.update(
                alias.asname or alias.name
                for alias in item.names
                if alias.name != "*"
            )

        def visit_FunctionDef(self, item: ast.FunctionDef) -> None:  # noqa: N802
            names.add(item.name)

        def visit_AsyncFunctionDef(  # noqa: N802
            self, item: ast.AsyncFunctionDef
        ) -> None:
            names.add(item.name)

        def visit_ClassDef(self, item: ast.ClassDef) -> None:  # noqa: N802
            names.add(item.name)

        def visit_Lambda(self, item: ast.Lambda) -> None:  # noqa: N802
            return

        def _visit_comprehension(self, item: ast.AST) -> None:
            for generator in item.generators:
                self.visit(generator.iter)
                for condition in generator.ifs:
                    self.visit(condition)
            if isinstance(item, ast.DictComp):
                self.visit(item.key)
                self.visit(item.value)
            else:
                self.visit(item.elt)

        visit_ListComp = _visit_comprehension  # noqa: N815
        visit_SetComp = _visit_comprehension  # noqa: N815
        visit_GeneratorExp = _visit_comprehension  # noqa: N815
        visit_DictComp = _visit_comprehension  # noqa: N815

        def visit_ExceptHandler(self, item: ast.ExceptHandler) -> None:  # noqa: N802
            if isinstance(item.name, str):
                names.add(item.name)
            self.generic_visit(item)

    visitor = LocalVisitor()
    for statement in node.body if not isinstance(node, ast.Lambda) else (node.body,):
        visitor.visit(statement)
    return names - globals_ - nonlocals


def _loaded_global_names_v1(node: ast.AST) -> set[str]:
    loaded: set[str] = set()
    scopes: list[set[str]] = []

    class GlobalLoadVisitor(ast.NodeVisitor):
        def visit_Name(self, item: ast.Name) -> None:  # noqa: N802
            if isinstance(item.ctx, ast.Load) and not any(
                item.id in scope for scope in reversed(scopes)
            ):
                loaded.add(item.id)

        def _visit_function(
            self, item: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            for expression in (
                *item.decorator_list,
                *item.args.defaults,
                *(value for value in item.args.kw_defaults if value is not None),
                *(argument.annotation for argument in item.args.posonlyargs),
                *(argument.annotation for argument in item.args.args),
                *(argument.annotation for argument in item.args.kwonlyargs),
                item.args.vararg.annotation if item.args.vararg is not None else None,
                item.args.kwarg.annotation if item.args.kwarg is not None else None,
                item.returns,
            ):
                if expression is not None:
                    self.visit(expression)
            scopes.append(_function_local_names_v1(item))
            for statement in item.body:
                self.visit(statement)
            scopes.pop()

        def visit_FunctionDef(self, item: ast.FunctionDef) -> None:  # noqa: N802
            self._visit_function(item)

        def visit_AsyncFunctionDef(  # noqa: N802
            self, item: ast.AsyncFunctionDef
        ) -> None:
            self._visit_function(item)

        def visit_Lambda(self, item: ast.Lambda) -> None:  # noqa: N802
            for expression in (
                *item.args.defaults,
                *(value for value in item.args.kw_defaults if value is not None),
            ):
                self.visit(expression)
            scopes.append(_function_local_names_v1(item))
            self.visit(item.body)
            scopes.pop()

        def _visit_comprehension(self, item: ast.AST) -> None:
            scopes.append(set())
            for generator in item.generators:
                self.visit(generator.iter)
                scopes[-1].update(
                    target.id
                    for target in ast.walk(generator.target)
                    if isinstance(target, ast.Name)
                    and isinstance(target.ctx, ast.Store)
                )
                for condition in generator.ifs:
                    self.visit(condition)
            if isinstance(item, ast.DictComp):
                self.visit(item.key)
                self.visit(item.value)
            else:
                self.visit(item.elt)
            scopes.pop()

        visit_ListComp = _visit_comprehension  # noqa: N815
        visit_SetComp = _visit_comprehension  # noqa: N815
        visit_GeneratorExp = _visit_comprehension  # noqa: N815
        visit_DictComp = _visit_comprehension  # noqa: N815

        def visit_ClassDef(self, item: ast.ClassDef) -> None:  # noqa: N802
            for expression in (*item.decorator_list, *item.bases):
                self.visit(expression)
            for keyword in item.keywords:
                self.visit(keyword.value)
            for statement in item.body:
                self.visit(statement)

    GlobalLoadVisitor().visit(node)
    return loaded


def _critical_binding_slice_v1(
    tree: ast.Module,
    *,
    roots: list[str],
    normalized_literals: set[str] | None = None,
    strip_function_bodies: bool = False,
) -> tuple[str, list[str]]:
    """Hash the closed consumer-critical top-level binding slice.

    Imports, classes, and any definition-time executable statement are pinned.
    Only unrelated undecorated functions with inert defaults/annotations and
    unrelated literal constants may remain outside the slice. Dependencies of
    every pinned binding are followed recursively and serialized in source
    order. This permits harmless additive model code without allowing import-
    time rebinding of a consumed adapter or decision surface.
    """

    violations: list[str] = []
    nodes = list(tree.body)
    full_nodes = deepcopy(nodes)
    stripped_nodes = full_nodes
    if strip_function_bodies:
        class FunctionBodyStripper(ast.NodeTransformer):
            def visit_FunctionDef(  # noqa: N802
                self, item: ast.FunctionDef
            ) -> ast.AST:
                item = self.generic_visit(item)
                item.body = [ast.Pass()]
                return item

            def visit_AsyncFunctionDef(  # noqa: N802
                self, item: ast.AsyncFunctionDef
            ) -> ast.AST:
                item = self.generic_visit(item)
                item.body = [ast.Pass()]
                return item

        stripper = FunctionBodyStripper()
        stripped_nodes = [stripper.visit(node) for node in deepcopy(nodes)]
    bindings: dict[str, list[int]] = {}
    for index, node in enumerate(nodes):
        for name in _module_scope_bindings(node):
            bindings.setdefault(name, []).append(index)

    selected: set[int] = set()
    for root in roots:
        indexes = bindings.get(root, [])
        if len(indexes) != 1:
            violations.append(
                f"critical binding {root} must have one definition, found "
                f"{len(indexes)}"
            )
        else:
            selected.add(indexes[0])

    # Pin the import-time environment. A newly added import/class/evaluated
    # expression therefore changes the slice even when no consumed function
    # refers to its bound name directly.
    for index, node in enumerate(nodes):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef)):
            selected.add(index)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _inert_additive_function_v1(node):
                selected.add(index)
        elif _inert_additive_literal_v1(node):
            continue
        elif (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        else:
            selected.add(index)

    visited_names: set[str] = set()
    import_time_functions: set[int] = set()
    while True:
        loaded = {
            name
            for index in selected
            for name in _loaded_global_names_v1(
                full_nodes[index]
                if index in import_time_functions
                else stripped_nodes[index]
            )
        }
        pending = loaded - visited_names
        if not pending:
            break
        visited_names.update(pending)
        for name in sorted(pending):
            indexes = bindings.get(name, [])
            if not indexes:
                continue
            if len(indexes) != 1:
                violations.append(
                    f"critical dependency {name} must have one binding, found "
                    f"{len(indexes)}"
                )
                continue
            index = indexes[0]
            selected.add(index)
            if strip_function_bodies and isinstance(
                nodes[index], (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                # A locally bound function referenced by module/class
                # definition-time code may execute while the module imports.
                # Its body and recursive dependencies are therefore authority,
                # unlike ordinary model-owned runtime callable bodies.
                import_time_functions.add(index)

    normalized_names = set(normalized_literals or ())

    class LiteralNormalizer(ast.NodeTransformer):
        @staticmethod
        def _without_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]
            return body or [ast.Pass()]

        def visit_FunctionDef(self, item: ast.FunctionDef) -> ast.AST:  # noqa: N802
            item = self.generic_visit(item)
            item.body = self._without_docstring(item.body)
            return item

        def visit_AsyncFunctionDef(  # noqa: N802
            self, item: ast.AsyncFunctionDef
        ) -> ast.AST:
            item = self.generic_visit(item)
            item.body = self._without_docstring(item.body)
            return item

        def visit_ClassDef(self, item: ast.ClassDef) -> ast.AST:  # noqa: N802
            item = self.generic_visit(item)
            item.body = self._without_docstring(item.body)
            return item

        def visit_Assign(self, item: ast.Assign) -> ast.AST:  # noqa: N802
            item = self.generic_visit(item)
            if (
                len(item.targets) == 1
                and isinstance(item.targets[0], ast.Name)
                and item.targets[0].id in normalized_names
            ):
                item.value = ast.Constant(
                    value=f"<consumer-release-label:{item.targets[0].id}>"
                )
            return item

        def visit_AnnAssign(self, item: ast.AnnAssign) -> ast.AST:  # noqa: N802
            item = self.generic_visit(item)
            if isinstance(item.target, ast.Name) and item.target.id in normalized_names:
                item.value = ast.Constant(
                    value=f"<consumer-release-label:{item.target.id}>"
                )
            return item

    normalizer = LiteralNormalizer()
    normalized = ast.Module(
        body=[
            normalizer.visit(
                full_nodes[index]
                if index in import_time_functions
                else stripped_nodes[index]
            )
            for index in range(len(nodes))
            if index in selected
        ],
        type_ignores=[],
    )
    return _ast_sha256(normalized), violations


def _literal_module_constants(
    tree: ast.Module,
    *,
    names: set[str],
) -> Dict[str, Any]:
    """Resolve exact constants with deterministic module last-write semantics.

    Only direct, single-name literal assignments are trusted. Conditional or
    dynamic writes poison the value unless a later direct literal assignment
    deterministically overwrites them.
    """

    constants: Dict[str, Any] = {}
    for node in tree.body:
        assigned_name = ""
        value_node: ast.AST | None = None
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            assigned_name = node.targets[0].id
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(
            node.target, ast.Name
        ):
            assigned_name = node.target.id
            value_node = node.value

        if assigned_name in names:
            if value_node is None:
                constants.pop(assigned_name, None)
                continue
            try:
                constants[assigned_name] = ast.literal_eval(value_node)
            except (TypeError, ValueError):
                constants.pop(assigned_name, None)
            continue

        for rebound_name in _module_scope_bindings(node) & names:
            constants.pop(rebound_name, None)
    return constants


def _unique_literal_binding_v1(tree: ast.Module, name: str) -> Any:
    if len(_binding_nodes(tree, name)) != 1:
        raise ValueError("binding is not unique")
    values = _literal_module_constants(tree, names={name})
    if name not in values:
        raise ValueError("binding is not a direct literal")
    return values[name]


def _same_literal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        if not isinstance(actual, (list, tuple)):
            return False
        return len(actual) == len(expected) and all(
            _same_literal(left, right)
            for left, right in zip(actual, expected)
        )
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return actual.keys() == expected.keys() and all(
            _same_literal(actual[key], expected[key]) for key in expected
        )
    return type(actual) is type(expected) and actual == expected


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _ast_sha256(node: ast.AST) -> str:
    payload = ast.dump(
        node,
        annotate_fields=True,
        include_attributes=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def semantic_compatibility_policy_v1() -> Dict[str, Any]:
    """Load the consumer-owned semantic admission policy.

    Contract ids, fixture-set ids, taxonomy revisions, and routing catalog
    revisions are deliberately absent. They are model-owned release labels.
    This policy freezes only the ABI and security properties the Leadpoet
    runtime directly consumes.
    """

    try:
        document = json.loads(
            SEMANTIC_COMPATIBILITY_POLICY_V1_PATH.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            "semantic sourcing compatibility policy is unavailable"
        ) from exc
    if not isinstance(document, dict):
        raise ValueError("semantic sourcing compatibility policy is invalid")
    if (
        document.get("schema_version")
        != SEMANTIC_COMPATIBILITY_POLICY_SCHEMA_V1
        or isinstance(document.get("contract_schema_major"), bool)
        or document.get("contract_schema_major") != 1
    ):
        raise ValueError("semantic sourcing compatibility policy is invalid")
    if (
        document.get("consumer_api_version")
        != SEMANTIC_COMPATIBILITY_CONSUMER_API_V1
    ):
        raise ValueError(
            "semantic sourcing compatibility consumer API is unsupported"
        )
    runtime_invariants = document.get("runtime_invariants")
    adapter_dependencies = (
        runtime_invariants.get("adapter_dependencies")
        if isinstance(runtime_invariants, Mapping)
        else None
    )
    build_query_probe = (
        adapter_dependencies.get("build_query")
        if isinstance(adapter_dependencies, Mapping)
        else None
    )
    if (
        not isinstance(runtime_invariants, Mapping)
        or runtime_invariants.get("schema_version")
        != SEMANTIC_COMPATIBILITY_RUNTIME_INVARIANTS_SCHEMA_V1
        or not isinstance(adapter_dependencies, Mapping)
        or set(adapter_dependencies) != {"build_query", "flow_modes"}
        or not isinstance(build_query_probe, Mapping)
        or set(build_query_probe) != {"icp", "source"}
        or not isinstance(build_query_probe.get("icp"), Mapping)
        or not isinstance(build_query_probe.get("source"), str)
        or adapter_dependencies.get("flow_modes") != ["branch", "legacy"]
        or not isinstance(runtime_invariants.get("company_fit"), Mapping)
        or not isinstance(
            runtime_invariants.get("runtime_capabilities"), Mapping
        )
    ):
        raise ValueError(
            "semantic sourcing compatibility runtime invariants are invalid"
        )
    slices = document.get("critical_binding_slices")
    if not isinstance(slices, Mapping) or not slices or not all(
        isinstance(relative, str)
        and isinstance(specification, Mapping)
        and isinstance(specification.get("roots"), list)
        and bool(specification.get("roots"))
        and all(isinstance(name, str) and name for name in specification["roots"])
        and len(set(specification["roots"])) == len(specification["roots"])
        and re.fullmatch(
            r"sha256:[0-9a-f]{64}", str(specification.get("sha256") or "")
        )
        is not None
        for relative, specification in slices.items()
    ):
        raise ValueError(
            "semantic sourcing compatibility critical binding policy is invalid"
        )
    import_time_slices = document.get("import_time_binding_slices")
    if not isinstance(import_time_slices, Mapping) or not import_time_slices or not all(
        isinstance(relative, str)
        and re.fullmatch(r"sha256:[0-9a-f]{64}", str(digest or "")) is not None
        for relative, digest in import_time_slices.items()
    ):
        raise ValueError(
            "semantic sourcing compatibility import-time binding policy is invalid"
        )
    dispatch_v3 = document.get("additive_dispatch_custody_v3")
    if not isinstance(dispatch_v3, Mapping):
        raise ValueError(
            "semantic sourcing compatibility typed dispatch policy is invalid"
        )
    if (
        dispatch_v3.get("contract_id")
        != ADDITIVE_DISPATCH_CUSTODY_V3_CONTRACT_ID
        or dispatch_v3.get("contract_sha256")
        != ADDITIVE_DISPATCH_CUSTODY_V3_CONTRACT_SHA256
        or dispatch_v3.get("parity_sha256")
        != ADDITIVE_DISPATCH_CUSTODY_V3_PARITY_SHA256
        or dispatch_v3.get("contract_snapshot_path")
        != "research_lab/sourcing_model_contract_v68.json"
        or dispatch_v3.get("parity_snapshot_path")
        != "research_lab/sourcing_model_parity_fixtures_v28.json"
    ):
        raise ValueError(
            "semantic sourcing compatibility typed dispatch identity is invalid"
        )
    for key in ("callables", "exact_constants", "critical_binding_slices"):
        value = dispatch_v3.get(key)
        if not isinstance(value, Mapping) or not value:
            raise ValueError(
                "semantic sourcing compatibility typed dispatch policy is invalid"
            )
    if (
        dict(
            dict(dispatch_v3["exact_constants"]).get(
                "sourcing_model/routing/compiler.py"
            )
            or {}
        ).get("COMPILER_VERSION")
        != ADDITIVE_DISPATCH_CUSTODY_V3_ROUTING_COMPILER_VERSION
    ):
        raise ValueError(
            "semantic sourcing compatibility typed dispatch compiler is invalid"
        )
    for relative, specification in (
        dispatch_v3.get("critical_binding_slices") or {}
    ).items():
        if (
            not isinstance(relative, str)
            or not isinstance(specification, Mapping)
            or not isinstance(specification.get("roots"), list)
            or not specification.get("roots")
            or len(set(specification["roots"])) != len(specification["roots"])
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}", str(specification.get("sha256") or "")
            )
            is None
        ):
            raise ValueError(
                "semantic sourcing compatibility typed dispatch slices are invalid"
            )
    metadata_binding = dispatch_v3.get("metadata_binding")
    if (
        not isinstance(metadata_binding, Mapping)
        or metadata_binding.get("adapter_version")
        != "sourcing-model-research-lab-adapter:v10"
        or metadata_binding.get("metadata_function")
        != "model_runner_custody_metadata"
        or metadata_binding.get("adapter_metadata_key") != "dispatch_custody"
        or metadata_binding.get("dispatch_metadata_sha256")
        != ADDITIVE_DISPATCH_CUSTODY_V3_METADATA_SHA256
        or not isinstance(metadata_binding.get("required_model_keys"), list)
        or not isinstance(metadata_binding.get("required_dispatch_keys"), list)
        or not metadata_binding.get("required_model_keys")
        or not metadata_binding.get("required_dispatch_keys")
        or not set(metadata_binding["required_model_keys"]).issubset(
            set(metadata_binding["required_dispatch_keys"])
        )
    ):
        raise ValueError(
            "semantic sourcing compatibility typed dispatch metadata policy is invalid"
        )
    return document


def _same_json_literal(actual: Any, expected: Any) -> bool:
    """Compare decoded JSON values without Python numeric type coercion."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _same_json_literal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_json_literal(left, right)
            for left, right in zip(actual, expected)
        )
    return actual == expected


def approved_typed_dispatch_custody_v3_metadata_v1() -> Dict[str, Any]:
    """Build the exact adapter v10 metadata from the reviewed v28 snapshot."""

    if (
        _snapshot_sha256(CONTRACT_V68_PATH)
        != ADDITIVE_DISPATCH_CUSTODY_V3_CONTRACT_SHA256
    ):
        raise ValueError("typed dispatch custody contract snapshot differs")
    if (
        _snapshot_sha256(PARITY_FIXTURE_V28_PATH)
        != ADDITIVE_DISPATCH_CUSTODY_V3_PARITY_SHA256
    ):
        raise ValueError("typed dispatch custody parity snapshot differs")
    violations: List[str] = []
    parity = _safe_json_document(
        PARITY_FIXTURE_V28_PATH,
        label="typed dispatch custody parity snapshot",
        violations=violations,
    )
    if violations or not isinstance(parity, dict):
        raise ValueError("typed dispatch custody parity snapshot is invalid")
    projection = parity.get("expected_model_runner_custody_v3_projection")
    if not isinstance(projection, dict):
        raise ValueError("typed dispatch custody parity projection is invalid")

    metadata_binding = semantic_compatibility_policy_v1()[
        "additive_dispatch_custody_v3"
    ]["metadata_binding"]
    required_model_keys = metadata_binding["required_model_keys"]
    required_dispatch_keys = metadata_binding["required_dispatch_keys"]
    if (
        len(required_model_keys) != len(set(required_model_keys))
        or len(required_dispatch_keys) != len(set(required_dispatch_keys))
    ):
        raise ValueError("typed dispatch custody metadata policy is invalid")
    try:
        expected = {
            key: deepcopy(projection[key]) for key in required_model_keys
        }
    except KeyError as exc:
        raise ValueError(
            "typed dispatch custody parity projection is incomplete"
        ) from exc
    expected.update(
        {
            "completion_included": False,
            "initial_dispatch_entrypoint": (
                "dispatch_runner_initial_custody_v3"
            ),
            "initial_dispatch_schema_version": (
                "model-runner-custody:v3-initial-dispatch"
            ),
            "start_entrypoint": "build_runner_start_custody_v3",
            "start_validation_entrypoint": (
                "validate_runner_start_custody_v3"
            ),
            "action_entrypoint": "build_runner_action_custody_v3",
            "action_validation_entrypoint": (
                "validate_runner_action_custody_v3"
            ),
            "continuation_entrypoint": (
                "build_runner_initial_continuation_custody_v3"
            ),
            "continuation_validation_entrypoint": (
                "validate_runner_initial_continuation_custody_v3"
            ),
        }
    )
    if set(expected) != set(required_dispatch_keys):
        raise ValueError("typed dispatch custody metadata policy is invalid")
    if _sha256_json(expected) != ADDITIVE_DISPATCH_CUSTODY_V3_METADATA_SHA256:
        raise ValueError("typed dispatch custody approved metadata differs")
    return expected


def validate_typed_dispatch_custody_v3_metadata_v1(
    value: Any,
) -> Dict[str, Any]:
    """Require exact JSON structure and canonical bytes for adapter v10."""

    expected = approved_typed_dispatch_custody_v3_metadata_v1()
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError("typed dispatch custody metadata differs")
    try:
        actual_sha256 = _sha256_json(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("typed dispatch custody metadata differs") from exc
    if (
        actual_sha256 != ADDITIVE_DISPATCH_CUSTODY_V3_METADATA_SHA256
        or not _same_json_literal(value, expected)
    ):
        raise ValueError("typed dispatch custody metadata differs")
    return deepcopy(value)


def semantic_compatibility_policy_hash_v1() -> str:
    """Return the complete consumer admission-policy identity.

    The semantic policy document and the narrow historical release exemption
    table are both authorization inputs, so receipts and cache keys bind both.
    """

    reviewed_profiles = [
        {
            "contract_id": str(snapshot["contract"]["contract_id"]),
            "contract_sha256": str(snapshot["contract_sha256"]),
            "parity_sha256": str(snapshot["parity_sha256"]),
            "release_identities": sorted(
                (dict(item) for item in snapshot["release_identities"]),
                key=lambda item: (
                    item["source_tree_hash"],
                    item["git_commit_sha"],
                    item["manifest_hash"],
                    item["image_digest"],
                ),
            ),
            "positional_exact_signatures": bool(
                snapshot["positional_exact_signatures"]
            ),
            "variadic_parameters": dict(
                sorted(dict(snapshot["variadic_parameters"]).items())
            ),
        }
        for snapshot in sorted(
            reviewed_consumer_profiles(),
            key=lambda item: (
                str(item["contract"]["contract_id"]),
                str(item["contract_sha256"]),
                str(item["parity_sha256"]),
            ),
        )
    ]
    return _sha256_json(
        {
            "schema_version": "research-lab-consumer-admission-policy.v1",
            "semantic_policy_sha256": _snapshot_sha256(
                SEMANTIC_COMPATIBILITY_POLICY_V1_PATH
            ),
            "reviewed_legacy_profiles": reviewed_profiles,
        }
    )


def semantic_compatibility_policy_identity_v1() -> tuple[Dict[str, Any], str]:
    """Load one stable, supported consumer policy identity.

    The committed policy is protected by the release manifest. Rechecking its
    hash around the parse also prevents a mutable checkout from mixing one
    policy document with another policy hash in a cache key or receipt.
    """

    hash_before = semantic_compatibility_policy_hash_v1()
    policy = semantic_compatibility_policy_v1()
    hash_after = semantic_compatibility_policy_hash_v1()
    if hash_before != hash_after:
        raise ValueError(
            "semantic sourcing compatibility policy changed during admission"
        )
    return policy, hash_after


def qualification_protocol_policy_identity_v2() -> tuple[Dict[str, Any], str]:
    """Load the canonical qualification-v2 policy identity without model code."""

    try:
        policy_before = json.loads(
            QUALIFICATION_OUTCOME_CONTRACT_V2_PATH.read_text(encoding="utf-8")
        )
        policy_after = json.loads(
            QUALIFICATION_OUTCOME_CONTRACT_V2_PATH.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            "qualification protocol compatibility policy is unavailable"
        ) from exc
    if not isinstance(policy_before, dict) or not isinstance(policy_after, dict):
        raise ValueError("qualification protocol compatibility policy is invalid")
    hash_before = _sha256_json(policy_before)
    hash_after = _sha256_json(policy_after)
    if (
        hash_before != QUALIFICATION_PROTOCOL_POLICY_SHA256_V2
        or hash_after != QUALIFICATION_PROTOCOL_POLICY_SHA256_V2
    ):
        raise ValueError(
            "qualification protocol compatibility policy changed during admission"
        )
    return policy_before, hash_before


COMPATIBILITY_ADMISSION_POLICY_PROFILE_REGISTRY = (
    (
        (
            SEMANTIC_COMPATIBILITY_RECEIPT_SCHEMA_V1,
            SEMANTIC_COMPATIBILITY_CONSUMER_API_V1,
            "legacy_exact",
            SEMANTIC_COMPATIBILITY_ACCEPTED_DECISION,
        ),
        "semantic_policy_v1",
    ),
    (
        (
            SEMANTIC_COMPATIBILITY_RECEIPT_SCHEMA_V1,
            SEMANTIC_COMPATIBILITY_CONSUMER_API_V1,
            "semantic_v1",
            SEMANTIC_COMPATIBILITY_ACCEPTED_DECISION,
        ),
        "semantic_policy_v1",
    ),
    (
        (
            QUALIFICATION_PROTOCOL_COMPATIBILITY_RECEIPT_SCHEMA_V2,
            QUALIFICATION_PROTOCOL_CONSUMER_API_V2,
            QUALIFICATION_PROTOCOL_ADMISSION_MODE_V2,
            SEMANTIC_COMPATIBILITY_ACCEPTED_DECISION,
        ),
        "qualification_policy_v2",
    ),
)


def compatibility_admission_policy_identity(
    receipt: Mapping[str, Any],
) -> tuple[Dict[str, Any], str]:
    """Resolve and revalidate the policy for one admitted receipt profile."""

    profile_key = (
        receipt.get("schema_version"),
        receipt.get("consumer_api_version"),
        receipt.get("admission_mode"),
        receipt.get("decision"),
    )
    profile = dict(COMPATIBILITY_ADMISSION_POLICY_PROFILE_REGISTRY).get(
        profile_key
    )
    if profile == "semantic_policy_v1":
        policy, policy_hash = semantic_compatibility_policy_identity_v1()
    elif profile == "qualification_policy_v2":
        policy, policy_hash = qualification_protocol_policy_identity_v2()
    else:
        raise ValueError("compatibility admission policy profile is unsupported")
    if receipt.get("policy_hash") != policy_hash:
        raise ValueError("compatibility admission policy differs from its profile")
    return policy, policy_hash


def _safe_json_document(path: Path, *, label: str, violations: List[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        violations.append(
            f"{label} is unreadable: {type(exc).__name__}"
        )
        return None


def _semantic_tree(
    root: Path,
    relative: str,
    *,
    parsed: Dict[str, ast.Module],
    violations: List[str],
) -> ast.Module | None:
    if relative in parsed:
        return parsed[relative]
    path = root / relative
    if not path.is_file() or path.is_symlink():
        violations.append(f"missing semantic compatibility file: {relative}")
        return None
    try:
        tree = ast.parse(path.read_bytes())
    except SyntaxError as exc:
        violations.append(
            f"unparseable semantic compatibility module {relative}: "
            f"{exc.msg} (line {exc.lineno})"
        )
        return None
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        violations.append(
            f"unreadable semantic compatibility module {relative}: "
            f"{type(exc).__name__}"
        )
        return None
    parsed[relative] = tree
    return tree


def _manifest_document(manifest: Any) -> dict[str, Any]:
    if manifest is None:
        return {}
    if isinstance(manifest, Mapping):
        return dict(manifest)
    to_dict = getattr(manifest, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    return {}


def _manifest_pair_violations(
    *,
    manifest: Mapping[str, Any],
    contract_id: str,
    contract_path: str,
    contract_hash: str,
    parity_path: str,
    parity_hash: str,
    source_tree_hash: str,
) -> List[str]:
    if not manifest:
        return []
    violations: List[str] = []
    contract = manifest.get("compatibility_contract")
    fixtures = manifest.get("consumer_parity_fixtures")
    if not isinstance(contract, Mapping) or dict(contract) != {
        "contract_id": contract_id,
        "path": contract_path,
        "sha256": contract_hash,
    }:
        violations.append(
            "signed manifest compatibility contract differs from source"
        )
    if not isinstance(fixtures, Mapping) or dict(fixtures) != {
        "path": parity_path,
        "sha256": parity_hash,
    }:
        violations.append("signed manifest parity fixtures differ from source")
    if source_tree_hash and str(manifest.get("model_artifact_hash") or "") != source_tree_hash:
        violations.append("signed manifest source tree hash differs")
    if not re.fullmatch(
        r"sha256:[0-9a-f]{64}", str(manifest.get("manifest_hash") or "")
    ):
        violations.append("signed manifest hash is invalid")
    if not re.fullmatch(
        r"[^\s@]+@sha256:[0-9a-f]{64}",
        str(manifest.get("image_digest") or ""),
    ):
        violations.append("signed manifest image digest is invalid")
    return violations


def _semantic_compatibility_receipt(
    *,
    mode: str,
    consumer_api_version: str,
    policy_hash: str,
    source_tree_hash: str,
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
    contract_hash: str,
    parity_hash: str,
    bindings: Mapping[str, str],
) -> Dict[str, Any]:
    body = {
        "schema_version": SEMANTIC_COMPATIBILITY_RECEIPT_SCHEMA_V1,
        "consumer_api_version": consumer_api_version,
        "decision": SEMANTIC_COMPATIBILITY_ACCEPTED_DECISION,
        "admission_mode": mode,
        "policy_hash": policy_hash,
        "source_tree_hash": source_tree_hash,
        "manifest_hash": str(manifest.get("manifest_hash") or ""),
        "image_digest": str(manifest.get("image_digest") or ""),
        "contract_id": str(contract.get("contract_id") or ""),
        "contract_schema_major": contract.get("schema_version"),
        "contract_hash": contract_hash,
        "parity_hash": parity_hash,
        "bindings": dict(sorted(bindings.items())),
    }
    return {**body, "receipt_hash": _sha256_json(body)}


def validate_source_tree_compatibility_receipt_v1(
    receipt: Mapping[str, Any],
    *,
    manifest: Any,
    source_tree_hash: str,
    policy: Mapping[str, Any] | None = None,
    policy_hash: str = "",
) -> Dict[str, Any]:
    """Validate one cached or transported receipt against consumer authority."""

    normalized = dict(receipt)
    manifest_document = _manifest_document(manifest)
    if policy is None:
        resolved_policy, resolved_policy_hash = (
            semantic_compatibility_policy_identity_v1()
        )
    else:
        resolved_policy = dict(policy)
        resolved_policy_hash = str(policy_hash or "")
    body = {
        key: value
        for key, value in normalized.items()
        if key != "receipt_hash"
    }
    violations = _manifest_pair_violations(
        manifest=manifest_document,
        contract_id=str(normalized.get("contract_id") or ""),
        contract_path=str(resolved_policy["canonical_contract_path"]),
        contract_hash=str(normalized.get("contract_hash") or ""),
        parity_path=str(resolved_policy["canonical_parity_path"]),
        parity_hash=str(normalized.get("parity_hash") or ""),
        source_tree_hash=str(source_tree_hash or ""),
    )
    legacy_identity_valid = True
    if normalized.get("admission_mode") == "legacy_exact":
        try:
            validate_reviewed_legacy_release_manifest_identity_v1(
                normalized,
                manifest_document,
            )
        except ValueError:
            legacy_identity_valid = False
    profiled_legacy_source = any(
        str(snapshot["contract_sha256"])
        == str(normalized.get("contract_hash") or "")
        and str(snapshot["parity_sha256"])
        == str(normalized.get("parity_hash") or "")
        and bool(
            _reviewed_legacy_release_identities(
                snapshot,
                source_tree_hash=str(normalized.get("source_tree_hash") or ""),
            )
        )
        for snapshot in reviewed_consumer_profiles()
    )
    mode_identity_valid = normalized.get("admission_mode") == (
        "legacy_exact" if profiled_legacy_source else "semantic_v1"
    )
    if (
        set(normalized) != SEMANTIC_COMPATIBILITY_RECEIPT_FIELDS_V1
        or normalized.get("schema_version")
        != SEMANTIC_COMPATIBILITY_RECEIPT_SCHEMA_V1
        or normalized.get("consumer_api_version")
        != resolved_policy.get("consumer_api_version")
        or normalized.get("decision")
        != SEMANTIC_COMPATIBILITY_ACCEPTED_DECISION
        or normalized.get("admission_mode") not in {"legacy_exact", "semantic_v1"}
        or normalized.get("policy_hash") != resolved_policy_hash
        or normalized.get("source_tree_hash") != source_tree_hash
        or normalized.get("manifest_hash")
        != str(manifest_document.get("manifest_hash") or "")
        or normalized.get("image_digest")
        != str(manifest_document.get("image_digest") or "")
        or normalized.get("contract_schema_major")
        != resolved_policy.get("contract_schema_major")
        or not isinstance(normalized.get("bindings"), Mapping)
        or normalized.get("receipt_hash") != _sha256_json(body)
        or not legacy_identity_valid
        or not mode_identity_valid
        or violations
    ):
        raise ValueError("compatibility receipt differs from signed artifact")
    return normalized


def _semantic_contract_shape_violations_v1(
    contract: Mapping[str, Any],
) -> list[str]:
    """Validate the supported producer-document schema, never its authority."""

    violations: list[str] = []
    required_files = contract.get("required_files")
    if not isinstance(required_files, list) or not all(
        isinstance(item, str) for item in required_files
    ):
        violations.append("model compatibility required files are invalid")

    functions = contract.get("functions")
    functions_valid = isinstance(functions, Mapping)
    if functions_valid:
        functions_valid = all(
            isinstance(relative, str)
            and isinstance(declarations, Mapping)
            and all(
                isinstance(name, str)
                and isinstance(parameters, list)
                and all(isinstance(parameter, str) for parameter in parameters)
                for name, parameters in declarations.items()
            )
            for relative, declarations in functions.items()
        )
    if not functions_valid:
        violations.append("model compatibility functions declaration is invalid")

    for key, label in (
        ("full_parameters", "full parameter declaration"),
        ("required_keyword_only", "required keyword-only declaration"),
    ):
        declarations = contract.get(key)
        if not isinstance(declarations, Mapping) or not all(
            isinstance(name, str)
            and isinstance(parameters, list)
            and all(isinstance(parameter, str) for parameter in parameters)
            for name, parameters in declarations.items()
        ):
            violations.append(f"model compatibility {label} is invalid")

    frozen_asyncness = contract.get("frozen_asyncness")
    if not isinstance(frozen_asyncness, Mapping) or not all(
        isinstance(name, str) and isinstance(value, bool)
        for name, value in frozen_asyncness.items()
    ):
        violations.append(
            "model compatibility frozen asyncness declaration is invalid"
        )
    return violations


def _typed_dispatch_custody_v3_requested(
    contract: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    root: Path | None = None,
) -> bool:
    """Detect a custody-v3 claim before exact identity verification."""

    dispatch_v3 = policy["additive_dispatch_custody_v3"]
    declared_exact_signatures = contract.get("exact_signatures")
    exact_signatures = (
        set(declared_exact_signatures)
        if isinstance(declared_exact_signatures, list)
        and all(isinstance(item, str) for item in declared_exact_signatures)
        else set()
    )
    exact_markers = {
        "research_lab_adapter.py:dispatch_runner_initial_custody_v3",
        "sourcing_model/model_runner.py:model_runner_custody_metadata",
    }
    marker_claim = exact_markers.issubset(exact_signatures)
    contract_constants = contract.get("exact_constants")
    adapter_constants = (
        contract_constants.get("research_lab_adapter.py")
        if isinstance(contract_constants, Mapping)
        else None
    )
    expected_adapter_version = str(
        dispatch_v3["metadata_binding"]["adapter_version"]
    )
    contract_version_claim = (
        isinstance(adapter_constants, Mapping)
        and adapter_constants.get("ADAPTER_VERSION") == expected_adapter_version
    )
    source_version_claim = False
    if root is not None:
        try:
            adapter_tree = ast.parse((Path(root) / "research_lab_adapter.py").read_bytes())
            source_version_claim = (
                _unique_literal_binding_v1(adapter_tree, "ADAPTER_VERSION")
                == expected_adapter_version
            )
        except (OSError, SyntaxError, ValueError, UnicodeDecodeError):
            source_version_claim = False
    return (
        str(contract.get("contract_id") or "")
        == str(dispatch_v3["contract_id"])
        or marker_claim
        or contract_version_claim
        or source_version_claim
    )


def _merge_typed_dispatch_policy(
    policy: Mapping[str, Any],
) -> Dict[str, Any]:
    """Overlay v3 requirements while preserving its reviewed v2 runner ABI."""

    merged = deepcopy(dict(policy))
    dispatch_v3 = dict(policy["additive_dispatch_custody_v3"])
    for key in ("callables", "exact_constants", "required_imports"):
        base = deepcopy(dict(merged.get(key) or {}))
        for relative, values in (dispatch_v3.get(key) or {}).items():
            if key == "required_imports":
                base.setdefault(relative, [])
                base[relative] = [*base[relative], *deepcopy(list(values))]
            elif key == "callables":
                existing = dict(base.get(relative) or {})
                existing.update(deepcopy(dict(values)))
                base[relative] = existing
            else:
                base[relative] = deepcopy(dict(values))
        merged[key] = base
    integer_minimums = deepcopy(dict(merged.get("integer_minimums") or {}))
    for relative, values in (dispatch_v3.get("integer_minimums") or {}).items():
        integer_minimums[relative] = deepcopy(dict(values))
    merged["integer_minimums"] = integer_minimums
    slices = deepcopy(dict(merged.get("critical_binding_slices") or {}))
    for relative, values in (dispatch_v3.get("critical_binding_slices") or {}).items():
        slices[relative] = deepcopy(dict(values))
    merged["critical_binding_slices"] = slices
    if "import_time_binding_slices" in dispatch_v3:
        merged["import_time_binding_slices"] = deepcopy(
            dict(dispatch_v3.get("import_time_binding_slices") or {})
        )
    opaque = deepcopy(dict(merged.get("opaque_constants") or {}))
    for relative, values in (dispatch_v3.get("opaque_constants") or {}).items():
        current = dict(opaque.get(relative) or {})
        current.update(dict(values))
        opaque[relative] = current
    merged["opaque_constants"] = opaque
    required_files = list(merged.get("required_files") or ())
    for relative in dispatch_v3.get("required_files") or ():
        if relative not in required_files:
            required_files.append(relative)
    merged["required_files"] = required_files
    return merged


def _typed_dispatch_metadata_violations(
    root: Path,
    *,
    policy: Mapping[str, Any],
    parsed: Dict[str, ast.Module],
) -> List[str]:
    """Check the adapter/model metadata join without importing model code."""

    dispatch_v3 = policy["additive_dispatch_custody_v3"]
    requirements = dispatch_v3.get("metadata_binding") or {}
    adapter_relative = "research_lab_adapter.py"
    path = root / adapter_relative
    try:
        tree = parsed.get(adapter_relative)
        if tree is None:
            tree = ast.parse(path.read_bytes())
            parsed[adapter_relative] = tree
    except (OSError, SyntaxError, ValueError, UnicodeDecodeError):
        return ["typed dispatch adapter metadata is unreadable"]
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    metadata = functions.get("adapter_metadata")
    if metadata is None or isinstance(metadata, ast.AsyncFunctionDef):
        return ["typed dispatch adapter metadata callable is invalid"]
    violations: List[str] = []
    expected_version = str(requirements.get("adapter_version") or "")
    try:
        adapter_version = _unique_literal_binding_v1(
            tree, "ADAPTER_VERSION"
        )
    except ValueError:
        adapter_version = None
    if adapter_version != expected_version:
        violations.append("typed dispatch adapter version is not v10")
    metadata_name = str(requirements.get("metadata_function") or "")
    if not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == metadata_name
        for node in ast.walk(metadata)
    ):
        violations.append("typed dispatch metadata is not sourced from the model")
    key_name = str(requirements.get("adapter_metadata_key") or "")
    if not any(
        isinstance(node, ast.Dict)
        and any(
            isinstance(key, ast.Constant) and key.value == key_name
            for key in node.keys
        )
        for node in ast.walk(metadata)
    ):
        violations.append("typed dispatch metadata binding key is missing")
    dispatch_dicts = []
    for node in ast.walk(metadata):
        if not isinstance(node, ast.Dict):
            continue
        keys = {
            key.value
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if key_name in keys:
            dispatch_index = next(
                (
                    index
                    for index, key in enumerate(node.keys)
                    if isinstance(key, ast.Constant) and key.value == key_name
                ),
                None,
            )
            if dispatch_index is not None and isinstance(
                node.values[dispatch_index], ast.Dict
            ):
                dispatch_dicts.append(node.values[dispatch_index])
    required_dispatch_keys = set(
        str(item) for item in requirements.get("required_dispatch_keys") or ()
    )
    observed_dispatch_keys = {
        key.value
        for dispatch in dispatch_dicts
        for key in dispatch.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    if any(
        any(
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == metadata_name
            for key, value in zip(dispatch.keys, dispatch.values)
            if key is None
        )
        for dispatch in dispatch_dicts
    ):
        observed_dispatch_keys.update(
            str(item) for item in requirements.get("required_model_keys") or ()
        )
    if not dispatch_dicts or not required_dispatch_keys.issubset(
        observed_dispatch_keys
    ):
        violations.append("typed dispatch adapter metadata fields are incomplete")

    model_relative = "sourcing_model/model_runner.py"
    model_path = root / model_relative
    try:
        model_tree = parsed.get(model_relative)
        if model_tree is None:
            model_tree = ast.parse(model_path.read_bytes())
            parsed[model_relative] = model_tree
    except (OSError, SyntaxError, ValueError, UnicodeDecodeError):
        return [*violations, "typed dispatch model metadata is unreadable"]
    model_function = next(
        (
            node
            for node in model_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == metadata_name
        ),
        None,
    )
    model_return = next(
        (
            node.value
            for node in ast.walk(model_function)
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
        ),
        None,
    ) if model_function is not None else None
    observed_model_keys = {
        key.value
        for key in (model_return.keys if isinstance(model_return, ast.Dict) else ())
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    required_model_keys = set(
        str(item) for item in requirements.get("required_model_keys") or ()
    )
    if not required_model_keys.issubset(observed_model_keys):
        violations.append("typed dispatch model metadata fields are incomplete")
    return violations


def verify_semantic_source_tree_compatibility_v1(
    root: Path,
    *,
    manifest: Any = None,
    source_tree_hash: str = "",
    _policy: Mapping[str, Any] | None = None,
    _policy_hash: str = "",
) -> tuple[List[str], Dict[str, Any] | None]:
    """Verify an unseen model contract by consumer-owned characteristics.

    The model's contract and corpus are checked for signed/source
    self-consistency, but neither one defines what Leadpoet accepts. The
    measured policy below independently inspects the source ABI, capability
    boundary, metadata bindings, receipt/output identities, and safety floors.
    """

    root = Path(root)
    if _policy is None:
        policy, policy_hash = semantic_compatibility_policy_identity_v1()
    else:
        policy = dict(_policy)
        policy_hash = str(_policy_hash or "")
        if not policy_hash:
            raise ValueError(
                "semantic sourcing compatibility policy identity is unavailable"
            )
    violations: List[str] = []
    canonical_path = str(policy["canonical_contract_path"])
    parity_path = str(policy["canonical_parity_path"])
    contract_file = root / canonical_path
    parity_file = root / parity_path
    contract = _safe_json_document(
        contract_file,
        label="model-owned compatibility contract",
        violations=violations,
    )
    parity = _safe_json_document(
        parity_file,
        label="model-owned parity fixtures",
        violations=violations,
    )
    if not isinstance(contract, dict):
        contract = {}
    if not isinstance(parity, dict):
        parity = {}

    schema_major = contract.get("schema_version")
    if (
        isinstance(schema_major, bool)
        or schema_major != int(policy["contract_schema_major"])
    ):
        violations.append(
            "unsupported model compatibility contract schema major"
        )
    declared_consumer_api = contract.get("consumer_api_version")
    if declared_consumer_api is not None and (
        not isinstance(declared_consumer_api, str)
        or declared_consumer_api != policy["consumer_api_version"]
    ):
        violations.append("unsupported model consumer API version")
    if contract.get("canonical_path") != canonical_path:
        violations.append("model compatibility canonical path differs")
    if contract.get("parity_fixture_path") != parity_path:
        violations.append("model compatibility parity path differs")
    contract_id = str(contract.get("contract_id") or "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", contract_id):
        violations.append("model compatibility contract id is invalid")
    if parity.get("schema_version") != 1 or isinstance(
        parity.get("schema_version"), bool
    ):
        violations.append("unsupported model parity fixture schema major")
    violations.extend(_semantic_contract_shape_violations_v1(contract))

    contract_hash = _snapshot_sha256(contract_file) if contract_file.is_file() else ""
    parity_hash = _snapshot_sha256(parity_file) if parity_file.is_file() else ""
    typed_dispatch_requested = _typed_dispatch_custody_v3_requested(
        contract,
        policy=policy,
        root=root,
    )
    if typed_dispatch_requested:
        dispatch_v3 = policy["additive_dispatch_custody_v3"]
        exact_signatures = contract.get("exact_signatures")
        exact_signatures_valid = isinstance(exact_signatures, list) and all(
            isinstance(item, str) for item in exact_signatures
        )
        if not exact_signatures_valid:
            violations.append(
                "model compatibility exact signatures declaration is invalid"
            )
        if contract_id != str(dispatch_v3["contract_id"]):
            violations.append("typed dispatch contract identity is not approved")
        if contract_hash != str(dispatch_v3["contract_sha256"]):
            violations.append("typed dispatch contract snapshot differs")
        if parity_hash != str(dispatch_v3["parity_sha256"]):
            violations.append("typed dispatch parity snapshot differs")
        reviewed_snapshots_match = True
        for snapshot_path, observed_path, label in (
            (
                dispatch_v3["contract_snapshot_path"],
                contract_file,
                "typed dispatch contract",
            ),
            (
                dispatch_v3["parity_snapshot_path"],
                parity_file,
                "typed dispatch parity",
            ),
        ):
            expected_path = Path(__file__).with_name(Path(snapshot_path).name)
            try:
                if (
                    not expected_path.is_file()
                    or not observed_path.is_file()
                    or expected_path.read_bytes() != observed_path.read_bytes()
                ):
                    reviewed_snapshots_match = False
                    violations.append(f"{label} does not match reviewed snapshot")
            except OSError:
                reviewed_snapshots_match = False
                violations.append(f"{label} snapshot is unreadable")
        typed_dispatch_verified = (
            exact_signatures_valid
            and contract_id == str(dispatch_v3["contract_id"])
            and contract_hash == str(dispatch_v3["contract_sha256"])
            and parity_hash == str(dispatch_v3["parity_sha256"])
            and reviewed_snapshots_match
        )
        if typed_dispatch_verified:
            policy = _merge_typed_dispatch_policy(policy)

    for relative in (canonical_path, parity_path):
        required_path = root / relative
        if not required_path.is_file() or required_path.is_symlink():
            violations.append(
                f"missing or non-regular semantic compatibility file: {relative}"
            )

    policy_callables = dict(policy.get("callables") or {})
    surface_document = {
        "required_files": list(policy.get("required_files") or ()),
        "functions": {
            relative: {
                name: list(spec.get("positional") or ())
                for name, spec in functions.items()
            }
            for relative, functions in policy_callables.items()
        },
        "exact_signatures": [
            f"{relative}:{name}"
            for relative, functions in policy_callables.items()
            for name in functions
        ],
        "required_positional": {
            f"{relative}:{name}": int(spec.get("required_positional") or 0)
            for relative, functions in policy_callables.items()
            for name, spec in functions.items()
        },
        "required_keyword_only": {
            f"{relative}:{name}": list(
                spec.get("required_keyword_only") or ()
            )
            for relative, functions in policy_callables.items()
            for name, spec in functions.items()
        },
        "frozen_asyncness": {
            f"{relative}:{name}": bool(spec.get("is_async"))
            for relative, functions in policy_callables.items()
            for name, spec in functions.items()
        },
        "integer_minimums": dict(policy.get("integer_minimums") or {}),
        "exact_constants": dict(policy.get("exact_constants") or {}),
    }
    surface_document["full_parameters"] = {
        f"{relative}:{name}": list(spec.get("full_parameters") or ())
        for relative, functions in policy_callables.items()
        for name, spec in functions.items()
        if spec.get("full_parameters") is not None
    }
    violations.extend(
        _verify_source_tree_contract_document(
            root,
            document=surface_document,
            reviewed_snapshot={
                "contract_path": contract_file,
                "parity_path": parity_file,
                "positional_exact_signatures": True,
                "variadic_parameters": {},
            },
            verify_snapshot_pair=False,
        )
    )
    parsed: Dict[str, ast.Module] = {}

    for relative, requirements in (policy.get("required_imports") or {}).items():
        tree = _semantic_tree(
            root,
            str(relative),
            parsed=parsed,
            violations=violations,
        )
        if tree is None:
            continue
        for requirement in requirements:
            binding = str(requirement.get("binding") or "")
            if not _required_import_binding_valid_v1(tree, requirement):
                violations.append(
                    f"hard import binding drift {relative}:{binding}"
                )

    observed_release_labels: dict[tuple[str, str], str] = {}
    for relative, constants in (policy.get("opaque_constants") or {}).items():
        tree = _semantic_tree(
            root,
            str(relative),
            parsed=parsed,
            violations=violations,
        )
        if tree is None:
            continue
        for name, pattern in constants.items():
            try:
                value = _unique_literal_binding_v1(tree, str(name))
            except ValueError:
                value = None
            if not isinstance(value, str) or re.fullmatch(str(pattern), value) is None:
                violations.append(f"hard release label drift {relative}:{name}")
            else:
                observed_release_labels[(str(relative), str(name))] = value

    for relative, specification in (
        policy.get("critical_binding_slices") or {}
    ).items():
        tree = _semantic_tree(
            root,
            str(relative),
            parsed=parsed,
            violations=violations,
        )
        if tree is None:
            continue
        observed_hash, slice_violations = _critical_binding_slice_v1(
            tree,
            roots=[str(item) for item in specification.get("roots") or ()],
            normalized_literals=set(
                (policy.get("opaque_constants") or {}).get(relative) or {}
            ),
        )
        violations.extend(
            f"hard critical binding drift {relative}:{item}"
            for item in slice_violations
        )
        if observed_hash != str(specification.get("sha256") or ""):
            violations.append(f"hard module semantic drift {relative}")

    if typed_dispatch_requested:
        violations.extend(
            _typed_dispatch_metadata_violations(
                root,
                policy=policy,
                parsed=parsed,
            )
        )

    for relative, expected_hash in (
        policy.get("import_time_binding_slices") or {}
    ).items():
        tree = _semantic_tree(
            root,
            str(relative),
            parsed=parsed,
            violations=violations,
        )
        if tree is None:
            continue
        observed_hash, slice_violations = _critical_binding_slice_v1(
            tree,
            roots=[str(name) for name in policy_callables.get(relative) or ()],
            normalized_literals=set(
                (policy.get("opaque_constants") or {}).get(relative) or {}
            ),
            strip_function_bodies=True,
        )
        violations.extend(
            f"hard import-time binding drift {relative}:{item}"
            for item in slice_violations
        )
        if observed_hash != str(expected_hash or ""):
            violations.append(f"hard import-time semantic drift {relative}")

    manifest_document = _manifest_document(manifest)
    violations.extend(
        _manifest_pair_violations(
            manifest=manifest_document,
            contract_id=contract_id,
            contract_path=canonical_path,
            contract_hash=contract_hash,
            parity_path=parity_path,
            parity_hash=parity_hash,
            source_tree_hash=str(source_tree_hash or ""),
        )
    )
    if violations:
        return violations, None
    exact_constants = dict(policy.get("exact_constants") or {})
    adapter_constants = dict(
        exact_constants.get("research_lab_adapter.py") or {}
    )
    bindings = {
        "adapter_version": str(
            observed_release_labels[("research_lab_adapter.py", "ADAPTER_VERSION")]
        ),
        "capability_contract_version": str(
            dict(
                exact_constants.get("sourcing_model/runtime_capabilities.py")
                or {}
            )["CAPABILITY_CONTRACT_VERSION"]
        ),
        "component_registry_version": str(
            adapter_constants["COMPONENT_REGISTRY_VERSION"]
        ),
        "routing_compiler_version": str(
            observed_release_labels[
                ("sourcing_model/routing/compiler.py", "COMPILER_VERSION")
            ]
        ),
        "scoring_adapter_version": str(
            adapter_constants["SCORING_ADAPTER_VERSION"]
        ),
    }
    return [], _semantic_compatibility_receipt(
        mode="semantic_v1",
        consumer_api_version=str(policy["consumer_api_version"]),
        policy_hash=policy_hash,
        source_tree_hash=str(source_tree_hash or ""),
        manifest=manifest_document,
        contract=contract,
        contract_hash=contract_hash,
        parity_hash=parity_hash,
        bindings=bindings,
    )


def _verify_source_tree_contract_document(
    root: Path,
    *,
    document: Mapping[str, Any],
    reviewed_snapshot: Mapping[str, Any],
    verify_snapshot_pair: bool,
) -> List[str]:
    root = Path(root)
    violations: List[str] = []
    document = dict(document)
    reviewed_snapshot = dict(reviewed_snapshot)
    expected_contract_path = Path(reviewed_snapshot["contract_path"])
    expected_parity_path = Path(reviewed_snapshot["parity_path"])
    frozen_asyncness = dict(document.get("frozen_asyncness") or {})
    exact_signatures = set(document.get("exact_signatures") or ())
    frozen_required_keyword_only = dict(
        document.get("required_keyword_only") or {}
    )
    frozen_required_positional = dict(
        document.get("required_positional") or {}
    )
    positional_exact_signatures = bool(
        reviewed_snapshot.get("positional_exact_signatures", False)
    )
    reviewed_variadic_parameters = dict(
        reviewed_snapshot.get("variadic_parameters") or {}
    )

    canonical_relative = str(document.get("canonical_path") or "")
    canonical_path = root / canonical_relative
    if verify_snapshot_pair and canonical_path.is_file():
        try:
            if canonical_path.read_bytes() != expected_contract_path.read_bytes():
                violations.append(
                    "model-owned compatibility contract differs from the "
                    "reviewed Lab snapshot"
                )
        except OSError as exc:
            violations.append(
                "unable to compare model-owned compatibility contract: "
                f"{type(exc).__name__}"
            )
    parity_relative = str(document.get("parity_fixture_path") or "")
    parity_path = root / parity_relative
    if verify_snapshot_pair and parity_path.is_file():
        try:
            if parity_path.read_bytes() != expected_parity_path.read_bytes():
                violations.append(
                    "model-owned parity fixtures differ from the reviewed "
                    "Lab snapshot"
                )
        except OSError as exc:
            violations.append(
                "unable to compare model-owned parity fixtures: "
                f"{type(exc).__name__}"
            )

    for relative in document.get("required_files", []):
        if not (root / relative).is_file():
            violations.append(f"missing required file: {relative}")

    parsed: Dict[str, ast.Module] = {}

    def _tree(relative: str) -> ast.Module | None:
        if relative in parsed:
            return parsed[relative]
        path = root / relative
        if not path.is_file():
            return None
        try:
            # Parse raw bytes so PEP 263 coding declarations are honored the
            # same way the interpreter honors them — a legal non-UTF-8 module
            # must parse, and an unreadable one must be a VIOLATION, never an
            # exception that lets the build gate fail open.
            tree = ast.parse(path.read_bytes())
        except SyntaxError as exc:
            violations.append(f"unparseable module {relative}: {exc.msg} (line {exc.lineno})")
            return None
        except (ValueError, UnicodeDecodeError, OSError) as exc:
            violations.append(
                f"unreadable module {relative}: {type(exc).__name__}: {str(exc)[:120]}"
            )
            return None
        parsed[relative] = tree
        return tree

    for relative, functions in (document.get("functions") or {}).items():
        tree = _tree(relative)
        if tree is None:
            continue
        symbols = _module_symbols(tree)
        for name, expected_params in functions.items():
            actual = symbols["functions"].get(name)
            if actual is None:
                violations.append(f"missing function {relative}:{name}")
                continue
            expected = list(expected_params)
            actual_params = actual["params"]
            contract_key = f"{relative}:{name}"
            expected_full = (document.get("full_parameters") or {}).get(
                contract_key
            )
            if (
                expected_full is not None
                and (
                    actual["all_params"] != list(expected_full)
                    or actual["positional_only"]
                    or actual["vararg"] is not None
                    or actual["kwarg"] is not None
                )
            ):
                violations.append(
                    f"full parameter drift {relative}:{name}: expected "
                    f"{list(expected_full)}, found {actual['all_params']} "
                    f"(positional_only={actual['positional_only']}, "
                    f"vararg={actual['vararg']!r}, kwarg={actual['kwarg']!r})"
                )
            # Newer contracts separate the exact positional surface in
            # ``functions`` from the complete keyword-only surface in
            # ``full_parameters``. Older snapshots retain the original
            # all-parameter exactness.
            exact_actual = (
                actual["params"]
                if expected_full is not None or positional_exact_signatures
                else actual["all_params"]
            )
            expected_variadic = reviewed_variadic_parameters.get(
                contract_key,
                {"vararg": None, "kwarg": None},
            )
            if contract_key in exact_signatures and (
                exact_actual != expected
                or actual["positional_only"]
                or actual["vararg"] != expected_variadic["vararg"]
                or actual["kwarg"] != expected_variadic["kwarg"]
            ):
                violations.append(
                    f"exact parameter drift {relative}:{name}: expected "
                    f"{expected}, found {exact_actual} "
                    f"(positional_only={actual['positional_only']}, "
                    f"vararg={actual['vararg']!r}, kwarg={actual['kwarg']!r})"
                )
            elif expected and actual_params[: len(expected)] != expected:
                violations.append(
                    f"parameter drift {relative}:{name}: expected leading "
                    f"parameters {expected}, found {actual_params}"
                )
            elif actual["required_positional"] > len(expected):
                violations.append(
                    f"required parameter drift {relative}:{name}: expected at most "
                    f"{len(expected)} required positional parameters, found "
                    f"{actual['required_positional']}"
                )
            expected_required_positional = frozen_required_positional.get(
                contract_key
            )
            if (
                expected_required_positional is not None
                and actual["required_positional"]
                != int(expected_required_positional)
            ):
                violations.append(
                    f"required parameter drift {relative}:{name}: expected "
                    f"{expected_required_positional}, found "
                    f"{actual['required_positional']}"
                )
            expected_required_keyword_only = frozen_required_keyword_only.get(
                contract_key, []
            )
            if actual["required_keyword_only"] != expected_required_keyword_only:
                violations.append(
                    f"required keyword-only parameter drift {relative}:{name}: "
                    f"expected {expected_required_keyword_only}, found "
                    f"{actual['required_keyword_only']}"
                )
            frozen_async = frozen_asyncness.get(contract_key)
            if frozen_async is not None and actual["is_async"] != frozen_async:
                violations.append(
                    f"asyncness drift {relative}:{name}: frozen surface is "
                    f"{'async' if frozen_async else 'sync'}, found "
                    f"{'async' if actual['is_async'] else 'sync'}"
                )

    for relative, required_modules in (
        document.get("required_imports") or {}
    ).items():
        tree = _tree(relative)
        if tree is None:
            continue
        bound_imports = _module_bound_imports(tree)
        for module_name in required_modules:
            if str(module_name) not in bound_imports:
                violations.append(
                    f"missing bound import {relative}:{module_name}"
                )

    for relative, minimums in (document.get("integer_minimums") or {}).items():
        tree = _tree(relative)
        if tree is None:
            continue
        constants = _module_symbols(tree)["constants"]
        for name, floor in minimums.items():
            value = constants.get(name)
            if value is None:
                violations.append(f"missing integer constant {relative}:{name}")
            elif value < int(floor):
                violations.append(
                    f"integer floor breach {relative}:{name}: {value} < {floor}"
                )

    for relative, expected_values in (
        document.get("exact_constants") or {}
    ).items():
        tree = _tree(relative)
        if tree is None:
            continue
        constants = _literal_module_constants(
            tree,
            names={str(name) for name in expected_values},
        )
        missing = object()
        for name, expected in expected_values.items():
            actual = constants.get(name, missing)
            if actual is missing or not _same_literal(actual, expected):
                violations.append(
                    f"exact constant drift {relative}:{name}: expected "
                    f"{expected!r}, found "
                    f"{None if actual is missing else actual!r}"
                )

    for relative, expected_values in (
        reviewed_snapshot.get("required_source_constants") or {}
    ).items():
        tree = _tree(relative)
        if tree is None:
            continue
        constants = _literal_module_constants(
            tree,
            names={str(name) for name in expected_values},
        )
        missing = object()
        for name, expected in expected_values.items():
            actual = constants.get(name, missing)
            if actual is missing or not _same_literal(actual, expected):
                violations.append(
                    f"reviewed source constant drift {relative}:{name}: expected "
                    f"{expected!r}, found "
                    f"{None if actual is missing else actual!r}"
                )

    return violations


def verify_source_tree_contract(root: Path) -> List[str]:
    """Return document/ABI violations for an exact pair or semantic tree.

    This diagnostic intentionally verifies the byte-exact legacy document
    surface without conferring legacy admission. Production admission also
    requires the reviewed signed source identity below.
    """

    root = Path(root)
    snapshot = _resolve_reviewed_consumer_contract_pair(root)
    if snapshot is None:
        violations, _receipt = verify_semantic_source_tree_compatibility_v1(root)
        return violations
    return _verify_source_tree_contract_document(
        root,
        document=snapshot["contract"],
        reviewed_snapshot=snapshot,
        verify_snapshot_pair=True,
    )


def _qualification_protocol_entrypoint_declared_v2(root: Path) -> bool:
    """Detect any static v2 declaration so invalid v2 cannot fall back.

    A candidate must not evade v2 admission by binding the entrypoint through
    an alias/assignment that the deliberately narrow ABI checker does not
    understand.  Likewise, advertising the protocol metadata opts the whole
    artifact into v2 even when the callable is missing or dynamically built.
    These forms are routed to the v2 checker and rejected there, never retried
    under the legacy semantic-v1 profile.
    """

    path = Path(root) / "research_lab_adapter.py"
    try:
        tree = ast.parse(path.read_bytes())
    except (OSError, SyntaxError, ValueError, UnicodeDecodeError):
        return False
    def _target_binds_name(target: ast.AST, name: str) -> bool:
        if isinstance(target, ast.Name):
            return target.id == name
        if isinstance(target, (ast.Tuple, ast.List)):
            return any(_target_binds_name(item, name) for item in target.elts)
        return False

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == QUALIFICATION_PROTOCOL_ENTRYPOINT_V2:
                return True
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                _target_binds_name(target, QUALIFICATION_PROTOCOL_ENTRYPOINT_V2)
                for target in targets
            ):
                return True
        elif isinstance(node, ast.Import):
            if any(
                (alias.asname or alias.name.split(".", 1)[0])
                == QUALIFICATION_PROTOCOL_ENTRYPOINT_V2
                for alias in node.names
            ):
                return True
        elif isinstance(node, ast.ImportFrom):
            if any(
                (alias.asname or alias.name) == QUALIFICATION_PROTOCOL_ENTRYPOINT_V2
                for alias in node.names
            ):
                return True

    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and any(
            isinstance(key, ast.Constant)
            and key.value == "qualification_outcome_protocol"
            for key in node.keys
        ):
            return True
        if (
            isinstance(node, ast.keyword)
            and node.arg == "qualification_outcome_protocol"
        ):
            return True
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(target, ast.Subscript)
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "qualification_outcome_protocol"
                for target in targets
            ):
                return True
    return False


def _qualification_protocol_adapter_surface_v2(root: Path) -> bool:
    """Measure call compatibility, allowing harmless same-major evolution."""

    path = Path(root) / "research_lab_adapter.py"
    try:
        tree = ast.parse(path.read_bytes())
    except (OSError, SyntaxError, ValueError, UnicodeDecodeError):
        return False
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    metadata = functions.get("adapter_metadata")
    outcome = functions.get(QUALIFICATION_PROTOCOL_ENTRYPOINT_V2)
    if (
        metadata is None
        or outcome is None
        or isinstance(metadata, ast.AsyncFunctionDef)
        or isinstance(outcome, ast.AsyncFunctionDef)
    ):
        return False
    metadata_args = [*metadata.args.posonlyargs, *metadata.args.args]
    outcome_args = [*outcome.args.posonlyargs, *outcome.args.args]
    metadata_required = len(metadata_args) - len(metadata.args.defaults)
    outcome_required = len(outcome_args) - len(outcome.args.defaults)
    return (
        metadata_required == 0
        and not any(item.arg is None for item in metadata.args.kwonlyargs)
        and not any(default is None for default in metadata.args.kw_defaults)
        and len(outcome_args) >= 2
        and outcome_required <= 2
        and not any(default is None for default in outcome.args.kw_defaults)
    )


def qualification_protocol_scoring_adapter_version_v2(
    root: Path,
    *,
    contract: Mapping[str, Any],
) -> str:
    """Measure the signed contract's exact supported scoring adapter.

    The already-published v1 rollback contract did not freeze this constant,
    so v1 alone may use the equivalent literal binding from its hash-bound
    adapter source. New v2 contracts must freeze the exact constant in the
    signed consumer contract. Unknown versions never inherit v2 behavior.
    """

    adapter_path = Path(root) / "research_lab_adapter.py"
    try:
        adapter_tree = ast.parse(adapter_path.read_bytes())
        source_version = _unique_literal_binding_v1(
            adapter_tree,
            "SCORING_ADAPTER_VERSION",
        )
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(
            "qualification protocol scoring adapter binding is unavailable"
        ) from exc
    exact_constants = contract.get("exact_constants")
    adapter_constants = (
        exact_constants.get("research_lab_adapter.py")
        if isinstance(exact_constants, Mapping)
        else None
    )
    contract_version = (
        adapter_constants.get("SCORING_ADAPTER_VERSION")
        if isinstance(adapter_constants, Mapping)
        else None
    )
    if (
        not isinstance(source_version, str)
        or source_version
        not in QUALIFICATION_SUPPORTED_SCORING_ADAPTER_VERSIONS
    ):
        raise ValueError(
            "qualification protocol scoring adapter version is unsupported"
        )
    if contract_version is None:
        if source_version != QUALIFICATION_SCORING_ADAPTER_VERSION_V1:
            raise ValueError(
                "qualification protocol v2 scoring adapter is not frozen in "
                "the signed consumer contract"
            )
    elif contract_version != source_version:
        raise ValueError(
            "qualification protocol scoring adapter differs from the signed "
            "consumer contract"
        )
    return source_version


def qualification_protocol_source_tree_admission_v2(
    root: Path,
    *,
    manifest: Any,
    source_tree_hash: str = "",
) -> Dict[str, Any]:
    """Build a provisional exact-artifact receipt for measured v2 probing."""

    root = Path(root)
    observed_hash = compute_compatibility_source_tree_hash_v1(root)
    claimed_hash = str(source_tree_hash or "")
    document = _manifest_document(manifest)
    if (
        (claimed_hash and claimed_hash != observed_hash)
        or str(document.get("model_artifact_hash") or "") != observed_hash
        or not re.fullmatch(r"[0-9a-f]{40}", str(document.get("git_commit_sha") or ""))
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(document.get("manifest_hash") or ""))
        or not re.fullmatch(
            r"[^\s@]+@sha256:[0-9a-f]{64}",
            str(document.get("image_digest") or ""),
        )
    ):
        raise ValueError(
            "qualification protocol source admission differs from signed artifact"
        )
    contract = document.get("compatibility_contract")
    parity = document.get("consumer_parity_fixtures")
    if not isinstance(contract, Mapping) or not isinstance(parity, Mapping):
        raise ValueError(
            "qualification protocol signed consumer documents are unavailable"
        )
    contract_path = root / str(contract.get("path") or "")
    parity_path = root / str(parity.get("path") or "")
    try:
        source_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if not isinstance(source_contract, Mapping):
            raise ValueError("signed consumer contract must be an object")
        scoring_adapter_version = qualification_protocol_scoring_adapter_version_v2(
            root,
            contract=source_contract,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            "qualification protocol signed consumer contract is invalid"
        ) from exc
    if (
        not contract_path.is_file()
        or not parity_path.is_file()
        or _snapshot_sha256(contract_path) != str(contract.get("sha256") or "")
        or _snapshot_sha256(parity_path) != str(parity.get("sha256") or "")
        or not str(contract.get("contract_id") or "")
        or not isinstance(source_contract, Mapping)
        or source_contract.get("contract_id") != contract.get("contract_id")
        or source_contract.get("canonical_path") != contract.get("path")
        or source_contract.get("parity_fixture_path") != parity.get("path")
        or document.get("scoring_adapter_version")
        != scoring_adapter_version
    ):
        raise ValueError(
            "qualification protocol signed consumer documents differ from source"
        )
    body = {
        "schema_version": QUALIFICATION_PROTOCOL_COMPATIBILITY_RECEIPT_SCHEMA_V2,
        "consumer_api_version": QUALIFICATION_PROTOCOL_CONSUMER_API_V2,
        "decision": SEMANTIC_COMPATIBILITY_ACCEPTED_DECISION,
        "admission_mode": QUALIFICATION_PROTOCOL_ADMISSION_MODE_V2,
        "policy_hash": QUALIFICATION_PROTOCOL_POLICY_SHA256_V2,
        "source_tree_hash": observed_hash,
        "git_commit_sha": str(document["git_commit_sha"]),
        "manifest_hash": str(document["manifest_hash"]),
        "image_digest": str(document["image_digest"]),
        "contract_id": str(contract["contract_id"]),
        "contract_hash": str(contract["sha256"]),
        "parity_hash": str(parity["sha256"]),
        "bindings": {
            "scoring_adapter_version": scoring_adapter_version,
        },
        "entrypoints": sorted(QUALIFICATION_PROTOCOL_REQUIRED_ENTRYPOINTS_V2),
    }
    return {**body, "receipt_hash": _sha256_json(body)}


def validate_qualification_protocol_source_receipt_v2(
    receipt: Mapping[str, Any],
    *,
    manifest: Any,
    source_tree_hash: str,
) -> Dict[str, Any]:
    normalized = dict(receipt)
    document = _manifest_document(manifest)
    body = {key: item for key, item in normalized.items() if key != "receipt_hash"}
    fields = {
        "schema_version",
        "consumer_api_version",
        "decision",
        "admission_mode",
        "policy_hash",
        "source_tree_hash",
        "git_commit_sha",
        "manifest_hash",
        "image_digest",
        "contract_id",
        "contract_hash",
        "parity_hash",
        "bindings",
        "entrypoints",
        "receipt_hash",
    }
    contract = document.get("compatibility_contract")
    parity = document.get("consumer_parity_fixtures")
    if (
        set(normalized) != fields
        or normalized.get("schema_version")
        != QUALIFICATION_PROTOCOL_COMPATIBILITY_RECEIPT_SCHEMA_V2
        or normalized.get("consumer_api_version")
        != QUALIFICATION_PROTOCOL_CONSUMER_API_V2
        or normalized.get("decision") != SEMANTIC_COMPATIBILITY_ACCEPTED_DECISION
        or normalized.get("admission_mode")
        != QUALIFICATION_PROTOCOL_ADMISSION_MODE_V2
        or normalized.get("policy_hash") != QUALIFICATION_PROTOCOL_POLICY_SHA256_V2
        or normalized.get("source_tree_hash") != source_tree_hash
        or normalized.get("source_tree_hash")
        != str(document.get("model_artifact_hash") or "")
        or normalized.get("git_commit_sha")
        != str(document.get("git_commit_sha") or "")
        or normalized.get("manifest_hash")
        != str(document.get("manifest_hash") or "")
        or normalized.get("image_digest") != str(document.get("image_digest") or "")
        or not isinstance(contract, Mapping)
        or not isinstance(parity, Mapping)
        or normalized.get("contract_id") != str(contract.get("contract_id") or "")
        or normalized.get("contract_hash") != str(contract.get("sha256") or "")
        or normalized.get("parity_hash") != str(parity.get("sha256") or "")
        or normalized.get("bindings")
        != {
            "scoring_adapter_version": str(
                document.get("scoring_adapter_version") or ""
            )
        }
        or normalized.get("entrypoints")
        != sorted(QUALIFICATION_PROTOCOL_REQUIRED_ENTRYPOINTS_V2)
        or normalized.get("receipt_hash") != _sha256_json(body)
    ):
        raise ValueError(
            "qualification protocol compatibility receipt differs from signed artifact"
        )
    return normalized


def source_tree_compatibility_admission(
    root: Path,
    *,
    manifest: Any = None,
    source_tree_hash: str = "",
    use_cache: bool = False,
) -> Dict[str, Any]:
    """Route exact legacy/semantic-v1 trees or measured protocol-v2 trees."""

    if _qualification_protocol_entrypoint_declared_v2(root):
        return qualification_protocol_source_tree_admission_v2(
            root,
            manifest=manifest,
            source_tree_hash=source_tree_hash,
        )
    return source_tree_compatibility_admission_v1(
        root,
        manifest=manifest,
        source_tree_hash=source_tree_hash,
        use_cache=use_cache,
    )


def validate_source_tree_compatibility_receipt(
    receipt: Mapping[str, Any],
    *,
    manifest: Any,
    source_tree_hash: str,
    policy: Mapping[str, Any] | None = None,
    policy_hash: str = "",
) -> Dict[str, Any]:
    if receipt.get("admission_mode") == QUALIFICATION_PROTOCOL_ADMISSION_MODE_V2:
        return validate_qualification_protocol_source_receipt_v2(
            receipt,
            manifest=manifest,
            source_tree_hash=source_tree_hash,
        )
    return validate_source_tree_compatibility_receipt_v1(
        receipt,
        manifest=manifest,
        source_tree_hash=source_tree_hash,
        policy=policy,
        policy_hash=policy_hash,
    )


def source_tree_compatibility_admission_v1(
    root: Path,
    *,
    manifest: Any = None,
    source_tree_hash: str = "",
    use_cache: bool = False,
) -> Dict[str, Any]:
    """Admit an exact legacy snapshot or an unseen semantic-v1 contract.

    The tree is recomputed before cache lookup and after admission. A caller
    digest is only an additional equality claim; it is never cache authority.
    The cache key binds the verified source, signed manifest, immutable image,
    consumer API, policy, and accepted decision.
    """

    root = Path(root)
    manifest_document = deepcopy(_manifest_document(manifest))
    policy, policy_hash = semantic_compatibility_policy_identity_v1()
    consumer_api_version = str(policy["consumer_api_version"])
    observed_source_tree_hash = compute_compatibility_source_tree_hash_v1(root)
    claimed_source_tree_hash = str(source_tree_hash or "")
    if (
        claimed_source_tree_hash
        and claimed_source_tree_hash != observed_source_tree_hash
    ):
        raise ValueError(
            "sourcing model compatibility admission failed: "
            "caller source tree hash differs from canonical extraction"
        )
    if manifest_document and str(
        manifest_document.get("model_artifact_hash") or ""
    ) != observed_source_tree_hash:
        raise ValueError(
            "sourcing model compatibility admission failed: "
            "signed manifest source tree hash differs from canonical extraction"
        )
    cache_key = (
        observed_source_tree_hash,
        str(manifest_document.get("manifest_hash") or ""),
        str(manifest_document.get("image_digest") or ""),
        str(manifest_document.get("git_commit_sha") or ""),
        policy_hash,
        consumer_api_version,
        SEMANTIC_COMPATIBILITY_ACCEPTED_DECISION,
    )
    cacheable = bool(use_cache and all(cache_key))
    cached_receipt: Dict[str, Any] | None = None
    if cacheable:
        with _SEMANTIC_COMPATIBILITY_CACHE_LOCK:
            cached = _SEMANTIC_COMPATIBILITY_CACHE.get(cache_key)
            if cached is not None:
                _SEMANTIC_COMPATIBILITY_CACHE.move_to_end(cache_key)
                cached_receipt = deepcopy(cached)
    if cached_receipt is not None:
        try:
            cached_receipt = validate_source_tree_compatibility_receipt_v1(
                cached_receipt,
                manifest=manifest_document,
                source_tree_hash=observed_source_tree_hash,
                policy=policy,
                policy_hash=policy_hash,
            )
        except ValueError as exc:
            raise ValueError(
                "sourcing model compatibility admission failed: "
                "cached receipt differs from signed artifact"
            ) from exc
        if (
            compute_compatibility_source_tree_hash_v1(root)
            != observed_source_tree_hash
        ):
            raise ValueError(
                "sourcing model compatibility admission failed: "
                "source tree changed during cached admission"
            )
        if semantic_compatibility_policy_hash_v1() != policy_hash:
            raise ValueError(
                "sourcing model compatibility policy changed during cached admission"
            )
        return cached_receipt

    try:
        snapshot = _reviewed_consumer_snapshot_for_source_hash(
            root,
            source_tree_hash=observed_source_tree_hash,
            manifest=manifest_document,
        )
    except ValueError as exc:
        raise ValueError(
            "sourcing model compatibility admission failed: " + str(exc)
        ) from exc
    if snapshot is None:
        violations, receipt = verify_semantic_source_tree_compatibility_v1(
            root,
            manifest=manifest_document,
            source_tree_hash=observed_source_tree_hash,
            _policy=policy,
            _policy_hash=policy_hash,
        )
    else:
        violations = _verify_source_tree_contract_document(
            root,
            document=snapshot["contract"],
            reviewed_snapshot=snapshot,
            verify_snapshot_pair=True,
        )
        contract = dict(snapshot["contract"])
        contract_hash = str(snapshot["contract_sha256"])
        parity_hash = str(snapshot["parity_sha256"])
        violations.extend(
            _manifest_pair_violations(
                manifest=manifest_document,
                contract_id=str(contract["contract_id"]),
                contract_path=str(contract["canonical_path"]),
                contract_hash=contract_hash,
                parity_path=str(contract["parity_fixture_path"]),
                parity_hash=parity_hash,
                source_tree_hash=observed_source_tree_hash,
            )
        )
        exact_constants = dict(contract.get("exact_constants") or {})
        adapter_constants = dict(
            exact_constants.get("research_lab_adapter.py") or {}
        )
        compiler_constants = dict(
            exact_constants.get("sourcing_model/routing/compiler.py") or {}
        )
        reviewed_runtime_constants = dict(
            dict(snapshot.get("required_source_constants") or {}).get(
                "sourcing_model/runtime_capabilities.py"
            )
            or {}
        )
        bindings = {
            "adapter_version": str(adapter_constants.get("ADAPTER_VERSION") or ""),
            "capability_contract_version": str(
                dict(
                    exact_constants.get("sourcing_model/runtime_capabilities.py")
                    or {}
                ).get("CAPABILITY_CONTRACT_VERSION")
                or reviewed_runtime_constants.get("CAPABILITY_CONTRACT_VERSION")
                or "sourcing-model-runtime-capabilities:v2"
            ),
            "component_registry_version": str(
                adapter_constants.get("COMPONENT_REGISTRY_VERSION") or ""
            ),
            "routing_compiler_version": str(
                compiler_constants.get("COMPILER_VERSION") or ""
            ),
            "scoring_adapter_version": str(
                adapter_constants.get("SCORING_ADAPTER_VERSION")
                or "qualification-company-scorer:v1"
            ),
        }
        receipt = (
            None
            if violations
            else _semantic_compatibility_receipt(
                mode="legacy_exact",
                consumer_api_version=consumer_api_version,
                policy_hash=policy_hash,
                source_tree_hash=observed_source_tree_hash,
                manifest=manifest_document,
                contract=contract,
                contract_hash=contract_hash,
                parity_hash=parity_hash,
                bindings=bindings,
            )
        )
    if violations or receipt is None:
        raise ValueError(
            "sourcing model compatibility admission failed: "
            + "; ".join(violations[:12])
        )
    receipt = validate_source_tree_compatibility_receipt_v1(
        receipt,
        manifest=manifest_document,
        source_tree_hash=observed_source_tree_hash,
        policy=policy,
        policy_hash=policy_hash,
    )
    if (
        compute_compatibility_source_tree_hash_v1(root)
        != observed_source_tree_hash
    ):
        raise ValueError(
            "sourcing model compatibility admission failed: "
            "source tree changed during admission"
        )
    if semantic_compatibility_policy_hash_v1() != policy_hash:
        raise ValueError(
            "semantic sourcing compatibility policy changed during admission"
        )
    if cacheable:
        with _SEMANTIC_COMPATIBILITY_CACHE_LOCK:
            _SEMANTIC_COMPATIBILITY_CACHE[cache_key] = deepcopy(receipt)
            _SEMANTIC_COMPATIBILITY_CACHE.move_to_end(cache_key)
            while len(_SEMANTIC_COMPATIBILITY_CACHE) > _SEMANTIC_COMPATIBILITY_CACHE_SIZE:
                _SEMANTIC_COMPATIBILITY_CACHE.popitem(last=False)
    return deepcopy(receipt)


def clear_source_tree_compatibility_admission_cache_v1() -> None:
    """Clear semantic admission decisions (tests and bounded maintenance)."""

    with _SEMANTIC_COMPATIBILITY_CACHE_LOCK:
        _SEMANTIC_COMPATIBILITY_CACHE.clear()
