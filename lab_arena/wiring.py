"""Production wiring from the environment for the service, runner, and admin
entrypoints (labarena.md sections 4, 15.1, 16). Imported lazily by the
scripts so ``--help`` and the boundary tests never touch AWS, the chain, or
Supabase. Secret values are read from the environment and passed into
objects that never print them.
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from lab_arena import benchmark, broker as broker_module, chain as chain_module, contracts, credentials, funding, runtime, scoring, signing
from lab_arena.api import create_app
from lab_arena.service import ArenaService, RoundDefaults, S3ObjectStore, ServiceConfig, ServiceError
from lab_arena.store import ArenaStore, PostgrestTransport


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ServiceError("environment %s is required" % name, 500)
    return value


class OpenRouterGenerationProvider:
    """Benchmark generation through the Arena's own OpenRouter account."""

    def __init__(self, api_key: str, *, urlopen=urllib.request.urlopen) -> None:
        if not api_key:
            raise ServiceError("generation api key is required", 500)
        self._api_key = api_key
        self._urlopen = urlopen

    def __repr__(self) -> str:
        return "OpenRouterGenerationProvider(<redacted>)"

    def chat(self, *, messages, temperature, max_tokens, timeout_seconds):
        body: Dict[str, Any] = {"model": benchmark.GENERATOR_MODEL, "messages": list(messages), "temperature": temperature}
        if max_tokens is not None:
            body["max_tokens"] = int(max_tokens)
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": "Bearer " + self._api_key, "Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with self._urlopen(request, timeout=float(timeout_seconds)) as response:
                raw = response.read(16 * 1024 * 1024)
                status = getattr(response, "status", 200)
        except Exception as exc:  # transport or HTTP failure: the outcome is unknown
            raise benchmark.ProviderFailure(type(exc).__name__) from exc
        if status != 200:
            raise benchmark.ProviderFailure("http_%s" % status)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise benchmark.ProviderFailure("invalid_json") from exc


class ChainReadsAdapter:
    """The service's chain reads over ``ArenaChain`` plus the epoch cutover."""

    def __init__(self, arena_chain: chain_module.ArenaChain, cutover: Any) -> None:
        self._chain = arena_chain
        self._cutover = cutover

    def finalized_head(self):
        return self._chain.finalized_head()

    def metagraph(self, finalized: bool = True):
        return self._chain.metagraph(finalized=finalized)

    def current_settlement_epoch(self) -> int:
        return chain_module.current_settlement_epoch(self._chain, self._cutover)

    def hotkeys_owned_by_same_coldkey(self, hotkey: str) -> List[str]:
        snapshot = self._chain.metagraph()
        uid = chain_module.uid_for_hotkey(snapshot, hotkey)
        if uid is None:
            return []
        return chain_module.hotkeys_owned_by_coldkey(snapshot, snapshot.coldkeys[uid])

    def uid_for_hotkey(self, hotkey: str) -> Optional[int]:
        return chain_module.uid_for_hotkey(self._chain.metagraph(), hotkey)

    def validator_permit_hotkeys(self) -> List[str]:
        snapshot = self._chain.metagraph()
        return [hotkey for hotkey, permit in zip(snapshot.hotkeys, snapshot.validator_permit) if permit]


def funding_confirmer(*, chain, config: funding.FundingConfig, store, price_source, clock=None):
    """The ``POST /funding/confirm`` handler: verify one finalized transfer, credit once.

    Returns structured, secret-free results; verification failures name the
    published rule and create no credit.
    """

    def confirm(miner_hotkey: str, body: Mapping[str, Any]) -> Dict[str, Any]:
        moment = (clock or (lambda: datetime.now(timezone.utc)))()
        try:
            receipt = funding.confirm_funding(
                chain=chain, config=config, store=store, miner_hotkey=miner_hotkey,
                block_hash=body.get("block_hash"), extrinsic_index=body.get("extrinsic_index"), now=moment, price_source=price_source,
            )
        except funding.DepositRejected as exc:
            return {"credited": False, "idempotent": False, "rejected": True, "rule": exc.rule_id}
        except funding.MalformedReference as exc:
            return {"credited": False, "idempotent": False, "rejected": True, "rule": "reference_malformed"}
        except (funding.PriceUnavailable, funding.FundingStoreError, chain_module.ArenaChainError) as exc:
            raise ServiceError("funding_unavailable:%s" % type(exc).__name__, 503) from exc
        return {
            "credited": bool(receipt.credited), "idempotent": bool(receipt.idempotent), "balance_microusd": int(receipt.balance_microusd),
            "payment_reference": receipt.deposit_doc.get("payment_reference"), "amount_microusd": receipt.deposit_doc.get("amount_microusd"),
        }

    return confirm


def credential_registrar(*, decryptor, urlopen=urllib.request.urlopen, clock=None):
    """The ``POST /credentials/openrouter`` handler: decrypt once in the broker identity, preflight, record."""

    def register(envelope: Mapping[str, Any]) -> Dict[str, Any]:
        try:
            return credentials.register_openrouter_key(envelope, decryptor=decryptor, urlopen=urlopen, expected_recipient_key_hash=decryptor.recipient_key_hash, now=clock)
        except credentials.OpenRouterKeyError as exc:
            raise ServiceError("credential_rejected:%s" % str(exc)[:80], 400) from exc
        except contracts.ArenaContractError as exc:
            raise ServiceError("envelope_invalid:%s" % str(exc)[:80], 400) from exc

    return register


def banned_hotkeys_from_environment() -> List[str]:
    """Operator-provided ban snapshot: a JSON list at LAB_ARENA_BANNED_HOTKEYS_PATH."""

    path = os.environ.get("LAB_ARENA_BANNED_HOTKEYS_PATH", "").strip()
    if not path:
        return []
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, list):
        raise ServiceError("banned hotkeys snapshot must be a JSON list", 500)
    return [str(item) for item in document]


def build_service_from_environment(mode: str):
    """Construct the production service and its FastAPI app from the environment."""

    transport = PostgrestTransport(_required("LAB_ARENA_SUPABASE_URL"), anon_key=_required("LAB_ARENA_SUPABASE_ANON_KEY"), service_jwt=_required("LAB_ARENA_SERVICE_JWT"))
    store = ArenaStore(transport)
    signer = signing.KmsSigner(_required("LAB_ARENA_SIGNING_KEY_ID"), region_name=os.environ.get("AWS_REGION"))
    objects = S3ObjectStore(_required("LAB_ARENA_BUCKET"), region_name=os.environ.get("AWS_REGION"))
    chain_config = chain_module.ArenaChainConfig(endpoint=_required("LAB_ARENA_CHAIN_ENDPOINT"), netuid=int(os.environ.get("LAB_ARENA_NETUID", "71")), network_name=os.environ.get("LAB_ARENA_NETWORK", "finney"), request_timeout_seconds=int(os.environ.get("LAB_ARENA_CHAIN_TIMEOUT_SECONDS", "30")))
    arena_chain = chain_module.ArenaChain(chain_config, chain_module.connect_substrate(chain_config))
    cutover = chain_module.load_arena_cutover()
    chain_reads = ChainReadsAdapter(arena_chain, cutover)
    generation_provider = OpenRouterGenerationProvider(_required("LAB_ARENA_GENERATION_OPENROUTER_API_KEY"))
    provider_credentials = broker_module.ArenaProviderCredentials(exa_api_key=_required("LAB_ARENA_EXA_API_KEY"), scrapingdog_api_key=_required("LAB_ARENA_SCRAPINGDOG_API_KEY"))
    decryptor = credentials.KmsDecryptor(_required("LAB_ARENA_OPENROUTER_KMS_KEY_ID"), region_name=os.environ.get("AWS_REGION"))
    lock = runtime.load_runtime_lock()
    floor = tuple(item for item in os.environ.get("LAB_ARENA_FLOOR_RUNNER_HOTKEYS", "").split(",") if item.strip())
    defaults = RoundDefaults(
        floor_runner_hotkeys=floor,
        openrouter_allowed_models=tuple(item for item in os.environ.get("LAB_ARENA_OPENROUTER_ALLOWED_MODELS", "openai/gpt-4o-mini").split(",") if item.strip()),
        base_image_digest=os.environ.get("LAB_ARENA_BASE_IMAGE_DIGEST", "sha256:" + "0" * 64),
        repository_commit=os.environ.get("LAB_ARENA_REPOSITORY_COMMIT", "0" * 40),
    )

    def openrouter_key_for(miner_hotkey: str) -> credentials.RuntimeKeyHandle:
        account = store.get_account(miner_hotkey)
        if account is None or not account.get("openrouter_ciphertext"):
            raise broker_module.BrokerError("broker_unavailable")
        envelope = json.loads(account["openrouter_ciphertext"])
        return credentials.decrypt_runtime_key(envelope, decryptor)

    def broker_factory(service: ArenaService, round_row: Mapping[str, Any]) -> broker_module.Broker:
        table = json.loads(objects.get("arena/%s/price_table.json" % round_row["round_id"]).decode("utf-8"))
        return broker_module.Broker(store=store, credentials=provider_credentials, openrouter_key_for=openrouter_key_for, price_table=table, allowed_models=round_row["configuration_doc"]["openrouter_allowed_models"], transport=broker_module.HttpxProviderTransport())

    def scorer_factory(policy: Mapping[str, Any]) -> scoring.Scorer:
        scoring.apply_policy_to_environment(policy, environ=os.environ, cache_dir=_required("LAB_ARENA_SCORING_CACHE_DIR"), credentials={name: _required("LAB_ARENA_SCORING_" + name) for name in scoring.CREDENTIAL_ENV_NAMES})
        return scoring.lab_scorer(policy)

    config = ServiceConfig(
        mode=mode, store=store, object_store=objects, signer=signer, chain=chain_reads, verify_signature=chain_module.verify_hotkey_signature,
        generation_provider=generation_provider, price_table_source=lambda models: broker_module.fetch_openrouter_price_table(models),
        banned_hotkeys_source=banned_hotkeys_from_environment, broker_factory=broker_factory, scorer_factory=scorer_factory, defaults=defaults,
        runtime_lock_hash=lock.runtime_lock_hash, scoring_workers=int(os.environ.get("LAB_ARENA_SCORING_WORKERS", "4")),
    )
    service = ArenaService(config)
    recipient = credentials.recipient_document(decryptor.public_key_der)
    funding_config = funding.FundingConfig(recipient_wallet=_required("LAB_ARENA_TAO_RECIPIENT_WALLET"), network_name=config.network_name)
    app = create_app(
        service,
        recipient_document=recipient,
        funding_confirm=funding_confirmer(chain=arena_chain, config=funding_config, store=store, price_source=funding.coingecko_price_source()),
        credential_register=credential_registrar(decryptor=decryptor),
    )
    return service, app


def build_runner_from_environment(args):
    from lab_arena import runner as runner_module

    from bittensor_wallet import Wallet

    wallet = Wallet(name=args.wallet_name, hotkey=args.hotkey_name)
    keypair = wallet.hotkey
    lock = runtime.load_runtime_lock()
    config = runtime.RuntimeConfig(runsc_path=Path(args.runsc_path), lock=lock, work_dir=Path(args.work_dir) / "sandboxes")
    sandbox_runtime = runtime.RunscRuntime(config)
    identity = runner_module.RunnerIdentity(hotkey=keypair.ss58_address, sign=lambda message: keypair.sign(message.encode("utf-8")).hex())
    release = runner_module.worker_release_identity(repository_commit=os.environ.get("LAB_ARENA_REPOSITORY_COMMIT", "0" * 40), runtime_lock_hash=lock.runtime_lock_hash)
    cache = runner_module.ImageCache(Path(args.work_dir) / "images", runner_module.docker_image_exporter)
    api = runner_module.HttpArenaApiClient(args.api_base_url)
    runner_config = runner_module.RunnerConfig(
        round_id=args.round_id, identity=identity, api=api, sandbox_runtime=sandbox_runtime, image_cache=cache, worker_release_hash=release["worker_release_hash"],
        work_dir=Path(args.work_dir) / "runs", max_parallel_runs=runner_module.max_parallel_runs_from_environment(), evaluation_date=args.round_id.replace("arena-", "")[:10],
    )
    Path(args.work_dir, "runs").mkdir(parents=True, exist_ok=True)
    return runner_module.Runner(runner_config)
