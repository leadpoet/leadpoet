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

from lab_arena import benchmark, broker as broker_module, chain as chain_module, contracts, credentials, model_release, runtime, scoring, signing
from lab_arena.api import create_app
from lab_arena.contracts import ArenaContractError
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


def credential_registrar(*, decryptor, urlopen=urllib.request.urlopen, clock=None):
    """The ``POST /credentials/{provider}`` handler: decrypt once in the broker identity, probe the provider, record."""

    def register(envelope: Mapping[str, Any]) -> Dict[str, Any]:
        try:
            return credentials.register_provider_key(envelope, decryptor=decryptor, urlopen=urlopen, expected_recipient_key_hash=decryptor.recipient_key_hash, now=clock)
        except credentials.ProviderKeyError as exc:
            raise ServiceError("credential_rejected:%s" % str(exc)[:80], 400) from exc
        except ArenaContractError as exc:
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


def _daily_cutoff_hour_from_environment() -> Optional[int]:
    """LAB_ARENA_DAILY_CUTOFF_UTC: the UTC hour of each day's cutoff (0..23); unset leaves round creation to the operator."""

    raw = os.environ.get("LAB_ARENA_DAILY_CUTOFF_UTC", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ServiceError("LAB_ARENA_DAILY_CUTOFF_UTC must be an integer hour", 500) from None
    if value < 0 or value > 23:
        raise ServiceError("LAB_ARENA_DAILY_CUTOFF_UTC must be between 0 and 23", 500)
    return value


def _max_challengers_from_environment() -> int:
    """LAB_ARENA_MAX_CHALLENGERS: the admitted challenger ceiling per round, 1..MAX_CHALLENGERS."""

    raw = os.environ.get("LAB_ARENA_MAX_CHALLENGERS", "").strip()
    if not raw:
        return contracts.MAX_CHALLENGERS
    try:
        value = int(raw)
    except ValueError:
        raise ServiceError("LAB_ARENA_MAX_CHALLENGERS must be an integer", 500) from None
    if value < 1 or value > contracts.MAX_CHALLENGERS:
        raise ServiceError("LAB_ARENA_MAX_CHALLENGERS must be between 1 and %d" % contracts.MAX_CHALLENGERS, 500)
    return value


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
    # Miners bring every provider key; the Arena holds no Exa, Scrapingdog, or Deepline credential of its own.
    decryptor = credentials.KmsDecryptor(_required("LAB_ARENA_OPENROUTER_KMS_KEY_ID"), region_name=os.environ.get("AWS_REGION"))
    lock = runtime.load_runtime_lock()
    floor = tuple(item for item in os.environ.get("LAB_ARENA_FLOOR_RUNNER_HOTKEYS", "").split(",") if item.strip())
    defaults = RoundDefaults(
        floor_runner_hotkeys=floor,
        openrouter_allowed_models=tuple(item for item in os.environ.get("LAB_ARENA_OPENROUTER_ALLOWED_MODELS", "openai/gpt-4o-mini").split(",") if item.strip()),
        base_image_digest=os.environ.get("LAB_ARENA_BASE_IMAGE_DIGEST", "sha256:" + "0" * 64),
        repository_commit=os.environ.get("LAB_ARENA_REPOSITORY_COMMIT", "0" * 40),
        max_challengers=_max_challengers_from_environment(),
        daily_cutoff_hour_utc=_daily_cutoff_hour_from_environment(),
        scorer_image_digest=_required("LAB_ARENA_SCORER_IMAGE_DIGEST"),
    )

    def key_for(miner_hotkey: str, provider: str) -> credentials.RuntimeKeyHandle:
        """Decrypt the miner's stored envelope for one provider, per call, in memory only."""

        account = store.get_account(miner_hotkey)
        entry = (account or {}).get("credentials", {}).get(provider) if account else None
        if not isinstance(entry, Mapping) or not entry.get("ciphertext"):
            raise broker_module.BrokerError("broker_unavailable")
        envelope = json.loads(entry["ciphertext"])
        handle = credentials.decrypt_runtime_key(envelope, decryptor)
        if handle.provider != provider:
            raise broker_module.BrokerError("broker_unavailable")
        return handle

    def broker_factory(service: ArenaService, round_row: Mapping[str, Any]) -> broker_module.Broker:
        table = json.loads(objects.get("arena/%s/price_table.json" % round_row["round_id"]).decode("utf-8"))
        judge_models = sorted({str(model) for model in (service.scorer_policy.get("judge_models") or {}).values() if model})
        return broker_module.Broker(store=store, key_for=key_for, judge_models=judge_models, price_table=table, allowed_models=round_row["configuration_doc"]["openrouter_allowed_models"], transport=broker_module.HttpxProviderTransport())

    def scorer_factory(policy: Mapping[str, Any]) -> scoring.Scorer:
        scoring.apply_policy_to_environment(policy, environ=os.environ, cache_dir=_required("LAB_ARENA_SCORING_CACHE_DIR"), credentials={name: _required("LAB_ARENA_SCORING_" + name) for name in scoring.CREDENTIAL_ENV_NAMES})
        return scoring.lab_scorer(policy)

    # The winning model is committed to the public sales-agent repository in
    # live mode only; shadow rounds never publish a model. Live mode fails
    # closed without the token so a release is never silently skipped.
    github_token = os.environ.get("LAB_ARENA_GITHUB_TOKEN", "").strip()
    model_release_client = None
    if mode == "live":
        if not github_token:
            raise ServiceError("LAB_ARENA_GITHUB_TOKEN is required in live mode", 500)
        model_release_client = model_release.GitHubClient(os.environ.get("LAB_ARENA_MODEL_REPOSITORY", model_release.DEFAULT_REPOSITORY).strip(), github_token)
    config = ServiceConfig(
        mode=mode, store=store, object_store=objects, signer=signer, chain=chain_reads, verify_signature=chain_module.verify_hotkey_signature,
        model_release_client=model_release_client, model_release_branch=os.environ.get("LAB_ARENA_MODEL_BRANCH", model_release.DEFAULT_BRANCH).strip() or model_release.DEFAULT_BRANCH,
        generation_provider=generation_provider, price_table_source=lambda models: broker_module.fetch_openrouter_price_table(models),
        banned_hotkeys_source=banned_hotkeys_from_environment, broker_factory=broker_factory, scorer_factory=scorer_factory, defaults=defaults,
        runtime_lock_hash=lock.runtime_lock_hash, scoring_workers=int(os.environ.get("LAB_ARENA_SCORING_WORKERS", "4")),
        # Replay recomputation of every accepted scoring assignment stays on in production.
        replay_verification=os.environ.get("LAB_ARENA_REPLAY_VERIFICATION", "1").strip() != "0",
        replay_work_dir=os.environ.get("LAB_ARENA_REPLAY_WORK_DIR") or None,
    )
    service = ArenaService(config)
    recipient = credentials.recipient_document(decryptor.public_key_der)
    app = create_app(
        service,
        recipient_document=recipient,
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
    # Fail startup unless this runner's identities are exactly what the round pins.
    runner_module.verify_release_against_round(api.round(args.round_id).get("configuration") or {}, worker_release_hash=release["worker_release_hash"], runtime_lock_hash=lock.runtime_lock_hash)
    runner_config = runner_module.RunnerConfig(
        round_id=args.round_id, identity=identity, api=api, sandbox_runtime=sandbox_runtime, image_cache=cache, worker_release_hash=release["worker_release_hash"],
        work_dir=Path(args.work_dir) / "runs", max_parallel_runs=runner_module.max_parallel_runs_from_environment(), evaluation_date=args.round_id.replace("arena-", "")[:10],
    )
    Path(args.work_dir, "runs").mkdir(parents=True, exist_ok=True)
    return runner_module.Runner(runner_config)
