"""Production wiring from the environment for the service, runner, and admin
entrypoints (labarena.md sections 4, 15.1, 16). Imported lazily by the
scripts so ``--help`` and the boundary tests never touch AWS, the chain, or
Supabase. Secret values are read from the environment and passed into
objects that never print them.
"""

from __future__ import annotations

import atexit
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from lab_arena import broker as broker_module, chain as chain_module, contracts, images, runtime, signing
from lab_arena.api import create_app
from lab_arena.credentials import CredentialManager
from lab_arena.service import ArenaService, RoundDefaults, S3ObjectStore, ServiceConfig, ServiceError
from lab_arena.source_bundle import MAX_SOURCE_ARCHIVE_BYTES
from lab_arena.store import ArenaStore, PostgrestTransport
from lab_arena.submission_runtime import SubmissionProviderKeys


_DIRECT_URLOPEN = urllib.request.build_opener(urllib.request.ProxyHandler({})).open


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ServiceError("environment %s is required" % name, 500)
    return value


def fetch_public_source_archive(url: str, max_bytes: int) -> bytes:
    """Download one public HTTPS archive without credentials or proxy state."""

    if not str(url).startswith("https://"):
        raise ServiceError("baseline source URL must use https", 500)
    limit = min(int(max_bytes), MAX_SOURCE_ARCHIVE_BYTES)
    request = urllib.request.Request(
        str(url), headers={"Accept": "application/gzip, application/octet-stream"}
    )
    try:
        with _DIRECT_URLOPEN(request, timeout=60.0) as response:
            if int(getattr(response, "status", 200)) != 200:
                raise ServiceError("baseline source download failed", 503)
            final_url = str(getattr(response, "geturl", lambda: url)())
            if not final_url.startswith("https://"):
                raise ServiceError("baseline source redirect must use https", 500)
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > limit:
                raise ServiceError("baseline source archive is too large", 500)
            data = response.read(limit + 1)
    except ServiceError:
        raise
    except Exception as exc:
        raise ServiceError("baseline source download failed", 503) from exc
    if not 1 <= len(data) <= limit:
        raise ServiceError("baseline source archive is too large", 500)
    return data


class ChainReadsAdapter:
    """The service's chain reads over ``ArenaChain`` plus the epoch cutover."""

    def __init__(self, arena_chain: chain_module.ArenaChain) -> None:
        self._chain = arena_chain
        self._cutover = None

    def finalized_head(self):
        return self._chain.finalized_head()

    def metagraph(self, finalized: bool = True):
        return self._chain.metagraph(finalized=finalized)

    def current_settlement_epoch(self) -> int:
        if self._cutover is None:
            self._cutover = chain_module.load_arena_cutover()
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
    """LAB_ARENA_DAILY_CUTOFF_UTC: each daily cutoff hour (0..23), default 00:00 UTC."""

    raw = os.environ.get("LAB_ARENA_DAILY_CUTOFF_UTC", "0").strip() or "0"
    try:
        value = int(raw)
    except ValueError:
        raise ServiceError("LAB_ARENA_DAILY_CUTOFF_UTC must be an integer hour", 500) from None
    if value < 0 or value > 23:
        raise ServiceError("LAB_ARENA_DAILY_CUTOFF_UTC must be between 0 and 23", 500)
    return value


def _pool_percent_from_environment() -> int:
    """The king's pool as a percent of total emissions: LAB_ARENA_POOL_PERCENT, default 25, integer 0..100."""

    raw = os.environ.get("LAB_ARENA_POOL_PERCENT", "").strip()
    if not raw:
        return int(contracts.LAB_ARENA_POOL_PERCENT)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ServiceError("LAB_ARENA_POOL_PERCENT must be an integer percent", 500) from exc
    if not 0 <= value <= 100:
        raise ServiceError("LAB_ARENA_POOL_PERCENT must be within 0..100", 500)
    return value


def _rewards_enabled_from_environment() -> bool:
    """Freeze LAB_ARENA_REWARDS_ENABLED into each new live round."""

    raw = os.environ.get("LAB_ARENA_REWARDS_ENABLED", "false").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ServiceError("LAB_ARENA_REWARDS_ENABLED must be true or false", 500)


def _max_challengers_from_environment() -> int:
    """LAB_ARENA_MAX_CHALLENGERS: the admitted challenger ceiling per round, 1..MAX_CHALLENGERS."""

    raw = os.environ.get("LAB_ARENA_MAX_CHALLENGERS", "").strip()
    if not raw:
        return contracts.DEFAULT_MAX_CHALLENGERS
    try:
        value = int(raw)
    except ValueError:
        raise ServiceError("LAB_ARENA_MAX_CHALLENGERS must be an integer", 500) from None
    if value < 1 or value > contracts.MAX_CHALLENGERS:
        raise ServiceError("LAB_ARENA_MAX_CHALLENGERS must be between 1 and %d" % contracts.MAX_CHALLENGERS, 500)
    return value


def _runner_hotkeys_from_environment() -> tuple[str, ...]:
    runners = tuple(
        item.strip()
        for item in os.environ.get("LAB_ARENA_RUNNER_HOTKEYS", "").split(",")
        if item.strip()
    )
    if not runners:
        raise ServiceError("environment LAB_ARENA_RUNNER_HOTKEYS is required", 500)
    return runners


def _max_image_bytes_from_environment() -> int:
    """LAB_ARENA_MAX_IMAGE_BYTES: compressed size ceiling of the trusted scorer."""

    raw = os.environ.get("LAB_ARENA_MAX_IMAGE_BYTES", "").strip()
    if not raw:
        return images.DEFAULT_MAX_IMAGE_BYTES
    try:
        value = int(raw)
    except ValueError:
        raise ServiceError("LAB_ARENA_MAX_IMAGE_BYTES must be an integer", 500) from None
    if value < 1:
        raise ServiceError("LAB_ARENA_MAX_IMAGE_BYTES must be positive", 500)
    return value


def registry_client_from_environment() -> images.RegistryClient:
    """Create the read-only client for the organizer's trusted scorer image."""

    repository = os.environ.get("LAB_ARENA_REGISTRY_REPOSITORY", "").strip()
    username = os.environ.get("LAB_ARENA_REGISTRY_USERNAME", "").strip()
    password = os.environ.get("LAB_ARENA_REGISTRY_PASSWORD", "")
    registry_host = images.parse_repository(repository)[0] if repository else ""

    def credentials_for(host: str):
        if username and password and registry_host and host == registry_host:
            return (username, password)
        return None

    return images.RegistryClient(credentials=credentials_for)


def build_service_from_environment(mode: str):
    """Construct the production service and its FastAPI app from the environment."""

    transport = PostgrestTransport(_required("LAB_ARENA_SUPABASE_URL"), anon_key=_required("LAB_ARENA_SUPABASE_ANON_KEY"), service_jwt=_required("LAB_ARENA_SERVICE_JWT"))
    store = ArenaStore(transport)
    objects = S3ObjectStore(_required("LAB_ARENA_BUCKET"), region_name=os.environ.get("AWS_REGION"))
    chain_config = chain_module.ArenaChainConfig(endpoint=_required("LAB_ARENA_CHAIN_ENDPOINT"), netuid=int(os.environ.get("LAB_ARENA_NETUID", "71")), network_name=os.environ.get("LAB_ARENA_NETWORK", "finney"), request_timeout_seconds=int(os.environ.get("LAB_ARENA_CHAIN_TIMEOUT_SECONDS", "30")))
    arena_chain = chain_module.ArenaChain(chain_config, chain_module.connect_substrate(chain_config))
    atexit.register(arena_chain.close)  # the websocket threads would otherwise hold the process open at exit
    chain_reads = ChainReadsAdapter(arena_chain)
    provider_keys = {
        "openrouter": _required("LAB_ARENA_OPENROUTER_API_KEY"),
        "scrapingdog": _required("LAB_ARENA_SCRAPINGDOG_API_KEY"),
        "deepline": _required("LAB_ARENA_DEEPLINE_API_KEY"),
    }
    credential_key_id = os.environ.get("LAB_ARENA_CREDENTIAL_KMS_KEY_ID", "").strip()
    # A missing miner vault must not interrupt baseline rebenchmarking. New
    # miner admission fails closed until the vault is configured.
    credential_manager = CredentialManager(kms_key_id=credential_key_id) if credential_key_id else None
    submission_keys = SubmissionProviderKeys(
        store=store, credentials=credential_manager, organizer_keys=provider_keys
    )
    runners = _runner_hotkeys_from_environment()
    # The trusted scorer remains an organizer-owned internal image. Miner and
    # baseline agents enter as source archives and need no registry account.
    registry = images.RegistryClient()
    image_rules = images.ImageRules(max_image_bytes=_max_image_bytes_from_environment())
    try:
        scorer = images.resolve_image(registry, images.parse_reference(_required("LAB_ARENA_SCORER_IMAGE")), image_rules)
    except images.ImageError as exc:
        raise ServiceError("scorer_image_unresolved:%s" % exc.rule_id, 500) from exc
    finally:
        registry.close()
    defaults = RoundDefaults(
        runner_hotkeys=runners,
        baseline_hotkey=_required("LAB_ARENA_BASELINE_HOTKEY"),
        baseline_source_url=os.environ.get(
            "LAB_ARENA_BASELINE_SOURCE_URL",
            "https://github.com/leadpoet/pydantic-harness/archive/refs/heads/main.tar.gz",
        ).strip(),
        max_challengers=_max_challengers_from_environment(),
        daily_cutoff_hour_utc=_daily_cutoff_hour_from_environment(),
        scorer_image_digest=scorer.image_digest,
        scorer_image_reference=str(scorer.reference),
        pool_percent=_pool_percent_from_environment(),
        rewards_enabled=_rewards_enabled_from_environment(),
    )

    # The catalog is organizer-held runtime state. It is not published in a
    # round and the baseline model never becomes an allowlist for miners.
    price_table = broker_module.fetch_openrouter_price_table()

    def key_for(provider: str) -> str:
        """Return one organizer-supplied key by provider name."""

        secret = provider_keys.get(provider)
        if not secret:
            raise broker_module.BrokerError("broker_unavailable")
        return secret

    def broker_factory(service: ArenaService, round_row: Mapping[str, Any]) -> broker_module.Broker:
        judge_models = sorted({str(model) for model in (service.scorer_policy.get("judge_models") or {}).values() if model})
        return broker_module.Broker(
            store=store, key_for=key_for, judge_models=judge_models,
            price_table=price_table, transport=broker_module.HttpxProviderTransport(),
            credential_for=submission_keys.credential_for,
            funding_source_for=submission_keys.funding_source_for,
        )

    def daily_icp_source(*, set_id: int, active_at: datetime) -> Mapping[str, Any]:
        del active_at  # the database function uses its own UTC statement time
        return store.current_daily_icp_set(set_id)

    config = ServiceConfig(
        mode=mode, store=store, object_store=objects, signer=None, chain=chain_reads, verify_signature=chain_module.verify_hotkey_signature,
        daily_icp_source=daily_icp_source,
        banned_hotkeys_source=banned_hotkeys_from_environment, broker_factory=broker_factory, defaults=defaults,
        baseline_source_fetcher=fetch_public_source_archive,
        credential_manager=credential_manager,
        reward_signer_factory=lambda: signing.KmsSigner(_required("LAB_ARENA_SIGNING_KEY_ID"), region_name=os.environ.get("AWS_REGION")),
    )
    service = ArenaService(config)
    app = create_app(service)
    return service, app


def build_runner_from_environment(args):
    from lab_arena import runner as runner_module

    from bittensor_wallet import Wallet

    wallet_arguments = {"name": args.wallet_name, "hotkey": args.hotkey_name}
    wallet_path = str(getattr(args, "wallet_path", "") or "").strip()
    if wallet_path:
        wallet_arguments["path"] = wallet_path
    wallet = Wallet(**wallet_arguments)
    keypair = wallet.hotkey
    runner_root = Path(args.work_dir)
    sandbox_work = runner_root / "sandboxes"
    runs_work = runner_root / "runs"
    for directory in (sandbox_work, runs_work):
        directory.mkdir(parents=True, exist_ok=True)
        if directory.is_symlink() or not directory.is_dir():
            raise runtime.RuntimeHostError("runner work directory is unsafe")
        directory.chmod(0o700)
    config = runtime.RuntimeConfig(runsc_path=Path(args.runsc_path), work_dir=sandbox_work)
    sandbox_runtime = runtime.RunscRuntime(config)
    identity = runner_module.RunnerIdentity(hotkey=keypair.ss58_address, sign=lambda message: keypair.sign(message.encode("utf-8")).hex())
    # Only the common trusted Python/scorer image is materialized. Miner code
    # arrives as a bounded source archive under its active execution lease.
    cache = runner_module.ImageCache(Path(args.work_dir) / "images", runner_module.registry_image_exporter(registry_client_from_environment()))
    api = runner_module.HttpArenaApiClient(args.api_base_url)
    source_cache = runner_module.SourceCache(
        Path(args.work_dir) / "sources", api.source
    )
    round_id = str(getattr(args, "round_id", "") or "").strip() or None
    runner_config = runner_module.RunnerConfig(
        round_id=round_id, identity=identity, api=api, sandbox_runtime=sandbox_runtime, image_cache=cache, source_cache=source_cache,
        work_dir=runs_work, max_parallel_runs=runner_module.max_parallel_runs_from_environment(),
    )
    if round_id is not None:
        api.round(round_id)
        runner_config.evaluation_date = round_id.replace("arena-", "")[:10]
    return runner_module.Runner(runner_config)
