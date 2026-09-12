#!/usr/bin/env python3
"""Run one real Lab Arena testnet401 round with rewards disabled.

By default, durable state uses a named loopback PostgreSQL database. Explicit
managed mode writes one shadow round through the configured Supabase PostgREST
service. Both modes use an explicit S3 prefix and the normal Arena service,
provider broker, KMS credential vault, and chain reader. Neither writes weights.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import threading
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MIGRATIONS = tuple(
    "scripts/%d-lab-arena-%s.sql" % (number, suffix)
    for number, suffix in (
        (179, "v1"),
        (180, "daily-competition"),
        (181, "source-submissions"),
        (182, "source-execution"),
        (183, "miner-reward-basis"),
        (184, "scoring-failure-isolation"),
        (185, "miner-credentials"),
        (187, "promotion-threshold"),
        (188, "baseline-promotion"),
        (189, "round-network-scope"),
        (190, "restart-claim-drain"),
        (193, "upload-recovery"),
        (194, "open-scorer-refresh"),
    )
) + (
    "scripts/197-lab-arena-reward-chain-scope.sql",
    "scripts/199-lab-arena-source-disclosure-time.sql",
    "scripts/200-lab-arena-next-day-icp-disclosure.sql",
    "scripts/201-lab-arena-daily-capacity.sql",
    "scripts/202-arena-accepted-weight-state.sql",
    "scripts/203-retire-legacy-incentive-weight-bridge.sql",
    "scripts/205-lab-arena-optional-scrapingdog-credential.sql",
    "scripts/206-lab-arena-combined-provider-budget.sql",
    "scripts/207-lab-arena-code-review.sql",
    "scripts/208-lab-arena-validator-scoring-authority.sql",
    "scripts/209-lab-arena-uncertain-cost-eligibility.sql",
    "scripts/211-lab-arena-owner-admission.sql",
    "scripts/212-lab-arena-accepted-judgment-cache.sql",
    "scripts/213-lab-arena-score-integrity.sql",
    "scripts/214-lab-arena-prior-credential-refusal.sql",
    "scripts/215-lab-arena-contacts.sql",
    "scripts/216-lab-arena-validator-participation.sql",
    "scripts/217-lab-arena-company-judgments.sql",
    "scripts/218-lab-arena-cross-provider-credential-refusal.sql",
    "scripts/223-lab-arena-cancelled-call-late-settlement.sql",
    "scripts/225-lab-arena-openrouter-delayed-cost-reconciliation.sql",
    "scripts/227-lab-arena-champion-funding.sql",
)
EXPECTED_SCHEMA_VERSION = 197
TESTNET_NETUID = 401
TESTNET_NETWORK = "test"
DEFAULT_DATABASE = "miner_testnet"
DEFAULT_DATABASE_PORT = 55432
DEFAULT_AWS_ACCOUNT = "493765492819"
DEFAULT_BUCKET = "leadpoet-attested-v2-artifacts-493765492819"
DEFAULT_RUNNER = "5GsGcRyR4kWCcsa1qEAwxtbDq34ZwkQt3rHAGniPFjv1JoXW"
DEFAULT_BASELINE = "5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"
DEFAULT_MINER = "5FEtvBzsh5Zc8nDyq4Jb2nZ7o6ZD2homYsKjbZtFj5tybqth"
ORGANIZER_KEY_ALIASES = {
    "LAB_ARENA_OPENROUTER_API_KEY": "OPENROUTER_API_KEY",
    "LAB_ARENA_DEEPLINE_API_KEY": "DEEPLINE_API_KEY",
    "LAB_ARENA_SCRAPINGDOG_API_KEY": "SCRAPINGDOG_API_KEY",
}
REQUIRED_SEED_KEYS = ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY")
_ROUND_RE = re.compile(r"^arena-[0-9]{4}-[0-9]{2}-[0-9]{2}-[a-z0-9]{1,16}$")
_PREFIX_RE = re.compile(r"^miner-testnet-[a-z0-9][a-z0-9_-]{5,63}$")
_DIGEST_REFERENCE_RE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")


class ConfigurationError(RuntimeError):
    """A safe startup failure whose text contains no secret material."""


class _RejectRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        del request, file_pointer, code, message, headers, new_url
        return None


_DIRECT_URLOPEN = urllib.request.build_opener(
    urllib.request.ProxyHandler({}), _RejectRedirect()
).open


def _json_output(**values: Any) -> None:
    print(json.dumps(values, sort_keys=True, separators=(",", ":"), default=str))


def _parse_environment_document(raw: str) -> dict[str, str]:
    """Parse the JSON or simple KEY=VALUE form used by the gateway secret."""

    try:
        document = json.loads(raw)
    except ValueError:
        document = None
    if isinstance(document, Mapping):
        rows = document.items()
    else:
        try:
            from dotenv import dotenv_values

            rows = dotenv_values(stream=io.StringIO(raw.replace("\x00", "\n"))).items()
        except Exception:
            raise ConfigurationError("gateway secret environment document is invalid") from None
    values: dict[str, str] = {}
    for raw_key, raw_value in rows:
        key = str(raw_key or "").strip()
        if not key:
            continue
        if isinstance(raw_value, (dict, list)):
            value = json.dumps(raw_value, sort_keys=True, separators=(",", ":"))
        else:
            value = "" if raw_value is None else str(raw_value)
        if key in values and values[key] != value:
            raise ConfigurationError("gateway secret contains a conflicting duplicate")
        values[key] = value
    if not values:
        raise ConfigurationError("gateway secret is empty")
    return values


def _load_gateway_secret(secret_id: str, region: str) -> dict[str, str]:
    import boto3

    response = boto3.client("secretsmanager", region_name=region).get_secret_value(
        SecretId=secret_id
    )
    if response.get("SecretString") is not None:
        raw = str(response["SecretString"])
    else:
        try:
            raw = base64.b64decode(response["SecretBinary"], validate=True).decode("utf-8")
        except Exception:
            raise ConfigurationError("gateway secret payload is invalid") from None
    return _parse_environment_document(raw)


def _require_secret_names(secret: Mapping[str, str], names: tuple[str, ...]) -> None:
    missing = [name for name in names if not str(secret.get(name) or "").strip()]
    if missing:
        raise ConfigurationError("gateway secret lacks required keys: %s" % ",".join(missing))


def _secret_alias(secret: Mapping[str, str], name: str) -> str:
    """Read one Arena key, accepting the pre-Arena gateway alias."""

    values = [str(secret.get(candidate) or "").strip() for candidate in (name, ORGANIZER_KEY_ALIASES.get(name, ""))]
    values = [value for value in values if value]
    if not values:
        raise ConfigurationError("gateway secret lacks required key: %s" % name)
    if len(set(values)) != 1:
        raise ConfigurationError("gateway secret has conflicting aliases for: %s" % name)
    return values[0]


def _database_parameters(
    dsn: str, *, expected_database: str, expected_port: int
) -> dict[str, str]:
    try:
        from psycopg2.extensions import parse_dsn

        parsed = {key: str(value) for key, value in parse_dsn(dsn).items()}
    except Exception:
        raise ConfigurationError("LAB_ARENA_TESTNET_DATABASE_DSN is invalid") from None
    allowed = {"host", "port", "dbname", "user", "connect_timeout", "sslmode"}
    if set(parsed) - allowed:
        raise ConfigurationError("database DSN contains unsupported connection options")
    if parsed.get("host") not in ("127.0.0.1", "localhost", "::1"):
        raise ConfigurationError("database must use an explicit loopback host")
    if parsed.get("dbname") != expected_database:
        raise ConfigurationError("database name does not match the isolated target")
    if parsed.get("user") != "postgres":
        raise ConfigurationError("database user must be postgres")
    try:
        port = int(parsed.get("port", "0"))
    except ValueError:
        port = 0
    if port != int(expected_port):
        raise ConfigurationError("database port does not match the isolated target")
    parsed.setdefault("connect_timeout", "10")
    return parsed


def _database_connect(args: argparse.Namespace):
    dsn = os.environ.get("LAB_ARENA_TESTNET_DATABASE_DSN", "").strip()
    if not dsn:
        raise ConfigurationError("LAB_ARENA_TESTNET_DATABASE_DSN is required")
    parameters = _database_parameters(
        dsn,
        expected_database=args.expected_database,
        expected_port=args.expected_db_port,
    )
    import psycopg2

    return lambda: psycopg2.connect(**parameters)


def _managed_postgrest_transport(args: argparse.Namespace):
    """Build the Arena transport from the protected LAB_ARENA environment.

    The managed path deliberately does not accept a DSN or a database name.
    Its only mutable target is the configured HTTPS PostgREST origin, and the
    service credential stays in the process environment.
    """

    if args.arena_environment_file is not None:
        from scripts.run_lab_arena_service import load_scoped_environment

        load_scoped_environment(args.arena_environment_file)
    from lab_arena.store import PostgrestTransport

    url = os.environ.get("LAB_ARENA_SUPABASE_URL", "").strip()
    anon_key = os.environ.get("LAB_ARENA_SUPABASE_ANON_KEY", "").strip()
    service_key = os.environ.get("LAB_ARENA_SERVICE_KEY", "").strip()
    service_jwt = os.environ.get("LAB_ARENA_SERVICE_JWT", "").strip()
    if not url or not anon_key or not (service_key or service_jwt):
        raise ConfigurationError(
            "managed PostgREST requires LAB_ARENA_SUPABASE_URL, "
            "LAB_ARENA_SUPABASE_ANON_KEY, and one Arena service credential"
        )
    try:
        return PostgrestTransport(
            url,
            anon_key=anon_key,
            service_key=service_key,
            service_jwt="" if service_key else service_jwt,
        )
    except Exception as exc:
        raise ConfigurationError("managed PostgREST transport is invalid") from exc


def _validate_s3_prefix(prefix: str) -> str:
    if not _PREFIX_RE.fullmatch(prefix):
        raise ConfigurationError(
            "S3 prefix must be one unique miner-testnet-<run> component"
        )
    return prefix


def _assert_aws_account(region: str, expected_account: str) -> None:
    import boto3

    actual = str(boto3.client("sts", region_name=region).get_caller_identity().get("Account") or "")
    if actual != expected_account:
        raise ConfigurationError("AWS caller account does not match the isolated target")


def _assert_kms_key(kms_key_id: str, *, region: str, account: str):
    expected_prefix = "arn:aws:kms:%s:%s:key/" % (region, account)
    if not kms_key_id.startswith(expected_prefix):
        raise ConfigurationError("KMS key ARN does not match the expected region and account")
    import boto3

    client = boto3.client("kms", region_name=region)
    metadata = client.describe_key(KeyId=kms_key_id).get("KeyMetadata") or {}
    if metadata.get("Arn") != kms_key_id or metadata.get("Enabled") is not True:
        raise ConfigurationError("KMS key is not the requested enabled key")
    return client


def _setup_database(args: argparse.Namespace) -> int:
    connect = _database_connect(args)
    connection = connect()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_database(), current_user")
            database, user = cursor.fetchone()
            if database != args.expected_database or user != "postgres":
                raise ConfigurationError("connected database identity does not match the isolated target")
            cursor.execute(
                "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname='public' "
                "AND tablename <> 'qualification_private_icp_sets' "
                "AND tablename NOT LIKE 'lab_arena_%' LIMIT 1"
            )
            if cursor.fetchone() is not None:
                raise ConfigurationError("isolated database contains unrelated public tables")
            cursor.execute(
                """
                CREATE SCHEMA IF NOT EXISTS extensions;
                CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA extensions;
                DO $roles$
                BEGIN
                  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname='anon') THEN
                    CREATE ROLE anon NOLOGIN;
                  END IF;
                  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname='authenticated') THEN
                    CREATE ROLE authenticated NOLOGIN;
                  END IF;
                  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname='service_role') THEN
                    CREATE ROLE service_role NOLOGIN;
                  END IF;
                END
                $roles$;
                CREATE TABLE IF NOT EXISTS public.qualification_private_icp_sets (
                  set_id BIGINT PRIMARY KEY,
                  icps JSONB NOT NULL,
                  active_from TIMESTAMPTZ,
                  active_until TIMESTAMPTZ,
                  is_active BOOLEAN NOT NULL DEFAULT FALSE
                );
                ALTER TABLE public.qualification_private_icp_sets ENABLE ROW LEVEL SECURITY;
                REVOKE ALL ON TABLE public.qualification_private_icp_sets
                  FROM PUBLIC, anon, authenticated;
                """
            )
            for relative_path in MIGRATIONS:
                cursor.execute((ROOT / relative_path).read_text(encoding="utf-8"))
    finally:
        connection.close()

    from lab_arena.store import ArenaStore, PsycopgTransport

    transport = PsycopgTransport(connect)
    try:
        store = ArenaStore(transport)
        identity = store.require_service_role()
        schema = transport.rpc("lab_arena_schema_version_v1", {})
        if not isinstance(schema, Mapping) or schema.get("version") != EXPECTED_SCHEMA_VERSION:
            raise ConfigurationError("Arena schema version verification failed")
    finally:
        transport.close()
    _json_output(
        command="setup-db",
        database=args.expected_database,
        port=args.expected_db_port,
        role=identity["current_user"],
        schema_version=EXPECTED_SCHEMA_VERSION,
    )
    return 0


def _validate_supabase_url(value: str) -> str:
    parts = urllib.parse.urlsplit(value.strip())
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
        or parts.port not in (None, 443)
    ):
        raise ConfigurationError("SUPABASE_URL is not a plain HTTPS origin")
    return value.rstrip("/")


def _current_icp_row(secret: Mapping[str, str], set_id: int) -> dict[str, Any]:
    base_url = _validate_supabase_url(secret["SUPABASE_URL"])
    query = urllib.parse.urlencode(
        {
            "select": "set_id,icps,active_from,active_until,is_active",
            "set_id": "eq.%d" % set_id,
            "is_active": "eq.true",
            "limit": "1",
        }
    )
    request = urllib.request.Request(
        "%s/rest/v1/qualification_private_icp_sets?%s" % (base_url, query),
        headers={
            "Accept": "application/json",
            "apikey": secret["SUPABASE_SERVICE_ROLE_KEY"],
            "Authorization": "Bearer " + secret["SUPABASE_SERVICE_ROLE_KEY"],
        },
        method="GET",
    )
    try:
        with _DIRECT_URLOPEN(request, timeout=30) as response:
            raw = response.read(8 * 1024 * 1024 + 1)
    except Exception:
        raise ConfigurationError("production ICP read failed") from None
    if len(raw) > 8 * 1024 * 1024:
        raise ConfigurationError("production ICP response exceeds the size cap")
    try:
        rows = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise ConfigurationError("production ICP response is invalid") from None
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise ConfigurationError("production current ICP set is not uniquely available")
    row = dict(rows[0])
    icps = row.get("icps")
    if row.get("set_id") != set_id or row.get("is_active") is not True or not isinstance(icps, list) or len(icps) != 20:
        raise ConfigurationError("production current ICP set fails the Arena contract")
    icp_ids = [str(item.get("icp_id") or "").strip() if isinstance(item, Mapping) else "" for item in icps]
    if any(not value for value in icp_ids) or len(set(icp_ids)) != 20:
        raise ConfigurationError("production current ICP identities fail the Arena contract")
    return row


def _seed_current(args: argparse.Namespace) -> int:
    _assert_aws_account(args.aws_region, args.expected_aws_account)
    secret = _load_gateway_secret(args.gateway_secret_id, args.aws_region)
    _require_secret_names(secret, REQUIRED_SEED_KEYS)
    set_id = int(datetime.now(timezone.utc).strftime("%Y%m%d"))
    row = _current_icp_row(secret, set_id)
    connect = _database_connect(args)
    connection = connect()
    connection.autocommit = False
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT icps FROM public.qualification_private_icp_sets WHERE set_id=%s FOR UPDATE",
                (set_id,),
            )
            existing = cursor.fetchone()
            if existing is not None and existing[0] != row["icps"]:
                raise ConfigurationError("isolated database already has different ICPs for this date")
            if existing is None:
                cursor.execute(
                    "INSERT INTO public.qualification_private_icp_sets "
                    "(set_id,icps,active_from,active_until,is_active) "
                    "VALUES (%s,%s::jsonb,%s,%s,TRUE)",
                    (set_id, json.dumps(row["icps"]), row.get("active_from"), row.get("active_until")),
                )
            else:
                cursor.execute(
                    "UPDATE public.qualification_private_icp_sets SET active_from=%s,active_until=%s,is_active=TRUE "
                    "WHERE set_id=%s",
                    (row.get("active_from"), row.get("active_until"), set_id),
                )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    _json_output(command="seed-current", set_id=set_id, icp_count=20, destination=args.expected_database)
    return 0


def _cutoff_and_round(args: argparse.Namespace) -> tuple[datetime, str]:
    try:
        cutoff = datetime.strptime(args.cutoff, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        raise ConfigurationError("cutoff must use YYYY-MM-DDTHH:MM:SSZ") from None
    now = datetime.now(timezone.utc)
    if not args.resume_round and cutoff.date() != now.date():
        raise ConfigurationError("cutoff must stay on the current UTC ICP date")
    if not args.resume_round and not 60 <= (cutoff - now).total_seconds() <= 3600:
        raise ConfigurationError("cutoff must be 1 to 60 minutes in the future")
    round_id = args.round_id or "arena-%s-e2e" % cutoff.strftime("%Y-%m-%d")
    if not _ROUND_RE.fullmatch(round_id) or round_id[6:16] != cutoff.strftime("%Y-%m-%d"):
        raise ConfigurationError("round id must match the cutoff date and include an isolated suffix")
    return cutoff, round_id


def _advance_pinned(service: Any, round_id: str) -> str:
    """Advance exactly one round without daily-round creation or rewards."""

    try:
        result = service.advance_round(round_id)
    except Exception as exc:
        return "failed advance_round %s: %s" % (round_id, type(exc).__name__)
    if isinstance(result, Mapping):
        status = str(result.get("status") or "unknown")
        return "advanced %s:%s" % (round_id, status)
    return "advanced %s" % round_id


def _validate_chain_endpoint(endpoint: str) -> str:
    parts = urllib.parse.urlsplit(endpoint)
    if (
        parts.scheme != "wss"
        or parts.hostname != "test.finney.opentensor.ai"
        or parts.port not in (None, 443)
        or parts.username
        or parts.password
        or parts.path not in ("", "/")
        or parts.query
        or parts.fragment
    ):
        raise ConfigurationError("chain endpoint must be test.finney.opentensor.ai over WSS port 443")
    return endpoint


def _selected_baseline_source_url(configured: str | None, default: str) -> str:
    """Select an explicit shadow baseline or the production service default."""

    return str(configured).strip() if configured is not None else default


def _resume_configuration_matches(
    configuration: Mapping[str, Any],
    *,
    cutoff: str,
    runner_hotkey: str,
    baseline_hotkey: str,
    baseline_source_url: str,
    scorer_image_reference: str,
) -> bool:
    """Require a resumed shadow test to use its exact frozen controls."""

    return (
        configuration.get("mode") == "shadow"
        and configuration.get("rewards_enabled") is False
        and (configuration.get("schedule") or {}).get("submission_cutoff") == cutoff
        and configuration.get("runner_hotkeys") == [runner_hotkey]
        and configuration.get("baseline_hotkey") == baseline_hotkey
        and configuration.get("baseline_source_url") == baseline_source_url
        and configuration.get("scorer_image_reference") == scorer_image_reference
    )


def _serve(args: argparse.Namespace) -> int:
    prefix = _validate_s3_prefix(args.s3_prefix)
    if args.bucket != DEFAULT_BUCKET:
        raise ConfigurationError("bucket does not match the approved artifact bucket")
    if not _DIGEST_REFERENCE_RE.fullmatch(args.scorer_image):
        raise ConfigurationError("scorer image must be an explicit sha256 digest reference")
    if len({args.runner_hotkey, args.miner_hotkey, args.baseline_hotkey}) != 3:
        raise ConfigurationError("runner, miner, and baseline hotkeys must be distinct")
    _validate_chain_endpoint(args.chain_endpoint)
    if args.managed_postgrest and args.port == 8792:
        args.port = 8794
    if args.resume_round and not args.round_id:
        raise ConfigurationError("--resume-round requires the exact --round-id")
    cutoff, round_id = _cutoff_and_round(args)
    _assert_aws_account(args.aws_region, args.expected_aws_account)
    secret = _load_gateway_secret(args.gateway_secret_id, args.aws_region)
    provider_keys = {
        "openrouter": _secret_alias(secret, "LAB_ARENA_OPENROUTER_API_KEY"),
        "deepline": _secret_alias(secret, "LAB_ARENA_DEEPLINE_API_KEY"),
        "scrapingdog": _secret_alias(secret, "LAB_ARENA_SCRAPINGDOG_API_KEY"),
    }
    kms_client = _assert_kms_key(
        args.kms_key_id, region=args.aws_region, account=args.expected_aws_account
    )

    import boto3
    from lab_arena import broker as broker_module
    from lab_arena import chain as chain_module
    from lab_arena import confirmation, contracts, images, scorer_image_access
    from lab_arena.api import create_app
    from lab_arena.credentials import CredentialManager
    from lab_arena.code_review_runtime import SubmissionCodeReviewer
    from lab_arena.driver import drive_once
    from lab_arena.service import (
        DEFAULT_BASELINE_SOURCE_URL,
        ArenaService,
        RoundDefaults,
        S3ObjectStore,
        ServiceConfig,
    )
    from lab_arena.store import ArenaStore, PsycopgTransport
    from lab_arena.submission_runtime import SubmissionProviderKeys
    from lab_arena.wiring import ChainReadsAdapter, fetch_public_source_archive, registry_client_from_environment

    if args.managed_postgrest:
        transport = _managed_postgrest_transport(args)
    else:
        connect = _database_connect(args)
        transport = PsycopgTransport(connect)
    store = ArenaStore(transport)
    chain_config = chain_module.ArenaChainConfig(
        endpoint=args.chain_endpoint,
        netuid=TESTNET_NETUID,
        network_name=TESTNET_NETWORK,
        request_timeout_seconds=30,
    )
    arena_chain = chain_module.ArenaChain(chain_config, chain_module.connect_substrate(chain_config))
    review_transport = None
    try:
        snapshot = arena_chain.metagraph()
        miner_uid = chain_module.uid_for_hotkey(snapshot, args.miner_hotkey)
        runner_uid = chain_module.uid_for_hotkey(snapshot, args.runner_hotkey)
        if miner_uid != args.expected_miner_uid:
            raise ConfigurationError("testnet miner UID does not match the expected admission identity")
        if runner_uid is not None and snapshot.coldkeys[miner_uid] == snapshot.coldkeys[runner_uid]:
            raise ConfigurationError("runner and miner must be owned by different coldkeys")

        image_reference = images.parse_reference(args.scorer_image)
        os.environ["LAB_ARENA_REGISTRY_REPOSITORY"] = "%s/%s" % (
            image_reference.registry,
            image_reference.repository,
        )
        registry = registry_client_from_environment()
        try:
            scorer = images.resolve_image(registry, image_reference, images.ImageRules())
        finally:
            registry.close()
        if str(scorer.reference) != args.scorer_image:
            raise ConfigurationError("resolved scorer image does not match the pinned reference")

        objects = S3ObjectStore(
            args.bucket,
            region_name=args.aws_region,
            prefix=prefix,
        )
        chain_reads = ChainReadsAdapter(arena_chain)
        credential_manager = CredentialManager(kms_key_id=args.kms_key_id, kms_client=kms_client)
        submission_keys = SubmissionProviderKeys(
            store=store, credentials=credential_manager, organizer_keys=provider_keys
        )
        price_table = broker_module.fetch_openrouter_price_table()
        review_transport = broker_module.HttpxProviderTransport()

        def broker_factory(service: ArenaService, round_row: Mapping[str, Any]):
            del round_row
            judge_models = sorted(
                {
                    str(model)
                    for model in (service.scorer_policy.get("judge_models") or {}).values()
                    if model
                }
            )
            return broker_module.Broker(
                store=store,
                key_for=lambda provider: provider_keys[provider],
                judge_models=judge_models,
                price_table=price_table,
                transport=broker_module.HttpxProviderTransport(),
                credential_for=submission_keys.credential_for,
                funding_source_for=submission_keys.funding_source_for,
            )

        baseline_source_url = _selected_baseline_source_url(
            args.baseline_source_url,
            DEFAULT_BASELINE_SOURCE_URL,
        )
        defaults = RoundDefaults(
            execution_cap_microusd=args.execution_cap_usd,
            scoring_cap_microusd=args.scoring_cap_usd,
            runner_hotkeys=(args.runner_hotkey,),
            baseline_hotkey=args.baseline_hotkey,
            baseline_source_url=baseline_source_url,
            stage_minutes={
                "benchmark": args.benchmark_minutes,
                "stage_1": args.stage_1_minutes,
                "stage_1_scoring": args.stage_1_scoring_minutes,
                "stage_2": args.stage_2_minutes,
                "final_scoring": args.final_scoring_minutes,
            },
            max_challengers=1,
            scorer_image_digest=scorer.image_digest,
            scorer_image_reference=str(scorer.reference),
            daily_cutoff_hour_utc=None,
            rewards_enabled=False,
            integrity_from=os.environ.get("LAB_ARENA_INTEGRITY_FROM") or None,
            contacts_from=os.environ.get("LAB_ARENA_CONTACTS_FROM") or None,
            company_quality_from=os.environ.get("LAB_ARENA_COMPANY_QUALITY_FROM") or None,
            benchmark_disclosure_from=os.environ.get("LAB_ARENA_BENCHMARK_DISCLOSURE_FROM") or None,
        )
        service = ArenaService(
            ServiceConfig(
                mode="shadow",
                store=store,
                object_store=objects,
                signer=None,
                chain=chain_reads,
                verify_signature=chain_module.verify_hotkey_signature,
                daily_icp_source=lambda *, set_id, active_at: store.current_daily_icp_set(set_id),
                confirmation_icp_source=lambda **kwargs: confirmation.fresh_confirmation_icps(
                    **kwargs, api_key=provider_keys["openrouter"],
                ),
                banned_hotkeys_source=lambda: (),
                broker_factory=broker_factory,
                defaults=defaults,
                network_name=TESTNET_NETWORK,
                netuid=TESTNET_NETUID,
                pinned_round_id=round_id,
                scorer_image_access=scorer_image_access.ecr_provider_from_repository(
                    os.environ["LAB_ARENA_REGISTRY_REPOSITORY"],
                    rules=images.ImageRules(),
                ),
                baseline_source_fetcher=fetch_public_source_archive,
                reward_signer_factory=None,
                credential_manager=credential_manager,
                code_reviewer=SubmissionCodeReviewer(
                    store=store,
                    objects=objects,
                    credential_for=submission_keys.code_review_key,
                    price_table=price_table,
                    transport=review_transport,
                ),
            )
        )
        checks = service.startup_checks()
        if checks.get("schema_version") != EXPECTED_SCHEMA_VERSION:
            raise ConfigurationError("Arena startup did not verify the current schema")
        if args.resume_round:
            if args.managed_postgrest:
                existing_round = store.get_round(round_id)
                if existing_round is None:
                    raise ConfigurationError("resume target does not exist")
            else:
                existing_rounds = store.list_rounds(limit=2)
                if len(existing_rounds) != 1 or existing_rounds[0].get("round_id") != round_id:
                    raise ConfigurationError("resume target is not the only round in the isolated database")
                existing_round = existing_rounds[0]
            existing_configuration = existing_round.get("configuration_doc") or {}
            if not _resume_configuration_matches(
                existing_configuration,
                cutoff=args.cutoff,
                runner_hotkey=args.runner_hotkey,
                baseline_hotkey=args.baseline_hotkey,
                baseline_source_url=baseline_source_url,
                scorer_image_reference=str(scorer.reference),
            ):
                raise ConfigurationError("existing round configuration does not match the resume request")
        elif args.managed_postgrest:
            if store.get_round(round_id) is not None:
                raise ConfigurationError("managed target round already exists")
            service.create_round(cutoff, round_id=round_id)
        else:
            existing_rounds = store.list_rounds(limit=2)
            if existing_rounds:
                raise ConfigurationError("isolated database already contains an Arena round")
            service.create_round(cutoff, round_id=round_id)
        initial = _advance_pinned(service, round_id) if args.managed_postgrest else drive_once(service)
        if "failed" in initial:
            raise ConfigurationError("initial Arena driver tick failed")
        _json_output(
            command="serve",
            bind="127.0.0.1",
            port=args.port,
            round_id=round_id,
            cutoff=args.cutoff,
            mode="shadow",
            rewards_enabled=False,
            netuid=TESTNET_NETUID,
            miner_uid=miner_uid,
            s3_prefix=prefix,
            schema_version=EXPECTED_SCHEMA_VERSION,
        )

        stop = threading.Event()

        def driver() -> None:
            while not stop.wait(max(5, args.tick_seconds)):
                outcome = (
                    _advance_pinned(service, round_id)
                    if args.managed_postgrest
                    else drive_once(service)
                )
                if "failed" in outcome:
                    print("Arena driver tick failed", file=sys.stderr)

        def review_submissions() -> None:
            while not stop.is_set():
                try:
                    service.review_pending_submissions()
                except Exception as exc:
                    # Keep source, provider responses, and credentials out of logs.
                    print(
                        "Arena code review worker failed: %s" % type(exc).__name__,
                        file=sys.stderr,
                    )
                stop.wait(5)

        driver_thread = threading.Thread(target=driver, name="testnet-arena-driver", daemon=True)
        review_thread = threading.Thread(
            target=review_submissions,
            name="testnet-arena-code-review",
            daemon=True,
        )
        driver_thread.start()
        review_thread.start()
        import uvicorn

        try:
            uvicorn.run(create_app(service), host="127.0.0.1", port=args.port, log_level="info")
        finally:
            stop.set()
            driver_thread.join(timeout=5)
            review_thread.join(timeout=5)
    finally:
        if review_transport is not None:
            review_transport.close()
        arena_chain.close()
        transport.close()
    return 0


def _positive_minutes(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 240:
        raise argparse.ArgumentTypeError("stage minutes must be between 1 and 240")
    return parsed


def _usd_cap(value: str) -> int:
    try:
        microusd = int(Decimal(value) * 1_000_000)
    except (InvalidOperation, ValueError):
        raise argparse.ArgumentTypeError("USD cap must be a decimal number") from None
    if not 100_000 <= microusd <= 100_000_000:
        raise argparse.ArgumentTypeError("USD cap must be between 0.10 and 100")
    return microusd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(
        expected_database=DEFAULT_DATABASE,
        expected_db_port=DEFAULT_DATABASE_PORT,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup-db", help="apply the isolated schema shim and current Arena migrations")

    def add_aws(command):
        command.add_argument("--aws-region", default="us-east-1")
        command.add_argument("--expected-aws-account", default=DEFAULT_AWS_ACCOUNT)
        command.add_argument("--gateway-secret-id", required=True)

    seed = commands.add_parser("seed-current", help="copy today's real active ICP set into isolated PostgreSQL")
    add_aws(seed)

    serve = commands.add_parser("serve", help="create and drive one isolated shadow testnet401 round")
    add_aws(serve)
    serve.add_argument(
        "--managed-postgrest",
        action="store_true",
        help="use the configured managed Supabase PostgREST target",
    )
    serve.add_argument(
        "--arena-environment-file",
        type=Path,
        help="load only LAB_ARENA_* values for managed PostgREST",
    )
    serve.add_argument("--chain-endpoint", required=True)
    serve.add_argument("--cutoff", required=True, help="current-day UTC instant, 1 to 60 minutes ahead")
    serve.add_argument("--round-id")
    serve.add_argument("--resume-round", action="store_true")
    serve.add_argument("--kms-key-id", required=True)
    serve.add_argument("--bucket", default=DEFAULT_BUCKET)
    serve.add_argument("--s3-prefix", required=True)
    serve.add_argument("--scorer-image", required=True, help="registry/repository@sha256:<digest>")
    serve.add_argument("--miner-hotkey", default=DEFAULT_MINER)
    serve.add_argument("--expected-miner-uid", type=int, default=11)
    serve.add_argument("--runner-hotkey", default=DEFAULT_RUNNER)
    serve.add_argument("--baseline-hotkey", default=DEFAULT_BASELINE)
    serve.add_argument(
        "--baseline-source-url",
        help="public HTTPS source archive used only as this shadow round's baseline",
    )
    serve.add_argument("--benchmark-minutes", type=_positive_minutes, default=2)
    serve.add_argument("--stage-1-minutes", type=_positive_minutes, default=30)
    serve.add_argument("--stage-1-scoring-minutes", type=_positive_minutes, default=60)
    serve.add_argument("--stage-2-minutes", type=_positive_minutes, default=30)
    serve.add_argument("--final-scoring-minutes", type=_positive_minutes, default=60)
    serve.add_argument("--execution-cap-usd", type=_usd_cap, default="5")
    serve.add_argument("--scoring-cap-usd", type=_usd_cap, default="10")
    serve.add_argument("--tick-seconds", type=int, default=15)
    serve.add_argument("--port", type=int, default=8792)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "setup-db":
            return _setup_database(args)
        if args.command == "seed-current":
            return _seed_current(args)
        if args.command == "serve":
            return _serve(args)
    except ConfigurationError as exc:
        print("refused: %s" % exc, file=sys.stderr)
        return 2
    except Exception as exc:
        print("refused: %s" % type(exc).__name__, file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
