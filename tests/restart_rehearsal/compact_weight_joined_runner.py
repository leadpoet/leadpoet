#!/usr/bin/env python3.11
"""Bounded joined rehearsal for the activated compact V2 weight path.

The fixture owns only sanitized inputs and strict implementations of the
database, Nitro, gateway-network, SDK, and chain boundaries.  Bundle
construction, compact ancestry, gateway publication/finalization, the primary
validator lifecycle, auditor verification/cache, and journal recovery all run
through candidate production functions.
"""

from __future__ import annotations

import asyncio
import base64
import copy
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

from bittensor_wallet import Keypair
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import httpx
from fastapi import HTTPException
from starlette.requests import Request

from Leadpoet.utils.subnet_epoch import SubnetEpochCutover, SubnetEpochSnapshot
from leadpoet_canonical.ancestry_checkpoint_v2 import (
    ANCESTRY_DELTA_SCHEMA_VERSION,
    build_compact_ancestry_proof_from_delta_v2,
    derive_ancestry_lineage_id_v2,
    issue_ancestry_certificate_v2,
)
from leadpoet_canonical.attested_v2 import (
    EMPTY_ARTIFACT_ROOT,
    EMPTY_HOST_OPERATION_ROOT,
    EMPTY_TRANSPORT_ROOT,
    build_receipt_graph,
    build_transport_attempt,
    canonical_json,
    merkle_root,
    sha256_bytes,
    sha256_json,
)
from leadpoet_canonical.hotkey_authority_v2 import (
    build_weight_extrinsic_authorization_v2,
    encode_signed_extrinsic_v2,
    signed_extrinsic_hash_v2,
)
from leadpoet_canonical.weight_authority_v2 import (
    GATEWAY_WEIGHT_INPUT_CATEGORIES,
    WEIGHT_INPUT_PURPOSES,
    weight_input_output_roots_v2,
    weight_input_value_documents_v2,
)
from leadpoet_canonical.weight_computation import (
    compute_final_weights,
    weight_config_hash,
)
from sanitized_weight_fixture import NOW, SanitizedWeightFixture


NETUID = 71
EXPECTED_CHAIN = "wss://entrypoint-finney.opentensor.ai:443"
GENESIS_HASH = (
    "0x2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03"
)
EPOCH_ID = 30_000


def _source_root() -> Path:
    configured = str(os.getenv("REHEARSAL_SOURCE_ROOT") or "/source").strip()
    root = Path(configured).resolve()
    required_paths = (
        root / "gateway/api/weights.py",
        root / "validator_tee/enclave/chain_signing_profile_v2.json",
    )
    if not root.is_dir() or any(not path.is_file() for path in required_paths):
        raise RuntimeError(
            "compact rehearsal source root is not a candidate source tree"
        )
    return root


def _candidate_sha() -> str:
    configured = str(os.getenv("REHEARSAL_CANDIDATE_SHA") or "").strip().lower()
    if configured:
        if len(configured) != 40 or any(
            character not in "0123456789abcdef" for character in configured
        ):
            raise RuntimeError("compact rehearsal candidate SHA is invalid")
        return configured
    value = subprocess.run(
        ["git", "-C", str(_source_root()), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().lower()
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RuntimeError("compact rehearsal candidate SHA is invalid")
    return value


@contextmanager
def _patched(target: Any, name: str, value: Any):
    original = getattr(target, name)
    setattr(target, name, value)
    try:
        yield
    finally:
        setattr(target, name, original)


@contextmanager
def _environment(values: Mapping[str, str]):
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update({name: str(value) for name, value in values.items()})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def _owned_event_loop():
    """Keep import-time asyncio primitives on the scenario's execution loop."""

    try:
        previous = asyncio.get_event_loop()
    except RuntimeError:
        previous = None
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        asyncio.set_event_loop(
            previous if previous is not None and not previous.is_closed() else None
        )
        loop.close()


@contextmanager
def _sdk_weight_module_boundary():
    """Supply only the missing external SDK symbol on older local hosts."""

    from validator_tee.host.enclave_hotkey_v2 import (
        EnclaveHotkeyV2Error,
        _weight_extrinsic_module,
    )

    try:
        _weight_extrinsic_module()
    except EnclaveHotkeyV2Error:
        import bittensor.core.extrinsics as extrinsics_package

        module_name = "bittensor.core.extrinsics.weights"
        sentinel = object()
        previous_module = sys.modules.get(module_name, sentinel)
        previous_attribute = getattr(extrinsics_package, "weights", sentinel)

        def unexpected_external_sdk_call(**_kwargs: Any):
            raise RuntimeError(
                "unpatched external SDK weight helper was invoked"
            )

        module = SimpleNamespace(
            get_encrypted_commit_v2=unexpected_external_sdk_call
        )
        sys.modules[module_name] = module
        setattr(extrinsics_package, "weights", module)
        try:
            yield {"adapted": True, "module": module_name}
        finally:
            if previous_module is sentinel:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous_module
            if previous_attribute is sentinel:
                delattr(extrinsics_package, "weights")
            else:
                setattr(extrinsics_package, "weights", previous_attribute)
    else:
        yield {"adapted": False, "module": "installed"}


class _MemoryPostgREST:
    """Exact append/read adapter used by production compact store helpers."""

    def __init__(self) -> None:
        self.tables: dict[str, list[dict[str, Any]]] = {}
        self.insert_count = 0
        self.checkpoint_rpc_writes: dict[str, int] = {}
        self.unknown_commit_root: str | None = None
        self.unknown_commit_pending: dict[str, dict[str, Any]] | None = None
        self.unknown_commit_readbacks = 0
        self.unknown_commit_sleep_delays: list[float] = []
        self.unknown_commit_visible = False

    @staticmethod
    def _matches(row: Mapping[str, Any], filters: Iterable[tuple]) -> bool:
        for item in filters:
            if len(item) == 2:
                field, expected = item
                if row.get(field) != expected:
                    return False
            elif len(item) == 3:
                field, operator, expected = item
                if operator == "in":
                    if row.get(field) not in expected:
                        return False
                else:
                    raise RuntimeError(
                        "compact rehearsal received an unsupported store filter"
                    )
            else:
                raise RuntimeError("compact rehearsal store filter is malformed")
        return True

    async def insert_row(self, table: str, row: Mapping[str, Any]) -> dict[str, Any]:
        normalized = copy.deepcopy(dict(row))
        rows = self.tables.setdefault(str(table), [])
        if normalized in rows:
            raise RuntimeError("23505 duplicate key unique constraint")
        rows.append(normalized)
        self.insert_count += 1
        return copy.deepcopy(normalized)

    async def insert_rows(
        self,
        table: str,
        rows: Iterable[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        payload = [copy.deepcopy(dict(row)) for row in rows]
        if not payload:
            raise ValueError(f"{table}: batch insert requires at least one row")

        # Preserve the atomicity of one production multi-row INSERT.  This
        # adapter may expose only the PostgREST boundary; graph ordering,
        # uniqueness reconciliation, and conflict handling remain production
        # code exercised by the joined scenario.
        normalized_table = str(table)
        prior_rows = copy.deepcopy(self.tables.get(normalized_table, []))
        prior_insert_count = self.insert_count
        try:
            return [await self.insert_row(normalized_table, row) for row in payload]
        except Exception:
            self.tables[normalized_table] = prior_rows
            self.insert_count = prior_insert_count
            raise

    async def select_one(
        self, table: str, *, filters: Iterable[tuple], **_kwargs: Any
    ) -> dict[str, Any] | None:
        from gateway.research_lab import attested_v2_store as store

        normalized_filters = tuple(filters)
        if (
            str(table) == store.ANCESTRY_CHECKPOINT_TABLE
            and self.unknown_commit_pending is not None
            and normalized_filters
            == (("root_receipt_hash", self.unknown_commit_root),)
        ):
            self.unknown_commit_readbacks += 1
            if (
                self.unknown_commit_readbacks == 3
                and not self.unknown_commit_visible
            ):
                checkpoint = copy.deepcopy(
                    self.unknown_commit_pending["checkpoint"]
                )
                activation = copy.deepcopy(
                    self.unknown_commit_pending["activation"]
                )
                self.tables.setdefault(
                    store.ANCESTRY_CHECKPOINT_TABLE,
                    [],
                ).append(checkpoint)
                self.tables.setdefault(
                    store.ANCESTRY_ACTIVATION_TABLE,
                    [],
                ).append(activation)
                self.insert_count += 2
                self.unknown_commit_visible = True
        rows = [
            row
            for row in self.tables.get(str(table), [])
            if self._matches(row, normalized_filters)
        ]
        if len(rows) > 1:
            raise RuntimeError("compact rehearsal select_one is ambiguous")
        return copy.deepcopy(rows[0]) if rows else None

    async def select_all(
        self,
        table: str,
        *,
        filters: Iterable[tuple] = (),
        order_by: Iterable[tuple[str, bool]] = (),
        max_rows: int | None = None,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        rows = [
            copy.deepcopy(row)
            for row in self.tables.get(str(table), [])
            if self._matches(row, filters)
        ]
        for field, descending in reversed(tuple(order_by)):
            rows.sort(key=lambda row: str(row.get(field)), reverse=bool(descending))
        if max_rows is not None and len(rows) > int(max_rows):
            raise RuntimeError("paginated select exceeded max_rows=%s" % max_rows)
        return rows

    async def select_many(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return await self.select_all(*args, **kwargs)

    async def call_rpc(self, name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        from gateway.research_lab import attested_v2_store as store

        if name != store.ANCESTRY_CHECKPOINT_RPC or set(payload) != {"checkpoint"}:
            raise RuntimeError("compact rehearsal received an unknown store RPC")
        row = copy.deepcopy(dict(payload["checkpoint"]))
        root = str(row["root_receipt_hash"])
        self.checkpoint_rpc_writes[root] = (
            self.checkpoint_rpc_writes.get(root, 0) + 1
        )
        existing = await self.select_one(
            store.ANCESTRY_CHECKPOINT_TABLE,
            filters=(("root_receipt_hash", root),),
        )
        certificate = row.get("certificate_doc")
        issuer = (
            certificate.get("issuer_boot_identity")
            if isinstance(certificate, Mapping)
            else None
        )
        if (
            self.unknown_commit_root is None
            and isinstance(issuer, Mapping)
            and issuer.get("role") == "validator_weights"
        ):
            if existing is not None:
                raise RuntimeError(
                    "compact unknown-commit fault targeted an existing checkpoint"
                )
            self.unknown_commit_root = root
            self.unknown_commit_pending = {
                "checkpoint": row,
                "activation": {
                    "lineage_id": row["lineage_id"],
                    "activation_root_receipt_hash": root,
                    "activation_certificate_hash": row["certificate_hash"],
                },
            }
            raise httpx.ReadTimeout(
                "compact rehearsal checkpoint response timed out after commit"
            )
        if existing is None:
            await self.insert_row(store.ANCESTRY_CHECKPOINT_TABLE, row)
            await self.insert_row(
                store.ANCESTRY_ACTIVATION_TABLE,
                {
                    "lineage_id": row["lineage_id"],
                    "activation_root_receipt_hash": root,
                    "activation_certificate_hash": row["certificate_hash"],
                },
            )
        elif existing != row:
            raise RuntimeError("compact rehearsal checkpoint RPC conflicts")
        return {
            "root_receipt_hash": root,
            "certificate_hash": row["certificate_hash"],
            "proof_hash": row["proof_hash"],
            "lineage_id": row["lineage_id"],
            "certificate_sequence": row["certificate_sequence"],
            "checkpoint_graph_hash": row["checkpoint_graph_hash"],
            "root_activated": True,
        }

    async def unknown_commit_sleep(self, seconds: float) -> None:
        self.unknown_commit_sleep_delays.append(float(seconds))


class _StrictCutoverQuery:
    """Fail-closed query builder for the two production cutover reads."""

    def __init__(
        self,
        boundary: "_StrictCutoverDatabase",
        *,
        table: str,
        allowed_table: str,
    ) -> None:
        if table != allowed_table:
            raise RuntimeError("compact cutover database table differs")
        self._boundary = boundary
        self._table = table
        self._select: str | None = None
        self._filter: tuple[str, Any] | None = None
        self._limit: int | None = None
        self._executed = False

    def select(self, fields: str) -> "_StrictCutoverQuery":
        if self._select is not None:
            raise RuntimeError("compact cutover database select repeated")
        self._select = str(fields)
        return self

    def eq(self, field: str, value: Any) -> "_StrictCutoverQuery":
        if self._filter is not None:
            raise RuntimeError("compact cutover database filter repeated")
        self._filter = (str(field), value)
        return self

    def limit(self, value: int) -> "_StrictCutoverQuery":
        if self._limit is not None:
            raise RuntimeError("compact cutover database limit repeated")
        self._limit = int(value)
        return self

    def execute(self) -> SimpleNamespace:
        expected = self._boundary.expected_query(self._table)
        observed = (self._table, self._select, self._filter, self._limit)
        if self._executed or observed != expected:
            raise RuntimeError("compact cutover database query contract differs")
        self._executed = True
        self._boundary.operations.append(observed)
        return SimpleNamespace(
            data=[copy.deepcopy(self._boundary.row_for(self._table))]
        )


class _StrictCutoverClient:
    def __init__(
        self,
        boundary: "_StrictCutoverDatabase",
        *,
        allowed_table: str,
    ) -> None:
        self._boundary = boundary
        self._allowed_table = allowed_table

    def table(self, table: str) -> _StrictCutoverQuery:
        return _StrictCutoverQuery(
            self._boundary,
            table=str(table),
            allowed_table=self._allowed_table,
        )


class _StrictCutoverDatabase:
    """Lowest-boundary adapter for production cutover authority validation."""

    STATE_TABLE = "research_lab_stateful_subnet_epoch_cutover_state_v1"
    LEDGER_TABLE = "research_lab_stateful_subnet_epoch_cutovers_v1"
    SERVICE_URL = "https://rehearsal.supabase.invalid"
    SERVICE_KEY = "rehearsal-service-role"

    def __init__(self, cutover: SubnetEpochCutover) -> None:
        self.cutover = cutover
        self.write_client_calls = 0
        self.http1_client_calls = 0
        self.operations: list[tuple[str, str | None, tuple[str, Any] | None, int | None]] = []

    def get_write_client(self) -> _StrictCutoverClient:
        self.write_client_calls += 1
        if self.write_client_calls != 1:
            raise RuntimeError("compact cutover write client count differs")
        return _StrictCutoverClient(self, allowed_table=self.STATE_TABLE)

    def create_http1_sync_client(
        self, supabase_url: str, supabase_key: str
    ) -> _StrictCutoverClient:
        self.http1_client_calls += 1
        if (
            self.http1_client_calls != 1
            or supabase_url != self.SERVICE_URL
            or supabase_key != self.SERVICE_KEY
        ):
            raise RuntimeError("compact cutover HTTP/1 client contract differs")
        return _StrictCutoverClient(self, allowed_table=self.LEDGER_TABLE)

    def expected_query(
        self, table: str
    ) -> tuple[str, str, tuple[str, Any], int]:
        if table == self.STATE_TABLE:
            return (
                table,
                "lifecycle_state,mapping_hash,last_legacy_epoch_id,"
                "first_settlement_epoch_id",
                ("singleton", True),
                2,
            )
        if table == self.LEDGER_TABLE:
            return (
                table,
                "mapping_hash,manifest_doc",
                ("mapping_hash", self.cutover.mapping_hash),
                2,
            )
        raise RuntimeError("compact cutover database table is undeclared")

    def row_for(self, table: str) -> dict[str, Any]:
        if table == self.STATE_TABLE:
            return {
                "lifecycle_state": "stateful_active",
                "mapping_hash": self.cutover.mapping_hash,
                "last_legacy_epoch_id": self.cutover.last_legacy_epoch_id,
                "first_settlement_epoch_id": (
                    self.cutover.first_settlement_epoch_id
                ),
            }
        if table == self.LEDGER_TABLE:
            return {
                "mapping_hash": self.cutover.mapping_hash,
                "manifest_doc": self.cutover.to_dict(),
            }
        raise RuntimeError("compact cutover database table is undeclared")

    def assert_complete(self) -> None:
        expected = [
            self.expected_query(self.STATE_TABLE),
            self.expected_query(self.LEDGER_TABLE),
        ]
        if (
            self.write_client_calls != 1
            or self.http1_client_calls != 1
            or self.operations != expected
        ):
            raise RuntimeError("compact cutover database boundary is incomplete")


class _DeterministicDrand:
    def generate_commit(self, **kwargs: Any) -> tuple[bytes, int]:
        expected_fields = {
            "uids",
            "weights_u16",
            "version_key",
            "last_epoch_block",
            "pending_epoch_at",
            "subnet_epoch_index",
            "tempo",
            "blocks_since_last_step",
            "current_block",
            "subnet_reveal_period_epochs",
            "block_time",
            "hotkey_public_key",
        }
        if set(kwargs) != expected_fields:
            raise RuntimeError("compact drand boundary fields differ")
        uids = [int(value) for value in kwargs["uids"]]
        weights = [int(value) for value in kwargs["weights_u16"]]
        hotkey = bytes(kwargs["hotkey_public_key"])
        if (
            not uids
            or uids != sorted(set(uids))
            or len(uids) != len(weights)
            or any(value < 0 or value > 65535 for value in uids)
            or any(value < 1 or value > 65535 for value in weights)
            or len(hotkey) != 32
        ):
            raise RuntimeError("compact drand boundary payload differs")
        payload = {
            key: kwargs[key]
            for key in expected_fields - {"hotkey_public_key"}
        }
        payload["uids"] = uids
        payload["weights_u16"] = weights
        payload["hotkey_public_key"] = hotkey.hex()
        body = canonical_json(payload).encode("utf-8")
        return hashlib.sha512(body).digest(), int(kwargs["subnet_epoch_index"]) + 1


class _StrictChainSource:
    """Read/finalization boundary consumed by both validator authorities."""

    observation_scope = "7a" * 16

    def __init__(
        self,
        *,
        cutover: SubnetEpochCutover,
        boundary_doc: Mapping[str, Any],
        current_doc: Mapping[str, Any],
        finalized_chain_state_root: str,
        chain_profile: Mapping[str, Any],
    ) -> None:
        self.cutover = cutover
        self.boundary_doc = dict(boundary_doc)
        self.current_doc = dict(current_doc)
        self.finalized_chain_state_root = str(finalized_chain_state_root)
        self.chain_profile = dict(chain_profile)
        self.expected_uids: list[int] = []
        self.expected_weights: list[int] = []
        self.broadcast_extrinsic_hex: str | None = None
        self.broadcast_extrinsic_hash: str | None = None
        self.finalization_calls = 0
        self.finalization_scan_ids: list[str] = []
        self.finalization_job_ids: list[str] = []

    @staticmethod
    def _attempt_and_artifacts(
        *, job_id: str, purpose: str, operation: str, attempt_number: int
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        request_body = ("request:" + operation).encode("ascii")
        response_body = ("response:" + operation).encode("ascii")
        request_hash = sha256_bytes(request_body)
        response_hash = sha256_bytes(response_body)
        attempt = build_transport_attempt(
            request_id="%032x" % (attempt_number + 1),
            logical_operation_id=f"{job_id}:{operation}",
            job_id=job_id,
            purpose=purpose,
            provider_id="bittensor_chain",
            attempt_number=attempt_number,
            method="POST",
            destination_host="entrypoint-finney.opentensor.ai",
            destination_port=443,
            path_hash=sha256_json({"path": "/"}),
            nonsecret_headers_hash=sha256_json({"accept": "application/json"}),
            body_hash=request_hash,
            credential_ref_hash=sha256_json({"credential": "none"}),
            retry_policy_hash=sha256_json({"retry": "chain"}),
            timeout_ms=30_000,
            started_at=NOW,
            terminal_status="authenticated_response",
            http_status=200,
            response_hash=response_hash,
            request_artifact_hash=request_hash,
            response_artifact_hash=response_hash,
            tls_peer_chain_hash=sha256_json({"tls": "chain"}),
            tls_protocol="TLSv1.3",
            failure_code=None,
            completed_at=NOW,
        )
        artifacts = [
            {
                "artifact_hash": request_hash,
                "kind": "chain_rpc_request",
                "body_b64": base64.b64encode(request_body).decode("ascii"),
            },
            {
                "artifact_hash": response_hash,
                "kind": "chain_rpc_response",
                "body_b64": base64.b64encode(response_body).decode("ascii"),
            },
        ]
        return attempt, artifacts

    def read_finalized_snapshot(self, *, netuid: int, epoch_id: int) -> dict[str, Any]:
        if int(netuid) != NETUID or int(epoch_id) != EPOCH_ID:
            raise RuntimeError("compact chain snapshot identity differs")
        operations = (
            ("chain-state", "validator.chain_state.v2", "head"),
            ("chain-state", "validator.chain_state.v2", "header"),
            ("metagraph-state", "validator.metagraph_state.v2", "metagraph"),
            (
                "subnet-epoch-current",
                "validator.subnet_epoch_snapshot.v2",
                "current",
            ),
            (
                "subnet-epoch-boundary",
                "validator.subnet_epoch_snapshot.v2",
                "boundary",
            ),
        )
        attempts: list[dict[str, Any]] = []
        artifacts: list[dict[str, Any]] = []
        jobs: dict[str, str] = {}
        for index, (kind, purpose, operation) in enumerate(operations):
            job_id = f"{kind}:{EPOCH_ID}"
            attempt, items = self._attempt_and_artifacts(
                job_id=job_id,
                purpose=purpose,
                operation=operation,
                attempt_number=index,
            )
            attempts.append(attempt)
            artifacts.extend(items)
            if kind == "chain-state":
                jobs["chain_state"] = job_id
            elif kind == "metagraph-state":
                jobs["metagraph_state"] = job_id
            elif kind == "subnet-epoch-current":
                jobs["subnet_epoch_snapshot"] = job_id
            else:
                jobs["subnet_epoch_boundary"] = job_id
        block = int(self.current_doc["current_block"])
        return {
            "observation_scope": self.observation_scope,
            "finalized_block_hash": str(self.current_doc["block_hash"]),
            "header": {
                "block": block,
                "state_root": "12" * 32,
                "state_root_commitment": self.finalized_chain_state_root,
                "parent_hash": "34" * 32,
                "extrinsics_root": "56" * 32,
            },
            "metagraph": {
                "netuid": NETUID,
                "block": block,
                "owner_hotkey": "burn-hotkey",
                "hotkeys": [
                    "burn-hotkey",
                    "fulfillment-hotkey",
                    "lab-hotkey",
                    "source-hotkey",
                ],
            },
            "attempts": attempts,
            "artifacts": artifacts,
            "jobs": jobs,
            "epoch_authority": dict(self.current_doc),
            "epoch_boundary": dict(self.boundary_doc),
        }

    def capture_stateful_epoch_boundary(
        self,
        *,
        cutover_manifest: Mapping[str, Any],
        settlement_epoch_id: int,
        capture_scope: str,
    ) -> dict[str, Any]:
        if (
            dict(cutover_manifest) != self.cutover.to_dict()
            or int(settlement_epoch_id) != EPOCH_ID
            or not str(capture_scope).startswith("sha256:")
        ):
            raise RuntimeError("compact boundary capture request differs")
        snapshot = self.read_finalized_snapshot(netuid=NETUID, epoch_id=EPOCH_ID)
        job_id = snapshot["jobs"]["subnet_epoch_boundary"]
        attempts = [item for item in snapshot["attempts"] if item["job_id"] == job_id]
        hashes = {
            value
            for item in attempts
            for value in (item["request_artifact_hash"], item["response_artifact_hash"])
        }
        return {
            "finalized_block_hash": snapshot["finalized_block_hash"],
            "header": snapshot["header"],
            "epoch_authority": dict(self.boundary_doc),
            "epoch_boundary": dict(self.boundary_doc),
            "attempts": attempts,
            "artifacts": [
                item for item in snapshot["artifacts"] if item["artifact_hash"] in hashes
            ],
            "jobs": {
                "subnet_epoch_snapshot": job_id,
                "subnet_epoch_boundary": job_id,
            },
        }

    def read_chain_signing_runtime(
        self, *, runtime_block_hash: str, max_block_drift: int
    ) -> dict[str, Any]:
        if not str(runtime_block_hash).startswith("0x") or int(max_block_drift) != int(
            self.chain_profile["max_snapshot_block_drift"]
        ):
            raise RuntimeError("compact signing runtime request differs")
        return {
            "spec_version": int(self.chain_profile["spec_version"]),
            "transaction_version": int(self.chain_profile["transaction_version"]),
            "genesis_hash": str(self.chain_profile["genesis_hash"]),
        }

    def record_broadcast(self, extrinsic: bytes) -> None:
        observed = signed_extrinsic_hash_v2(extrinsic)
        if self.broadcast_extrinsic_hash not in {None, observed}:
            raise RuntimeError("compact chain boundary received another extrinsic")
        self.broadcast_extrinsic_hex = bytes(extrinsic).hex()
        self.broadcast_extrinsic_hash = observed

    def find_finalized_extrinsic_inclusion(
        self,
        *,
        expected_extrinsics: Mapping[str, str],
        expected_commitments: Mapping[str, Mapping[str, Any]],
        minimum_block: int,
        maximum_block: int,
        epoch_id: int,
        finalization_scan_id: str,
    ) -> dict[str, Any]:
        if (
            int(epoch_id) != EPOCH_ID
            or len(expected_extrinsics) != 1
            or set(expected_extrinsics) != set(expected_commitments)
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", str(finalization_scan_id).lower()
            )
        ):
            raise RuntimeError("compact finalization scan identity differs")
        extrinsic_hash, extrinsic_hex = next(iter(expected_extrinsics.items()))
        if (
            extrinsic_hash != self.broadcast_extrinsic_hash
            or extrinsic_hex != self.broadcast_extrinsic_hex
        ):
            raise RuntimeError("compact finalization did not read the broadcast bytes")
        finalized_block = int(self.current_doc["current_block"]) + 1
        if not (int(minimum_block) <= finalized_block < int(maximum_block)):
            raise RuntimeError("compact finalized block is outside the mortal era")
        scan_id = str(finalization_scan_id).lower()
        job_id = "weight-finalization:%d:%s" % (
            int(epoch_id),
            scan_id[len("sha256:") :],
        )
        attempt, artifacts = self._attempt_and_artifacts(
            job_id=job_id,
            purpose="validator.weights.finalized.v2",
            operation="finalized-inclusion",
            attempt_number=0,
        )
        self.finalization_calls += 1
        self.finalization_scan_ids.append(scan_id)
        self.finalization_job_ids.append(job_id)
        return {
            "extrinsic_hash": extrinsic_hash,
            "extrinsic_hex": extrinsic_hex,
            "finalized_block": finalized_block,
            "finalized_block_hash": "ab" * 32,
            "state_transition_hash": sha256_json(
                {
                    "last_update": finalized_block,
                    "uids": self.expected_uids,
                    "weights_u16": self.expected_weights,
                }
            ),
            "attempts": [attempt],
            "artifacts": artifacts,
            "job_id": job_id,
        }


class _CompactEnclaveClient:
    """In-process Nitro adapter around the production enclave authorities."""

    def __init__(self, *, weight_authority: Any, hotkey_authority: Any) -> None:
        self.weight_authority = weight_authority
        self.hotkey_authority = hotkey_authority
        self.last_response: dict[str, Any] | None = None
        self.last_commit: dict[str, Any] | None = None
        self.last_authorization: dict[str, Any] | None = None

    def get_hotkey_state_v2(self) -> dict[str, Any]:
        return self.hotkey_authority.public_state()

    def compute_authoritative_weights_v2(
        self, request: Mapping[str, Any]
    ) -> dict[str, Any]:
        response = self.weight_authority.compute(dict(request))
        authorization_id = self.hotkey_authority.register_weight_result(
            {
                field: response[field]
                for field in (
                    "weight_snapshot",
                    "weight_result",
                    "weights_signature",
                    "receipt_graph_delta",
                    "ancestry_commitment",
                    "boot_identity",
                )
            }
        )
        result = {**response, "weight_authorization_id": authorization_id}
        self.last_response = copy.deepcopy(result)
        return result

    def sign_application_message_v2(
        self,
        message: bytes,
        *,
        parent_receipt_hash: str | None = None,
        compact_ancestry_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = self.hotkey_authority.sign_application_message(
            message_hex=bytes(message).hex(),
            parent_receipt_hash=parent_receipt_hash,
        )
        if compact_ancestry_context is not None:
            if set(compact_ancestry_context) != {
                "validator_receipt_delta",
                "upstream_ancestry_proofs",
                "epoch_authority",
            }:
                raise RuntimeError("compact publication ancestry request differs")
            result["validator_ancestry_proof"] = (
                self.weight_authority.issue_validator_publication_ancestry_proof(
                    validator_receipt_delta=compact_ancestry_context[
                        "validator_receipt_delta"
                    ],
                    upstream_ancestry_proofs=compact_ancestry_context[
                        "upstream_ancestry_proofs"
                    ],
                    binding_receipt=result["receipt"],
                    epoch_authority=compact_ancestry_context["epoch_authority"],
                )
            )
        return result

    def prepare_weight_commit_v2(
        self, request: Mapping[str, Any]
    ) -> dict[str, Any]:
        result = self.hotkey_authority.prepare_weight_commit(**dict(request))
        self.last_commit = {"request": dict(request), "result": dict(result)}
        return result

    def sign_weight_extrinsic_v2(
        self, request: Mapping[str, Any]
    ) -> dict[str, Any]:
        return self.hotkey_authority.sign_weight_extrinsic(**dict(request))

    def confirm_weight_publication_v2(
        self,
        weight_authorization_id: str,
        *,
        finalization_scan_id: str,
        compact_ancestry_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = self.hotkey_authority.confirm_weight_publication(
            weight_authorization_id=weight_authorization_id,
            finalization_scan_id=finalization_scan_id,
        )
        if compact_ancestry_context is not None:
            if set(compact_ancestry_context) != {
                "publication_ancestry_proof",
                "epoch_authority",
            }:
                raise RuntimeError("compact finalization ancestry request differs")
            result["validator_ancestry_proof"] = (
                self.weight_authority.issue_validator_finalization_ancestry_proof(
                    validator_receipt_delta=result["receipt_graph_delta"],
                    publication_ancestry_proof=compact_ancestry_context[
                        "publication_ancestry_proof"
                    ],
                    epoch_authority=compact_ancestry_context["epoch_authority"],
                )
            )
        return result

    def recover_compact_weight_publication_v2(self, **kwargs: Any) -> dict[str, Any]:
        return self.hotkey_authority.recover_compact_weight_publication(**kwargs)


class _Era:
    def encode(self, value: Mapping[str, Any]) -> None:
        self.value = dict(value)

    def birth(self, current: int) -> int:
        return int(current) - (int(current) % int(self.value["period"]))


class _RuntimeConfig:
    def create_scale_object(self, name: str) -> _Era:
        if name != "Era":
            raise RuntimeError("compact SDK requested an unknown SCALE object")
        return _Era()


class _CompactSDKSubstrate:
    runtime_config = _RuntimeConfig()

    def __init__(
        self,
        *,
        client: _CompactEnclaveClient,
        chain_source: _StrictChainSource,
        hotkey_public_key: bytes,
        profile: Mapping[str, Any],
        current_block: int,
    ) -> None:
        self.client = client
        self.chain_source = chain_source
        self.hotkey_public_key = bytes(hotkey_public_key)
        self.profile = dict(profile)
        self.current_block = int(current_block)
        self.runtime_config = _RuntimeConfig()
        self.create_signed_extrinsic = self._create_signed_extrinsic

    def init_runtime(self, block_hash: str | None = None) -> None:
        if block_hash != self.get_chain_finalised_head():
            raise RuntimeError("compact SDK runtime hash differs")

    def get_account_nonce(self, _address: str) -> int:
        return 7

    def get_chain_finalised_head(self) -> str:
        return "0x" + "cc" * 32

    def get_block_number(self, head: str) -> int:
        if head != self.get_chain_finalised_head():
            raise RuntimeError("compact SDK finalized head differs")
        return self.current_block

    def get_block_hash(self, block_id: int) -> str:
        if int(block_id) < 0:
            raise RuntimeError("compact SDK era birth block is invalid")
        return "0x" + hashlib.sha256(f"birth:{block_id}".encode()).hexdigest()

    def generate_signature_payload(self, **kwargs: Any) -> SimpleNamespace:
        commit = self.client.last_commit
        response = self.client.last_response
        if commit is None or response is None:
            raise RuntimeError("compact SDK signing preceded enclave authorization")
        request = commit["request"]
        result = response["weight_result"]
        era = dict(kwargs["era"])
        block_hash = self.get_block_hash(
            int(era["current"]) - (int(era["current"]) % int(era["period"]))
        ).removeprefix("0x")
        authorization = build_weight_extrinsic_authorization_v2(
            profile=self.profile,
            validator_hotkey=str(request["validator_hotkey"])
            if "validator_hotkey" in request
            else self.client.hotkey_authority.validator_hotkey,
            hotkey_public_key_hex=self.hotkey_public_key.hex(),
            epoch_id=int(result["epoch_id"]),
            netuid=int(result["netuid"]),
            subnet_epoch_index=int(request["subnet_epoch_index"]),
            weight_receipt_hash=str(
                response["receipt_graph_delta"]["root_receipt_hash"]
            ),
            weight_submission_event_hash=str(
                request["weight_submission_event_hash"]
            ),
            weights_hash=str(result["weights_hash"]),
            sparse_uids=result["sparse_uids"],
            sparse_weights_u16=result["sparse_weights_u16"],
            commitment=bytes.fromhex(commit["result"]["commitment_hex"]),
            reveal_round=int(commit["result"]["reveal_round"]),
            era_current=int(era["current"]),
            nonce=int(kwargs["nonce"]),
            block_hash=block_hash,
        )
        self.client.last_authorization = authorization
        return SimpleNamespace(data=bytes.fromhex(authorization["signed_message_hex"]))

    def _create_signed_extrinsic(self, **kwargs: Any) -> SimpleNamespace:
        authorization = self.client.last_authorization
        if authorization is None:
            raise RuntimeError("compact SDK extrinsic preceded authorization")
        signature = bytes(kwargs["signature"])
        encoded = encode_signed_extrinsic_v2(
            hotkey_public_key_hex=self.hotkey_public_key.hex(),
            signature_hex=signature.hex(),
            era_period=int(authorization["era_period"]),
            era_current=int(authorization["era_current"]),
            nonce=int(authorization["nonce"]),
            call_data_hex=str(authorization["call_data_hex"]),
        )
        self.chain_source.record_broadcast(encoded)
        return SimpleNamespace(data=SimpleNamespace(data=encoded), signature=signature)

    def rpc_request(self, method: str, params: list[Any]) -> dict[str, Any]:
        if method != "author_submitExtrinsic" or len(params) != 1:
            raise RuntimeError("compact recovery rebroadcast request differs")
        encoded = bytes.fromhex(str(params[0]).removeprefix("0x"))
        self.chain_source.record_broadcast(encoded)
        return {"result": self.chain_source.broadcast_extrinsic_hash}


class _CompactSubtensor:
    """Strict SDK/chain adapter invoked inside the production primary loop."""

    def __init__(
        self,
        *,
        substrate: _CompactSDKSubstrate,
        profile: Mapping[str, Any],
        hotkey_public_key: bytes,
        current_doc: Mapping[str, Any],
    ) -> None:
        self.substrate = substrate
        self.profile = dict(profile)
        self.hotkey_public_key = bytes(hotkey_public_key)
        self.current_doc = dict(current_doc)

    def set_weights(
        self,
        *,
        netuid: int,
        wallet: Any,
        uids: list[int],
        weights: list[float],
        wait_for_finalization: bool,
        mechid: int,
        period: int,
    ) -> tuple[bool, str]:
        from leadpoet_canonical.weights import normalize_to_u16
        from validator_tee.host.enclave_hotkey_v2 import _weight_extrinsic_module

        if (
            int(netuid) != NETUID
            or wait_for_finalization is not True
            or int(mechid) != 0
            or int(period) != int(self.profile["extrinsic_period"])
        ):
            raise RuntimeError("compact primary SDK call shape differs")
        normalized = normalize_to_u16(
            [int(uid) for uid in uids], [float(value) for value in weights]
        )
        expected = self.substrate.chain_source.expected_weights
        if (
            [int(uid) for uid in uids]
            != self.substrate.chain_source.expected_uids
            or normalized != expected
        ):
            raise RuntimeError("compact primary SDK vector differs")
        module = _weight_extrinsic_module()
        commitment, reveal_round = module.get_encrypted_commit_v2(
            uids=[int(uid) for uid in uids],
            weights=normalized,
            version_key=int(self.profile["version_key"]),
            last_epoch_block=int(self.current_doc["last_epoch_block"]),
            pending_epoch_at=int(self.current_doc["pending_epoch_at"]),
            subnet_epoch_index=int(self.current_doc["subnet_epoch_index"]),
            tempo=int(self.current_doc["tempo"]),
            blocks_since_last_step=int(self.current_doc["blocks_since_last_step"]) + 1,
            current_block=int(self.current_doc["current_block"]) + 1,
            subnet_reveal_period_epochs=int(
                self.profile["subnet_reveal_period_epochs"]
            ),
            block_time=float(self.profile["block_time_millis"]) / 1000.0,
            hotkey=self.hotkey_public_key,
        )
        call = SimpleNamespace(
            value={
                "call_module": "SubtensorModule",
                "call_function": "commit_timelocked_weights",
                "call_args": {
                    "netuid": NETUID,
                    "commitment": bytes(commitment).hex(),
                    "reveal_round": int(reveal_round),
                },
            }
        )
        extrinsic = self.substrate.create_signed_extrinsic(
            call=call,
            keypair=wallet.hotkey,
            era={
                "period": int(period),
                "current": int(self.current_doc["current_block"]) + 1,
            },
            nonce=7,
            tip=0,
            tip_asset_id=None,
            signature=None,
        )
        if signed_extrinsic_hash_v2(bytes(extrinsic.data.data)) != (
            self.substrate.chain_source.broadcast_extrinsic_hash
        ):
            raise RuntimeError("compact primary SDK broadcast bytes differ")
        return True, "strict local chain accepted exact signed bytes"


class _AuditorChainSubstrate:
    """Strict finalized LastUpdate/Weights readback boundary for one auditor."""

    def __init__(
        self,
        *,
        auditor_uid: int,
        baseline_last_update: int,
        finalized_last_update: int,
    ) -> None:
        self.auditor_uid = int(auditor_uid)
        self.baseline_last_update = int(baseline_last_update)
        self.finalized_last_update = int(finalized_last_update)
        self.last_update = int(baseline_last_update)
        self.weights: list[tuple[int, int]] = []
        self.reads: list[str] = []

    def get_chain_finalised_head(self) -> str:
        state = "after" if self.last_update == self.finalized_last_update else "before"
        return "0x" + hashlib.sha256(
            f"auditor-finalized-head:{self.auditor_uid}:{state}".encode("ascii")
        ).hexdigest()

    def get_block_hash(self, block_id: int) -> str:
        return "0x" + hashlib.sha256(
            f"auditor-block:{int(block_id)}".encode("ascii")
        ).hexdigest()

    def query(
        self,
        *,
        module: str,
        storage_function: str,
        params: list[int],
        block_hash: str | None,
    ) -> SimpleNamespace:
        if module != "SubtensorModule" or block_hash != self.get_chain_finalised_head():
            raise RuntimeError("auditor finalized storage query scope differs")
        self.reads.append(storage_function)
        if storage_function == "LastUpdate" and params == [NETUID]:
            values = [0] * (self.auditor_uid + 1)
            values[self.auditor_uid] = self.last_update
            return SimpleNamespace(value=values)
        if storage_function == "Weights" and params == [NETUID, self.auditor_uid]:
            return SimpleNamespace(value=list(self.weights))
        if storage_function == "Tempo" and params == [NETUID]:
            return SimpleNamespace(value=360)
        if storage_function == "RevealPeriodEpochs" and params == [NETUID]:
            return SimpleNamespace(value=1)
        raise RuntimeError("auditor finalized storage query differs")

    def record_finalized_vector(
        self,
        *,
        uids: list[int],
        weights_u16: list[int],
    ) -> None:
        if self.last_update != self.baseline_last_update or self.weights:
            raise RuntimeError("auditor chain adapter received a duplicate write")
        self.weights = list(zip(uids, weights_u16))
        self.last_update = self.finalized_last_update


class _AuditorSubtensor:
    """Terminate each auditor SDK chain write at an exact-vector adapter."""

    def __init__(
        self,
        *,
        auditor_uid: int,
        wallet: Any,
        expected_uids: list[int],
        expected_weights_u16: list[int],
        baseline_last_update: int,
        finalized_last_update: int,
    ) -> None:
        self.wallet = wallet
        self.expected_uids = list(expected_uids)
        self.expected_weights_u16 = list(expected_weights_u16)
        self.substrate = _AuditorChainSubstrate(
            auditor_uid=auditor_uid,
            baseline_last_update=baseline_last_update,
            finalized_last_update=finalized_last_update,
        )
        self.set_calls = 0

    def get_subnet_hyperparameters(
        self,
        _netuid: int,
        block: int | None = None,
    ) -> Any:
        raise RuntimeError(
            "auditor SDK hyperparameters escaped the compatibility adapter"
        )

    def set_weights(
        self,
        *,
        netuid: int,
        wallet: Any,
        uids: list[int],
        weights: list[float],
        wait_for_finalization: bool,
        mechid: int,
    ) -> tuple[bool, str]:
        from leadpoet_canonical.weights import normalize_to_u16

        normalized_uids = [int(value) for value in uids]
        normalized_weights = normalize_to_u16(
            normalized_uids,
            [float(value) for value in weights],
        )
        if (
            int(netuid) != NETUID
            or wallet is not self.wallet
            or wait_for_finalization is not True
            or int(mechid) != 0
            or normalized_uids != self.expected_uids
            or normalized_weights != self.expected_weights_u16
            or self.set_calls != 0
        ):
            raise RuntimeError("auditor SDK submission differs")
        self.set_calls += 1
        self.substrate.record_finalized_vector(
            uids=normalized_uids,
            weights_u16=normalized_weights,
        )
        return True, "strict auditor chain adapter finalized exact vector"


def _make_request(accept_encoding: str = "identity") -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/weights/v2/published-compact/%d/%d" % (NETUID, EPOCH_ID),
            "headers": [(b"accept-encoding", accept_encoding.encode("ascii"))],
            "query_string": b"",
            "server": ("127.0.0.1", 0),
            "client": ("127.0.0.1", 0),
            "scheme": "http",
        }
    )


class _AuthorityHTTPServer:
    def __init__(self, body: bytes) -> None:
        expected_path = "/weights/v2/published-compact/%d/%d" % (NETUID, EPOCH_ID)
        payload = bytes(body)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:
                return

            def do_GET(self) -> None:  # noqa: N802
                if self.path != expected_path:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def __enter__(self) -> "_AuthorityHTTPServer":
        self.thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("compact authority HTTP adapter did not stop")


def _allocation_guard(candidate_sha: str) -> dict[str, Any]:
    import neurons.validator as validator_module
    from research_lab import validator_integration
    from weight_readiness_runner import _build_handoff

    handoff = _build_handoff(EPOCH_ID)
    calls: list[tuple[str, int]] = []

    def fetch(gateway_url: str, epoch: int) -> dict[str, Any]:
        if gateway_url != "https://gateway.rehearsal.invalid" or int(epoch) != EPOCH_ID:
            raise RuntimeError("compact allocation fetch request differs")
        calls.append((gateway_url, int(epoch)))
        return copy.deepcopy(handoff)

    validator = validator_module.Validator.__new__(validator_module.Validator)
    validator.config = SimpleNamespace(netuid=NETUID)
    validator._research_lab_allocation_guard_cache = {}
    with _patched(
        validator_integration,
        "fetch_research_lab_attested_allocation_bundle",
        fetch,
    ), _environment(
        {
            "VALIDATOR_V2_GATEWAY_URL": "https://gateway.rehearsal.invalid",
            "BITTENSOR_NETWORK": "finney",
            "NETUID": str(NETUID),
            "RESEARCH_LAB_EMISSION_PERCENT": "20",
        }
    ):
        result = asyncio.run(validator._research_lab_pre_weight_submission_guard(EPOCH_ID))
    if result.get("verified") is not True or result.get("abort_chain_submission"):
        raise RuntimeError("production allocation guard rejected the compact fixture")
    if calls != [("https://gateway.rehearsal.invalid", EPOCH_ID)]:
        raise RuntimeError("production allocation guard fetch count differs")
    return result


def _build_runtime(candidate_sha: str, allocation: Mapping[str, Any]) -> dict[str, Any]:
    from validator_tee.enclave.hotkey_authority_v2 import (
        ValidatorHotkeyAuthorityV2,
        _Sr25519Backend,
        load_chain_signing_profile,
    )
    from validator_tee.enclave.weight_authority_v2 import ValidatorWeightAuthorityV2

    fixture = SanitizedWeightFixture(candidate_sha=candidate_sha, epoch_id=EPOCH_ID)
    cutover_block = EPOCH_ID * 360
    cutover = SubnetEpochCutover(
        network_genesis_hash=GENESIS_HASH,
        netuid=NETUID,
        cutover_block=cutover_block,
        cutover_block_hash="0x" + "20" * 32,
        first_subnet_epoch_index=EPOCH_ID,
        first_settlement_epoch_id=EPOCH_ID,
        last_legacy_epoch_id=EPOCH_ID - 1,
    )
    boundary = SubnetEpochSnapshot(
        network_genesis_hash=GENESIS_HASH,
        netuid=NETUID,
        head_kind="finalized",
        block_hash=cutover.cutover_block_hash,
        current_block=cutover_block,
        last_epoch_block=cutover_block,
        pending_epoch_at=cutover_block + 360,
        subnet_epoch_index=EPOCH_ID,
        tempo=360,
        blocks_since_last_step=0,
        observed_at=NOW,
    )
    current = SubnetEpochSnapshot(
        network_genesis_hash=GENESIS_HASH,
        netuid=NETUID,
        head_kind="finalized",
        block_hash="0x" + "30" * 32,
        current_block=cutover_block + 340,
        last_epoch_block=cutover_block,
        pending_epoch_at=cutover_block + 360,
        subnet_epoch_index=EPOCH_ID,
        tempo=360,
        blocks_since_last_step=340,
        observed_at=NOW,
    )
    boundary_doc = boundary.to_dict(cutover=cutover)
    current_doc = current.to_dict(cutover=cutover)

    def calculation(parent_hashes: Iterable[str], allocation_receipt: str) -> dict[str, Any]:
        value = fixture.calculation_snapshot(list(parent_hashes), allocation_receipt)
        value["block"] = int(current_doc["current_block"])
        value["research_lab_allocation_doc"] = copy.deepcopy(
            dict(allocation["allocation_component"]["allocation_doc"])
        )
        value["config_hash"] = weight_config_hash(value)
        return value

    preliminary = calculation((), "")
    gateway_config_hash = sha256_json({"compact_gateway": candidate_sha})
    gateway_boot = fixture._boot(
        role="gateway_coordinator",
        key=fixture.coordinator_key,
        config_hash=gateway_config_hash,
    )
    validator_boot = fixture._boot(
        role="validator_weights",
        key=fixture.weight_key,
        config_hash=preliminary["config_hash"],
    )
    lineage_id = derive_ancestry_lineage_id_v2(
        cutover_mapping_hash=cutover.mapping_hash,
        network_genesis_hash=cutover.network_genesis_hash,
        netuid=NETUID,
    )

    def verify_boot(identity: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        if identity.get("commit_sha") != candidate_sha:
            raise RuntimeError("compact rehearsal boot commit differs")
        expected_pcr0 = kwargs.get("expected_pcr0")
        if expected_pcr0 is not None and identity.get("pcr0") != expected_pcr0:
            raise RuntimeError("compact rehearsal boot PCR0 differs")
        return {"verified": True, "pcr0": identity.get("pcr0")}

    event_receipt = fixture.receipt(
        role="gateway_coordinator",
        purpose="research_lab.allocation.v2",
        job_id=f"compact-allocation:{EPOCH_ID}",
        key=fixture.coordinator_key,
        boot=gateway_boot,
        config_hash=gateway_config_hash,
        output_root=sha256_json(allocation["allocation_component"]),
        sequence=0,
    )
    event_delta = {
        "schema_version": ANCESTRY_DELTA_SCHEMA_VERSION,
        "root_receipt_hash": event_receipt["receipt_hash"],
        "boot_identities": [gateway_boot],
        "receipts": [event_receipt],
        "transport_attempts": [],
        "host_operations": [],
    }
    event_certificate = issue_ancestry_certificate_v2(
        local_delta=event_delta,
        lineage_id=lineage_id,
        certificate_sequence=0,
        issuer_boot_identity=gateway_boot,
        issued_at=NOW,
        sign_digest=fixture.coordinator_key.sign,
        boot_attestation_verifier=verify_boot,
        allowed_issuer_roles={"gateway_coordinator"},
        required_purposes={"research_lab.allocation.v2"},
    )
    finalized_chain_state_root = sha256_json(
        {"block": int(current_doc["current_block"]), "kind": "finalized"}
    )
    expected_roots = weight_input_output_roots_v2(
        calculation_snapshot=preliminary,
        finalized_chain_state_root=finalized_chain_state_root,
        gateway_authority_event_hash=event_receipt["receipt_hash"],
    )
    documents = weight_input_value_documents_v2(
        calculation_snapshot=preliminary,
        finalized_chain_state_root=finalized_chain_state_root,
        gateway_authority_event_hash=event_receipt["receipt_hash"],
    )
    proofs: dict[str, dict[str, Any]] = {}
    input_hashes: dict[str, str] = {}
    direct_attempts: list[dict[str, Any]] = []
    for sequence, category in enumerate(sorted(GATEWAY_WEIGHT_INPUT_CATEGORIES)):
        role, purpose = WEIGHT_INPUT_PURPOSES[category]
        if role != "gateway_coordinator":
            raise RuntimeError("compact gateway input role differs")
        job_id = f"compact-weight-input-{category}:{EPOCH_ID}"
        attempt = None
        if category != "anomaly_adjustments":
            attempt = fixture.source_attempt(
                category=category,
                job_id=job_id,
                purpose=purpose,
                sequence=sequence,
                provider_id="supabase",
                host="qplwoislplkcegvdmbim.supabase.co",
                method="GET",
            )
            direct_attempts.append(attempt)
        artifact_hashes = [sha256_json(documents[category]["value"])]
        if attempt is not None:
            artifact_hashes.extend(
                [attempt["request_artifact_hash"], attempt["response_artifact_hash"]]
            )
        receipt = fixture.receipt(
            role=role,
            purpose=purpose,
            job_id=job_id,
            key=fixture.coordinator_key,
            boot=gateway_boot,
            config_hash=gateway_config_hash,
            output_root=expected_roots[category],
            parents=(
                (event_receipt["receipt_hash"],)
                if category == "research_lab_allocation"
                else ()
            ),
            sequence=sequence + 10,
            transport_root=(
                merkle_root([attempt["attempt_hash"]], domain="leadpoet-transport-v2")
                if attempt is not None
                else EMPTY_TRANSPORT_ROOT
            ),
            artifact_root=merkle_root(artifact_hashes, domain="leadpoet-artifact-v2"),
        )
        delta = {
            "schema_version": ANCESTRY_DELTA_SCHEMA_VERSION,
            "root_receipt_hash": receipt["receipt_hash"],
            "boot_identities": [gateway_boot],
            "receipts": [receipt],
            "transport_attempts": [attempt] if attempt is not None else [],
            "host_operations": [],
        }
        certificate = issue_ancestry_certificate_v2(
            local_delta=delta,
            lineage_id=lineage_id,
            certificate_sequence=(1 if category == "research_lab_allocation" else 0),
            issuer_boot_identity=gateway_boot,
            issued_at=NOW,
            sign_digest=fixture.coordinator_key.sign,
            boot_attestation_verifier=verify_boot,
            allowed_issuer_roles={"gateway_coordinator"},
            parent_certificates=(
                [event_certificate] if category == "research_lab_allocation" else []
            ),
            required_purposes={purpose},
        )
        proofs[category] = build_compact_ancestry_proof_from_delta_v2(
            delta,
            certificate,
            expected_lineage_id=lineage_id,
            boot_attestation_verifier=verify_boot,
            allowed_issuer_roles={"gateway_coordinator"},
        )
        input_hashes[category] = receipt["receipt_hash"]

    calculation_snapshot = calculation(
        input_hashes.values(), input_hashes["research_lab_allocation"]
    )
    chain_profile = load_chain_signing_profile(
        _source_root() / "validator_tee/enclave/chain_signing_profile_v2.json"
    )
    chain_source = _StrictChainSource(
        cutover=cutover,
        boundary_doc=boundary_doc,
        current_doc=current_doc,
        finalized_chain_state_root=finalized_chain_state_root,
        chain_profile=chain_profile,
    )
    gateway_lineage = {
        candidate_sha: {
            "roles": {
                "gateway_coordinator": {
                    field: gateway_boot[field]
                    for field in (
                        "commit_sha",
                        "pcr0",
                        "build_manifest_hash",
                        "dependency_lock_hash",
                    )
                }
            }
        }
    }
    weight_authority = ValidatorWeightAuthorityV2(
        boot_identity_supplier=lambda: validator_boot,
        gateway_release_lineage_supplier=lambda: gateway_lineage,
        sign_digest=fixture.weight_key.sign,
        chain_source=chain_source,
        boot_verifier=verify_boot,
        clock=lambda: datetime(2026, 7, 25, tzinfo=timezone.utc),
    )
    seed = hashlib.sha256(b"leadpoet-compact-rehearsal-hotkey-v1").digest()
    sr25519 = _Sr25519Backend()
    hotkey_public, _hotkey_secret = sr25519.pair_from_seed(seed)
    hotkey = Keypair(public_key=bytes(hotkey_public).hex()).ss58_address
    hotkey_authority = ValidatorHotkeyAuthorityV2(
        boot_identity_supplier=lambda: validator_boot,
        gateway_release_lineage_supplier=lambda: gateway_lineage,
        validator_hotkey=hotkey,
        hotkey_public_key_hex=bytes(hotkey_public).hex(),
        chain_profile=chain_profile,
        sign_receipt_digest=fixture.weight_key.sign,
        attestation_supplier=lambda **_kwargs: b"compact-local-nitro-recipient",
        drand_backend=_DeterministicDrand(),
        chain_source=chain_source,
        sr25519_backend=sr25519,
        boot_verifier=verify_boot,
        clock=lambda: datetime(2026, 7, 25, tzinfo=timezone.utc),
    )
    recipient = hotkey_authority.recipient_request()
    recipient_key = serialization.load_der_public_key(
        base64.b64decode(recipient["recipient_public_key_der_b64"])
    )
    ciphertext = recipient_key.encrypt(
        seed,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    hotkey_authority.provision_seed(
        ciphertext_for_recipient_b64=base64.b64encode(ciphertext).decode("ascii")
    )
    client = _CompactEnclaveClient(
        weight_authority=weight_authority,
        hotkey_authority=hotkey_authority,
    )
    gateway_inputs = {
        "input_receipt_hashes": input_hashes,
        "gateway_authority_event_hash": event_receipt["receipt_hash"],
        "upstream_ancestry_proofs": proofs,
        "upstream_transport_attempts": direct_attempts,
    }
    return {
        "fixture": fixture,
        "cutover": cutover,
        "lineage_id": lineage_id,
        "gateway_boot": gateway_boot,
        "validator_boot": validator_boot,
        "verify_boot": verify_boot,
        "calculation_snapshot": calculation_snapshot,
        "gateway_inputs": gateway_inputs,
        "chain_profile": chain_profile,
        "chain_source": chain_source,
        "current_doc": current_doc,
        "client": client,
        "hotkey": hotkey,
        "hotkey_public": bytes(hotkey_public),
        "allocation_hash": allocation["allocation_component"]["allocation_hash"],
    }


def _publication_coordinator(runtime: Mapping[str, Any], store: _MemoryPostgREST):
    fixture = runtime["fixture"]
    boot = runtime["gateway_boot"]
    lineage_id = runtime["lineage_id"]
    verify_boot = runtime["verify_boot"]

    async def execute(**kwargs: Any) -> dict[str, Any]:
        from gateway.tee.coordinator_executor_v2 import (
            CoordinatorExecutorV2,
            OP_ATTEST_WEIGHT_PUBLICATION,
        )
        from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
        from gateway.research_lab.attested_v2_store import (
            persist_ancestry_checkpoint_v2,
            persist_receipt_graph_v2,
        )

        if (
            kwargs.get("operation") != OP_ATTEST_WEIGHT_PUBLICATION
            or kwargs.get("purpose") != "gateway.weights.publication.v2"
            or int(kwargs.get("epoch_id", -1)) != EPOCH_ID
            or len(kwargs.get("parent_ancestry_proofs") or ()) != 1
        ):
            raise RuntimeError("compact coordinator request differs")
        payload = dict(kwargs["payload"])
        measured = await CoordinatorExecutorV2()(
            OP_ATTEST_WEIGHT_PUBLICATION,
            payload,
            ExecutionContextV2(
                job_id=f"compact-publication:{EPOCH_ID}",
                purpose="gateway.weights.publication.v2",
                epoch_id=EPOCH_ID,
            ),
        )
        measured_doc = dict(measured.output)
        parent_proof = kwargs["parent_ancestry_proofs"][0]
        parent_root = str(
            parent_proof["certificate"]["claim"]["output_root_receipt_hash"]
        )
        receipt = fixture.receipt(
            role="gateway_coordinator",
            purpose="gateway.weights.publication.v2",
            job_id=f"compact-publication:{EPOCH_ID}",
            key=fixture.coordinator_key,
            boot=boot,
            config_hash=boot["config_hash"],
            input_root=sha256_json(payload),
            output_root=sha256_json(measured_doc),
            parents=(parent_root,),
            sequence=0,
        )
        delta = {
            "schema_version": ANCESTRY_DELTA_SCHEMA_VERSION,
            "root_receipt_hash": receipt["receipt_hash"],
            "boot_identities": [boot],
            "receipts": [receipt],
            "transport_attempts": [],
            "host_operations": [],
        }
        certificate = issue_ancestry_certificate_v2(
            local_delta=delta,
            lineage_id=lineage_id,
            certificate_sequence=int(
                parent_proof["certificate"]["claim"]["certificate_sequence"]
            )
            + 1,
            issuer_boot_identity=boot,
            issued_at=NOW,
            sign_digest=fixture.coordinator_key.sign,
            boot_attestation_verifier=verify_boot,
            allowed_issuer_roles={
                "gateway_coordinator",
                "validator_weights",
            },
            parent_proof_disclosures=((parent_proof, parent_root),),
            required_purposes={"gateway.weights.publication.v2"},
        )
        proof = build_compact_ancestry_proof_from_delta_v2(
            delta,
            certificate,
            expected_lineage_id=lineage_id,
            boot_attestation_verifier=verify_boot,
            allowed_issuer_roles={"gateway_coordinator"},
        )
        from leadpoet_canonical.attested_v2 import build_checkpointed_receipt_graph

        graph = build_checkpointed_receipt_graph(
            root_receipt_hash=receipt["receipt_hash"],
            boot_identities=(boot,),
            receipts=(receipt,),
            transport_attempts=(),
            host_operations=(),
            ancestry_lineage_id=lineage_id,
            ancestry_proof=proof,
            boot_attestation_verifier=verify_boot,
            require_boot_attestation_verification=True,
        )
        await persist_receipt_graph_v2(graph)
        await persist_ancestry_checkpoint_v2(
            proof,
            checkpointed_graph=graph,
            expected_lineage_id=lineage_id,
            boot_attestation_verifier=verify_boot,
            allowed_issuer_roles={"gateway_coordinator"},
        )
        return {
            "result": measured_doc,
            "receipt": receipt,
            "receipt_graph": graph,
            "ancestry_compact_proof": proof,
        }

    return execute


def _build_release_authorities(
    runtime: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build production-validated release documents for the strict file/S3 seams."""

    from gateway.tee.release_manifest_v2 import (
        BUILD_EVIDENCE_SCHEMA_VERSION,
        build_release_manifest,
    )
    from gateway.tee.topology import ROLE_SPECS, topology_hash
    from validator_tee.host.release_v2 import (
        build_validator_build_evidence,
        build_validator_release,
        build_validator_release_manifest,
    )

    candidate_sha = str(runtime["gateway_boot"]["commit_sha"])

    def digest(label: str) -> str:
        return sha256_json({"compact_release_fixture": label})

    gateway_evidence: list[dict[str, Any]] = []
    for role, spec in sorted(ROLE_SPECS.items()):
        if role == "gateway_coordinator":
            role_pcr0 = str(runtime["gateway_boot"]["pcr0"])
            execution_manifest_hash = str(
                runtime["gateway_boot"]["build_manifest_hash"]
            )
            dependency_lock_hash = str(
                runtime["gateway_boot"]["dependency_lock_hash"]
            )
        else:
            role_pcr0 = hashlib.sha384(
                f"compact-release-pcr0:{role}".encode("ascii")
            ).hexdigest()
            execution_manifest_hash = digest(f"{role}:execution-manifest")
            dependency_lock_hash = digest(f"{role}:dependency-lock")
        deterministic = {
            "commit_sha": candidate_sha,
            "pcr0": role_pcr0,
            "normalized_image_hash": digest(f"{role}:normalized-image"),
            "source_manifest_hash": digest(f"{role}:source-manifest"),
            "build_identity_hash": digest(f"{role}:build-identity"),
            "execution_manifest_hash": execution_manifest_hash,
            "dependency_lock_hash": dependency_lock_hash,
            "dockerfile_hash": digest(f"{role}:dockerfile"),
            "topology_hash": topology_hash(),
        }
        for builder_domain in ("gateway", "validator"):
            for build_ordinal in (1, 2, 3):
                gateway_evidence.append(
                    {
                        "schema_version": BUILD_EVIDENCE_SCHEMA_VERSION,
                        "builder_domain": builder_domain,
                        "builder_id": f"compact-{builder_domain}-builder",
                        "build_ordinal": build_ordinal,
                        "physical_role": role,
                        "service_role": spec["service_role"],
                        **deterministic,
                        "eif_hash": digest(f"{role}:eif"),
                    }
                )
    gateway_release = build_release_manifest(
        gateway_evidence,
        acceptance_signer_pubkey_hash=digest("acceptance-signer"),
    )

    validator_release = build_validator_release(
        commit_sha=candidate_sha,
        pcr0=str(runtime["validator_boot"]["pcr0"]),
        app_manifest_hash=str(runtime["validator_boot"]["build_manifest_hash"]),
        dependency_lock_hash=str(
            runtime["validator_boot"]["dependency_lock_hash"]
        ),
        normalized_image_hash=digest("validator:normalized-image"),
        eif_hash=digest("validator:eif"),
        dockerfile_hash=digest("validator:dockerfile"),
        base_dockerfile_hash=digest("validator:base-dockerfile"),
    )
    validator_evidence = [
        build_validator_build_evidence(
            validator_release,
            builder_domain=builder_domain,
            builder_id=f"compact-{builder_domain}-builder",
            build_ordinal=build_ordinal,
        )
        for builder_domain in ("gateway", "validator")
        for build_ordinal in (1, 2, 3)
    ]
    return gateway_release, build_validator_release_manifest(validator_evidence)


async def _run_joined(runtime: Mapping[str, Any]) -> dict[str, Any]:
    import gateway.api.weights as weights_api
    import gateway.db.client as db_client_module
    import gateway.research_lab.attested_coordinator_v2 as coordinator_module
    import gateway.research_lab.attested_v2_store as store_module
    import gateway.research_lab.stateful_epoch_authority_v1 as stateful_module
    import gateway.tee.release_lineage_v2 as release_lineage_module
    import gateway.utils.epoch as epoch_module
    import gateway.utils.logger as logger_module
    import neurons.auditor_validator as auditor_module
    import neurons.validator as validator_module
    from gateway.research_lab.attested_v2_store import (
        load_compact_weight_authority_for_identity_v2,
    )
    from leadpoet_canonical.auditor_latest_verified_bundle_v2 import (
        LatestVerifiedBundleStoreV2,
    )
    from leadpoet_canonical.compact_auditor_authority_v2 import (
        verify_compact_published_weight_authority_v2,
    )
    from validator_tee.host.authoritative_weight_flow_v2 import (
        finalize_authoritative_weight_publication_v2 as production_finalize,
        prepare_authoritative_weight_publication_v2 as production_prepare,
    )
    from validator_tee.host.enclave_hotkey_v2 import build_enclave_backed_wallet_v2
    from validator_tee.host.publication_journal_v2 import (
        AuthoritativeWeightPublicationJournalV2,
    )

    memory = _MemoryPostgREST()
    chain_source = runtime["chain_source"]
    client = runtime["client"]
    hotkey = runtime["hotkey"]
    cutover = runtime["cutover"]
    current_doc = runtime["current_doc"]
    cutover_database = _StrictCutoverDatabase(cutover)
    production_persist_post_cutover_evidence = (
        stateful_module.persist_post_cutover_evidence_v1
    )
    production_epoch_evidence_endpoint = (
        weights_api.persist_subnet_epoch_evidence_v1
    )
    epoch_evidence_acknowledgments: list[dict[str, Any]] = []
    compact_finalization_submissions: list[dict[str, Any]] = []
    release_archive_calls: list[str] = []

    gateway_release, validator_release = _build_release_authorities(runtime)

    def load_release_channel(commit: str) -> dict[str, Any]:
        normalized_commit = str(commit).lower()
        if normalized_commit != runtime["gateway_boot"]["commit_sha"]:
            raise RuntimeError("compact release archive commit differs")
        release_archive_calls.append(normalized_commit)
        return {
            "gateway_release_manifest": copy.deepcopy(gateway_release),
            "validator_release_manifest": copy.deepcopy(validator_release),
        }
    best_snapshot = SubnetEpochSnapshot(
        network_genesis_hash=cutover.network_genesis_hash,
        netuid=NETUID,
        head_kind="best",
        block_hash="0x" + "40" * 32,
        current_block=int(current_doc["current_block"]) + 1,
        last_epoch_block=int(current_doc["last_epoch_block"]),
        pending_epoch_at=int(current_doc["pending_epoch_at"]),
        subnet_epoch_index=int(current_doc["subnet_epoch_index"]),
        tempo=int(current_doc["tempo"]),
        blocks_since_last_step=int(current_doc["blocks_since_last_step"]) + 1,
        observed_at=NOW,
    )
    finalized_snapshot = SubnetEpochSnapshot(
        network_genesis_hash=cutover.network_genesis_hash,
        netuid=NETUID,
        head_kind="finalized",
        block_hash=str(current_doc["block_hash"]),
        current_block=int(current_doc["current_block"]),
        last_epoch_block=int(current_doc["last_epoch_block"]),
        pending_epoch_at=int(current_doc["pending_epoch_at"]),
        subnet_epoch_index=int(current_doc["subnet_epoch_index"]),
        tempo=int(current_doc["tempo"]),
        blocks_since_last_step=int(current_doc["blocks_since_last_step"]),
        observed_at=NOW,
    )

    def validate_lifecycle(
        *,
        cutover: SubnetEpochCutover,
        force_refresh: bool,
        network: str,
        netuid: int,
    ) -> dict[str, Any]:
        if (
            cutover.to_dict() != runtime["cutover"].to_dict()
            or not isinstance(force_refresh, bool)
            or network != "finney"
            or int(netuid) != NETUID
        ):
            raise RuntimeError("compact durable lifecycle request differs")
        return {
            "lifecycle_state": "active",
            "mapping_hash": cutover.mapping_hash,
        }

    def validate_archive(value: SubnetEpochCutover) -> None:
        if value.to_dict() != cutover.to_dict():
            raise RuntimeError("compact archive cutover input differs")

    def read_snapshot(
        _subtensor: Any,
        *,
        netuid: int,
        block_number: int | None = None,
        finalized: bool = True,
    ) -> SubnetEpochSnapshot:
        if int(netuid) != NETUID:
            raise RuntimeError("compact chain read netuid differs")
        if block_number is not None:
            if int(block_number) != int(current_doc["current_block"]):
                raise RuntimeError("compact exact-block read differs")
            return finalized_snapshot
        return finalized_snapshot if finalized else best_snapshot

    transparency_hash = sha256_json({"compact": "transparency", "epoch": EPOCH_ID})

    async def log_event(kind: str, document: Mapping[str, Any]) -> dict[str, Any]:
        if kind != "WEIGHT_SUBMISSION_V2" or int(document["epoch_id"]) != EPOCH_ID:
            raise RuntimeError("compact transparency event differs")
        return {"event_hash": transparency_hash.removeprefix("sha256:")}

    async def fetch_inputs(**kwargs: Any) -> dict[str, Any]:
        if (
            kwargs.get("validator_hotkey") != hotkey
            or kwargs.get("allocation_hash") != runtime["allocation_hash"]
            or kwargs.get("calculation_snapshot") != runtime["calculation_snapshot"]
        ):
            raise RuntimeError("compact gateway weight-input request differs")
        return copy.deepcopy(runtime["gateway_inputs"])

    async def gateway_post(
        url: str, payload: Mapping[str, Any], _timeout: float
    ) -> Mapping[str, Any]:
        path = urlparse(url).path
        if path.endswith("/weights/submit/compact/v2"):
            if (
                weights_api.persist_subnet_epoch_evidence_v1
                is not production_epoch_evidence_endpoint
                or stateful_module.persist_post_cutover_evidence_v1
                is not production_persist_post_cutover_evidence
            ):
                raise RuntimeError("compact epoch evidence production path was replaced")
            model = weights_api.CompactWeightSubmissionV2.model_validate(payload)
            acknowledgment = await weights_api.submit_compact_weights_v2(model)
            epoch_ack = acknowledgment.get("epoch_evidence_acknowledgment")
            if not isinstance(epoch_ack, Mapping):
                raise RuntimeError(
                    "real compact epoch evidence acknowledgment is missing"
                )
            epoch_evidence_acknowledgments.append(copy.deepcopy(dict(epoch_ack)))
            return acknowledgment
        if path.endswith("/weights/finalize/compact/v2"):
            model = weights_api.CompactWeightFinalizationV2.model_validate(payload)
            response = await weights_api.finalize_compact_weights_v2(model)
            compact_finalization_submissions.append(
                copy.deepcopy(model.model_dump(mode="python"))
            )
            return response.model_dump(mode="json")
        raise RuntimeError("compact gateway adapter received another endpoint")

    async def prepare_adapter(**kwargs: Any) -> dict[str, Any]:
        return await production_prepare(
            **kwargs,
            fetch_inputs=fetch_inputs,
            post_json=gateway_post,
        )

    async def finalize_adapter(**kwargs: Any) -> dict[str, Any]:
        return await production_finalize(**kwargs, post_json=gateway_post)

    wallet = build_enclave_backed_wallet_v2(
        name="validator_72",
        hotkey_name="default",
        path="/sanitized-public-wallet",
        client=client,
    )
    sdk_substrate = _CompactSDKSubstrate(
        client=client,
        chain_source=chain_source,
        hotkey_public_key=runtime["hotkey_public"],
        profile=runtime["chain_profile"],
        current_block=int(current_doc["current_block"]) + 1,
    )
    subtensor = _CompactSubtensor(
        substrate=sdk_substrate,
        profile=runtime["chain_profile"],
        hotkey_public_key=runtime["hotkey_public"],
        current_doc=current_doc,
    )

    with tempfile.TemporaryDirectory(prefix="compact-weight-joined-") as temp:
        release_manifest_path = Path(temp) / "gateway-v2-release-manifest.json"
        release_manifest_path.write_text(
            json.dumps(gateway_release, sort_keys=True),
            encoding="utf-8",
        )
        journal = AuthoritativeWeightPublicationJournalV2(
            Path(temp) / "publication.json",
            chain_profile=runtime["chain_profile"],
        )
        validator = validator_module.Validator.__new__(validator_module.Validator)
        validator.config = SimpleNamespace(netuid=NETUID)
        validator.wallet = wallet
        validator.subtensor = subtensor
        validator._epoch_cutover = cutover
        validator._weight_publication_journal_v2 = journal
        validator._validator_v2_client = client
        validator._research_lab_allocation_guard_cache = {}

        async def current_epoch(**_kwargs: Any) -> bool:
            return True

        async def open_lifecycle(**_kwargs: Any) -> bool:
            return True

        async def epoch_state() -> Any:
            return SimpleNamespace(workflow_epoch_id=EPOCH_ID)

        validator._weight_submission_epoch_is_current = current_epoch
        validator._weight_submission_lifecycle_is_open = open_lifecycle
        validator._get_epoch_state_async = epoch_state
        validator._get_best_epoch_state_async = epoch_state

        host_result = compute_final_weights(runtime["calculation_snapshot"])
        host_uids = list(host_result["uids"])
        host_weights = list(host_result["weights"])
        chain_source.expected_uids = list(host_result["sparse_uids"])
        chain_source.expected_weights = list(host_result["sparse_weights_u16"])

        coordinator = _publication_coordinator(runtime, memory)
        persistence_defaults = dict(
            production_persist_post_cutover_evidence.__kwdefaults__ or {}
        )
        if set(persistence_defaults) != {
            "persist_graph",
            "load_graph",
            "insert",
            "select",
        }:
            raise RuntimeError("compact stateful persistence seam differs")
        persistence_defaults.update(
            {
                "persist_graph": store_module.persist_receipt_graph_v2,
                "load_graph": store_module.load_receipt_graph_v2,
                "insert": memory.insert_row,
                "select": memory.select_one,
            }
        )
        with ExitStack() as stack:
            stack.enter_context(_patched(store_module, "insert_row", memory.insert_row))
            stack.enter_context(_patched(store_module, "insert_rows", memory.insert_rows))
            stack.enter_context(_patched(store_module, "select_one", memory.select_one))
            stack.enter_context(_patched(store_module, "select_all", memory.select_all))
            stack.enter_context(_patched(store_module, "select_many", memory.select_many))
            stack.enter_context(_patched(store_module, "call_rpc", memory.call_rpc))
            stack.enter_context(
                _patched(
                    store_module,
                    "_ancestry_checkpoint_unknown_commit_sleep",
                    memory.unknown_commit_sleep,
                )
            )
            stack.enter_context(
                _patched(
                    release_lineage_module,
                    "_fetch_historical_release",
                    load_release_channel,
                )
            )
            stack.enter_context(
                _patched(
                    release_lineage_module,
                    "verify_boot_identity_nitro",
                    runtime["verify_boot"],
                )
            )
            stack.enter_context(
                _patched(
                    db_client_module,
                    "get_write_client",
                    cutover_database.get_write_client,
                )
            )
            stack.enter_context(
                _patched(
                    db_client_module,
                    "create_http1_sync_client",
                    cutover_database.create_http1_sync_client,
                )
            )
            stack.enter_context(
                _patched(epoch_module, "_validated_cutover_authority_hash", None)
            )
            stack.enter_context(_patched(epoch_module, "_cutover_state_cache", None))
            stack.enter_context(
                _patched(
                    epoch_module,
                    "validate_epoch_runtime_lifecycle",
                    validate_lifecycle,
                )
            )
            stack.enter_context(_patched(weights_api, "PRIMARY_VALIDATOR_HOTKEYS", {hotkey}))
            stack.enter_context(_patched(weights_api, "ALLOWED_NETUIDS", {NETUID}))
            stack.enter_context(_patched(weights_api, "EXPECTED_CHAIN", EXPECTED_CHAIN))
            stack.enter_context(
                _patched(
                    weights_api,
                    "load_subnet_epoch_cutover",
                    lambda: cutover,
                )
            )
            stack.enter_context(
                _patched(
                    weights_api,
                    "validate_cutover_anchor_from_archive",
                    validate_archive,
                )
            )
            stack.enter_context(_patched(weights_api, "get_subtensor", lambda: object()))
            stack.enter_context(_patched(weights_api, "read_subnet_epoch_snapshot", read_snapshot))
            stack.enter_context(
                _patched(
                    weights_api,
                    "_verify_authoritative_v2_boot",
                    runtime["verify_boot"],
                )
            )
            stack.enter_context(
                _patched(
                    production_persist_post_cutover_evidence,
                    "__kwdefaults__",
                    persistence_defaults,
                )
            )
            stack.enter_context(_patched(logger_module, "log_event", log_event))
            stack.enter_context(
                _patched(
                    coordinator_module,
                    "execute_coordinator_v2",
                    coordinator,
                )
            )
            stack.enter_context(
                _patched(
                    validator_module,
                    "prepare_authoritative_weight_publication_v2",
                    prepare_adapter,
                )
            )
            stack.enter_context(
                _patched(
                    validator_module,
                    "finalize_authoritative_weight_publication_v2",
                    finalize_adapter,
                )
            )
            stack.enter_context(_environment({
                "VALIDATOR_V2_GATEWAY_URL": "https://gateway.rehearsal.invalid",
                "EXPECTED_CHAIN": EXPECTED_CHAIN,
                "BITTENSOR_NETWORK": "finney",
                "BITTENSOR_NETUID": str(NETUID),
                "SUPABASE_URL": cutover_database.SERVICE_URL,
                "SUPABASE_SERVICE_ROLE_KEY": cutover_database.SERVICE_KEY,
                "GATEWAY_V2_RELEASE_MANIFEST": str(release_manifest_path),
                "GITHUB_SHA": _candidate_sha(),
            }))
            succeeded = await validator._authorize_and_set_weights_v2(
                epoch_state=SimpleNamespace(
                    subnet_epoch_index=EPOCH_ID,
                    epoch_block=int(current_doc["epoch_block"]),
                ),
                snapshot=runtime["calculation_snapshot"],
                host_uids=host_uids,
                host_weights=host_weights,
                allocation_hash=runtime["allocation_hash"],
                leaderboard_window_start="2026-07-24T00:00:00Z",
                leaderboard_window_end="2026-07-25T00:00:00Z",
            )
            if not succeeded:
                raise RuntimeError("production compact primary lifecycle returned false")
            durable = await load_compact_weight_authority_for_identity_v2(
                netuid=NETUID,
                epoch_id=EPOCH_ID,
                validator_hotkey=hotkey,
            )
            if not isinstance(durable, Mapping) or durable.get("authority_stage") != "finalized":
                raise RuntimeError("compact finalized authority is not durable")
            if len(compact_finalization_submissions) != 1:
                raise RuntimeError(
                    "compact initial finalization submission count differs"
                )
            durable_tables_before_recovery = copy.deepcopy(memory.tables)
            durable_insert_count_before_recovery = int(memory.insert_count)
            checkpoint_writes_before_recovery = copy.deepcopy(
                memory.checkpoint_rpc_writes
            )
            response = await weights_api.get_compact_published_weights_v2(
                NETUID,
                EPOCH_ID,
                _make_request(),
            )
            authority_body = bytes(response.body)
            if json.loads(authority_body) != durable:
                raise RuntimeError("compact GET bytes differ from durable authority")

            def compact_verify(value: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
                return verify_compact_published_weight_authority_v2(
                    value,
                    identity_cache=kwargs["identity_cache"],
                    chain_signing_profile=kwargs["chain_signing_profile"],
                    expected_lineage_id=kwargs["expected_lineage_id"],
                    expected_chain=kwargs["expected_chain"],
                    boot_verifier=runtime["verify_boot"],
                )

            identity_cache = {
                "schema_version": "leadpoet.independent_pcr0_identities.v2",
                "entries": [
                    {
                        "physical_role": boot["physical_role"],
                        "role": boot["role"],
                        "commit_sha": boot["commit_sha"],
                        "pcr0": boot["pcr0"],
                        "build_manifest_hash": boot["build_manifest_hash"],
                        "dependency_lock_hash": boot["dependency_lock_hash"],
                        "verified_build_count": 3,
                    }
                    for boot in (runtime["gateway_boot"], runtime["validator_boot"])
                ],
            }
            audit_hashes = []
            auditor_submission_states = []
            with _AuthorityHTTPServer(authority_body) as server, _patched(
                auditor_module,
                "verify_compact_published_weight_authority_v2",
                compact_verify,
            ), _patched(auditor_module, "read_subnet_epoch_snapshot", read_snapshot):
                for index in range(2):
                    auditor = auditor_module.AuditorValidator.__new__(
                        auditor_module.AuditorValidator
                    )
                    auditor.config = SimpleNamespace(
                        netuid=NETUID,
                        subtensor=SimpleNamespace(network="finney"),
                    )
                    auditor.gateway_url = server.url
                    auditor.epoch_cutover = cutover
                    auditor.epoch_archive_endpoint = "local://archive-boundary"
                    auditor.epoch_archive_subtensor = object()
                    auditor.uid = index + 1
                    auditor.wallet = SimpleNamespace(
                        hotkey=Keypair.create_from_seed(
                            hashlib.sha256(
                                f"compact-auditor-{index}".encode("ascii")
                            ).hexdigest()
                        )
                    )
                    auditor._verified_bundle_store = LatestVerifiedBundleStoreV2(
                        Path(temp) / f"auditor-{index}.json"
                    )
                    auditor.last_submitted_epoch = None
                    auditor.last_authority_epoch = None
                    auditor._submission_lock = None
                    fetched = await auditor.fetch_attested_weights_v2(EPOCH_ID)
                    if fetched != durable:
                        raise RuntimeError("auditor compact HTTP bytes differ")
                    verified = auditor.verify_attested_weights_v2(
                        fetched,
                        identity_cache=identity_cache,
                    )
                    if verified is None:
                        raise RuntimeError("production auditor rejected compact authority")
                    auditor._persist_latest_verified_bundle(
                        authority=fetched,
                        identity_cache=identity_cache,
                        verified_bundle=verified,
                    )
                    loaded, _record = auditor._load_and_reverify_latest_verified_bundle(
                        submission_epoch_id=EPOCH_ID
                    )
                    if (
                        list(loaded["uids"]) != chain_source.expected_uids
                        or list(loaded["weights_u16"])
                        != chain_source.expected_weights
                    ):
                        raise RuntimeError(
                            "auditor compact vector differs from primary broadcast"
                        )
                    baseline_last_update = (
                        int(current_doc["last_epoch_block"]) - 1
                    )
                    auditor_chain = _AuditorSubtensor(
                        auditor_uid=index + 1,
                        wallet=auditor.wallet,
                        expected_uids=chain_source.expected_uids,
                        expected_weights_u16=chain_source.expected_weights,
                        baseline_last_update=baseline_last_update,
                        finalized_last_update=int(current_doc["current_block"]) + 1,
                    )
                    auditor.subtensor = auditor_chain
                    submitted = await auditor._submit_verified_authority_once(
                        source_epoch_id=EPOCH_ID,
                        submission_epoch_id=EPOCH_ID,
                        bundle=loaded,
                        submission_mode="current_epoch_verified",
                    )
                    finalized_state = (
                        auditor._read_finalized_weight_submission_state()
                    )
                    expected_pairs = list(
                        zip(
                            chain_source.expected_uids,
                            chain_source.expected_weights,
                        )
                    )
                    if (
                        submitted is not True
                        or auditor.last_submitted_epoch != EPOCH_ID
                        or auditor.last_authority_epoch != EPOCH_ID
                        or auditor_chain.set_calls != 1
                        or int(finalized_state["last_update"])
                        <= baseline_last_update
                        or finalized_state["weights"] != expected_pairs
                    ):
                        raise RuntimeError(
                            "production auditor submission/readback differs"
                        )
                    auditor_submission_states.append(
                        {
                            "uid": index + 1,
                            "last_update": int(finalized_state["last_update"]),
                            "vector_hash": sha256_json(
                                {
                                    "uids": chain_source.expected_uids,
                                    "weights_u16": chain_source.expected_weights,
                                }
                            ),
                            "storage_reads": list(
                                auditor_chain.substrate.reads
                            ),
                        }
                    )
                    audit_hashes.append(
                        sha256_json(
                            {
                                "authority_bytes_hash": sha256_bytes(
                                    authority_body
                                ),
                                "uids": loaded["uids"],
                                "weights_u16": loaded["weights_u16"],
                            }
                        )
                    )
            if len(set(audit_hashes)) != 1:
                raise RuntimeError("primary/auditor compact authority bytes diverged")
            if (
                len(auditor_submission_states) != 2
                or len(
                    {
                        item["vector_hash"]
                        for item in auditor_submission_states
                    }
                )
                != 1
            ):
                raise RuntimeError("auditor chain submission evidence diverged")

            restarted = validator_module.Validator.__new__(validator_module.Validator)
            restarted.config = validator.config
            restarted.wallet = wallet
            restarted.subtensor = subtensor
            restarted._epoch_cutover = cutover
            restarted._weight_publication_journal_v2 = journal
            restarted._validator_v2_client = client
            restarted._weight_submission_epoch_is_current = current_epoch
            restarted._weight_submission_lifecycle_is_open = open_lifecycle
            restarted._get_epoch_state_async = epoch_state
            restarted._get_best_epoch_state_async = epoch_state
            recovered = await restarted._recover_weight_publication_before_new_authority_v2(
                epoch_id=EPOCH_ID,
                gateway_url="https://gateway.rehearsal.invalid",
            )
            if recovered is not True or journal.load() is None:
                raise RuntimeError("compact same-epoch journal recovery failed")
            if len(compact_finalization_submissions) != 2:
                raise RuntimeError(
                    "compact fresh-scan recovery submission count differs"
                )
            initial_finalization, recovery_finalization = (
                compact_finalization_submissions
            )
            if (
                initial_finalization["finalization"]
                != recovery_finalization["finalization"]
                or initial_finalization["weight_submission_event_hash"]
                != recovery_finalization["weight_submission_event_hash"]
                or initial_finalization["ancestry_commitment"]
                != recovery_finalization["ancestry_commitment"]
                or initial_finalization["validator_receipt_delta"]
                == recovery_finalization["validator_receipt_delta"]
                or initial_finalization["validator_ancestry_proof"]
                == recovery_finalization["validator_ancestry_proof"]
                or initial_finalization["compact_finalization_hash"]
                == recovery_finalization["compact_finalization_hash"]
            ):
                raise RuntimeError(
                    "compact fresh-scan semantic/wrapper identity differs"
                )
            expected_finalization_job_ids = [
                "weight-finalization:%d:%s"
                % (EPOCH_ID, scan_id[len("sha256:") :])
                for scan_id in chain_source.finalization_scan_ids
            ]
            if (
                len(chain_source.finalization_scan_ids) < 2
                or len(set(chain_source.finalization_scan_ids))
                != len(chain_source.finalization_scan_ids)
                or chain_source.finalization_job_ids
                != expected_finalization_job_ids
            ):
                raise RuntimeError(
                    "compact finalization job identity is not scan-derived"
                )
            if (
                memory.insert_count != durable_insert_count_before_recovery
                or memory.tables != durable_tables_before_recovery
                or memory.checkpoint_rpc_writes
                != checkpoint_writes_before_recovery
            ):
                raise RuntimeError(
                    "compact fresh-scan recovery appended duplicate durable evidence"
                )

            mismatched_recovery = copy.deepcopy(recovery_finalization)
            mismatched_recovery["finalization"]["state_transition_hash"] = (
                sha256_json({"mismatch": "state-transition"})
            )
            mismatched_recovery["compact_finalization_hash"] = sha256_json(
                {
                    field: value
                    for field, value in mismatched_recovery.items()
                    if field != "compact_finalization_hash"
                }
            )
            try:
                await weights_api.finalize_compact_weights_v2(
                    weights_api.CompactWeightFinalizationV2.model_validate(
                        mismatched_recovery
                    )
                )
            except HTTPException as exc:
                if exc.status_code != 409:
                    raise RuntimeError(
                        "compact mismatched recovery did not remain a conflict"
                    ) from exc
            else:
                raise RuntimeError("compact mismatched recovery was accepted")
            if (
                memory.insert_count != durable_insert_count_before_recovery
                or memory.tables != durable_tables_before_recovery
                or memory.checkpoint_rpc_writes
                != checkpoint_writes_before_recovery
            ):
                raise RuntimeError(
                    "compact mismatched recovery appended durable evidence"
                )
            next_epoch = EPOCH_ID + 1

            async def next_state() -> Any:
                return SimpleNamespace(workflow_epoch_id=next_epoch)

            restarted._get_epoch_state_async = next_state
            restarted._get_best_epoch_state_async = next_state
            recovered_next = await restarted._recover_weight_publication_before_new_authority_v2(
                epoch_id=next_epoch,
                gateway_url="https://gateway.rehearsal.invalid",
            )
            if recovered_next is not False or journal.load() is not None:
                raise RuntimeError("compact next-epoch journal retirement failed")

        final_verified = verify_compact_published_weight_authority_v2(
            durable,
            identity_cache=identity_cache,
            chain_signing_profile=runtime["chain_profile"],
            expected_lineage_id=runtime["lineage_id"],
            expected_chain=EXPECTED_CHAIN,
            boot_verifier=runtime["verify_boot"],
        )
        if (
            list(final_verified["uids"]) != chain_source.expected_uids
            or list(final_verified["weights_u16"])
            != chain_source.expected_weights
        ):
            raise RuntimeError(
                "durable compact vector differs from primary broadcast"
            )
        vector_hash = sha256_json(
            {
                "uids": chain_source.expected_uids,
                "weights_u16": chain_source.expected_weights,
            }
        )
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", vector_hash):
            raise RuntimeError("compact primary/auditor vector hash is invalid")
        if any(
            not re.fullmatch(r"sha256:[0-9a-f]{64}", str(item["vector_hash"]))
            or item["vector_hash"] != vector_hash
            for item in auditor_submission_states
        ):
            raise RuntimeError("compact auditor vector hash evidence is invalid")

        unknown_commit_root = memory.unknown_commit_root
        unknown_commit_writes = memory.checkpoint_rpc_writes.get(
            str(unknown_commit_root),
            0,
        )
        expected_delays = list(
            store_module._ANCESTRY_CHECKPOINT_UNKNOWN_COMMIT_BACKOFF_SECONDS[
                :2
            ]
        )
        checkpoint_rows = [
            row
            for row in memory.tables.get(
                store_module.ANCESTRY_CHECKPOINT_TABLE,
                [],
            )
            if row.get("root_receipt_hash") == unknown_commit_root
        ]
        activation_rows = [
            row
            for row in memory.tables.get(
                store_module.ANCESTRY_ACTIVATION_TABLE,
                [],
            )
            if row.get("activation_root_receipt_hash")
            == unknown_commit_root
        ]
        authority_rows = memory.tables.get(
            store_module.COMPACT_WEIGHT_AUTHORITY_TABLE,
            [],
        )
        authority_stages = [
            str(row.get("authority_stage") or "") for row in authority_rows
        ]
        if (
            not isinstance(unknown_commit_root, str)
            or not unknown_commit_root.startswith("sha256:")
            or unknown_commit_writes != 1
            or memory.unknown_commit_readbacks != 3
            or memory.unknown_commit_sleep_delays != expected_delays
            or memory.unknown_commit_visible is not True
            or len(checkpoint_rows) != 1
            or len(activation_rows) != 1
            or authority_stages.count("published") != 1
            or authority_stages.count("finalized") != 1
            or len(authority_rows) != 2
            or {
                str(row.get("bundle_hash") or "") for row in authority_rows
            }
            != {str(final_verified["bundle_hash"])}
        ):
            raise RuntimeError(
                "compact ancestry unknown-commit recovery evidence differs"
            )

        cutover_database.assert_complete()
        if len(epoch_evidence_acknowledgments) != 1:
            raise RuntimeError("real compact epoch evidence execution count differs")
        epoch_ack = epoch_evidence_acknowledgments[0]
        snapshot_rows = memory.tables.get(stateful_module.SNAPSHOT_TABLE, [])
        boundary_rows = memory.tables.get(stateful_module.BOUNDARY_TABLE, [])
        expected_readback_hash = sha256_json(
            {
                "boundary": None,
                "snapshot": snapshot_rows[0] if len(snapshot_rows) == 1 else None,
                "receipt_graph_hash": epoch_ack.get("receipt_graph_hash"),
            }
        )
        if (
            len(snapshot_rows) != 1
            or boundary_rows
            or epoch_ack.get("durable_readback_hash")
            != expected_readback_hash
        ):
            raise RuntimeError("real compact epoch evidence readback differs")
        if release_archive_calls != [runtime["gateway_boot"]["commit_sha"]]:
            raise RuntimeError("compact release archive boundary count differs")
        return {
            "production_allocation_guard": True,
            "production_primary_compact_lifecycle": True,
            "gateway_compact_submit_persist_get_finalize": True,
            "real_epoch_evidence_endpoint": True,
            "stateful_epoch_evidence_persisted": True,
            "stateful_epoch_evidence_readback_exact": True,
            "cutover_authority_db_boundary_exact": True,
            "release_lineage_file_archive_boundary_exact": True,
            "compact_ancestry_checkpoint_persistence": True,
            "ancestry_unknown_commit_recovered_read_only": True,
            "ancestry_unknown_commit_rpc_write_count": unknown_commit_writes,
            "ancestry_unknown_commit_readback_count": (
                memory.unknown_commit_readbacks
            ),
            "single_canonical_publish_finalize_after_unknown_commit": True,
            "primary_auditor_byte_identity": True,
            "independent_auditor_count": 2,
            "independent_auditor_submission_count": len(
                auditor_submission_states
            ),
            "auditor_submission_success": True,
            "auditor_last_update_advanced": True,
            "auditor_finalized_vector_readback_equal": True,
            "auditor_submission_states": auditor_submission_states,
            "gateway_authority_bytes_hash": sha256_bytes(authority_body),
            "primary_auditor_vector_hash": vector_hash,
            "auditor_verified_cache_replay": True,
            "same_epoch_compact_journal_recovered": True,
            "same_epoch_compact_fresh_scan_recovered": True,
            "compact_finalization_job_ids_scan_derived": True,
            "compact_fresh_scan_recovery_writes": 0,
            "compact_mismatched_recovery_conflict": True,
            "next_epoch_compact_journal_retired": True,
            "bundle_hash": final_verified["bundle_hash"],
            "weights_hash": final_verified["weights_hash"],
            "extrinsic_hash": final_verified["extrinsic_hash"],
            "finalized_block": final_verified["finalized_block"],
            "durable_store_insert_count": memory.insert_count,
            "real_chain_broadcast_adapted": True,
            "physical_chain_last_update_vector_readback_unadaptable": True,
        }


def exercise_compact_weight_joined_path() -> dict[str, Any]:
    candidate_sha = _candidate_sha()
    with _environment(
        {"REHEARSAL_CANDIDATE_SHA": candidate_sha}
    ), _owned_event_loop() as event_loop, _sdk_weight_module_boundary() as sdk_boundary:
        allocation = _allocation_guard(candidate_sha)
        runtime = _build_runtime(candidate_sha, allocation)
        evidence = event_loop.run_until_complete(_run_joined(runtime))
        evidence["sdk_weight_module_boundary"] = dict(sdk_boundary)
        return evidence


if __name__ == "__main__":
    print(json.dumps(exercise_compact_weight_joined_path(), sort_keys=True))
