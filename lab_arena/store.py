"""Arena durable-state access (labarena.md sections 11 and 15.1).

The service reaches the six ``lab_arena_*`` tables only through the
dedicated ``lab_arena_service`` role over PostgREST: every write is one of
the SECURITY DEFINER functions in ``scripts/178-lab-arena-v1.sql`` and reads
are plain selects. The HTTP/1.1-pinned client construction is copied from
``gateway/db/client.py`` (never imported: the validator enclave image copies
that file). The operator-minted JWT is sent as the ``Authorization`` header
and the anon key as ``apikey``; the project service key is never held.

``PsycopgTransport`` exists for tests and local tooling only: it calls the
same SQL functions through a PostgreSQL driver so disposable-PostgreSQL
coverage exercises the exact production function contract.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import httpx

from lab_arena.contracts import (
    ArenaContractError,
    LEASE_TTL_SECONDS,
    canonical_json,
    require_sha256,
)

WHOAMI_SCHEMA_VERSION = "leadpoet.lab_arena.whoami.v1"
SERVICE_ROLE_NAME = "lab_arena_service"

# Parameter order and PostgreSQL casts for every service-callable function.
# PostgREST matches JSON keys to parameter names; the psycopg transport uses
# named notation with explicit casts so both transports share this table.
FUNCTION_SIGNATURES: Dict[str, Sequence[tuple]] = {
    "lab_arena_whoami": (),
    "lab_arena_create_round": (("p_round_id", "text"), ("p_configuration_hash", "text"), ("p_configuration_doc", "jsonb")),
    "lab_arena_transition_round": (("p_round_id", "text"), ("p_expected_status", "text"), ("p_next_status", "text"), ("p_patch", "jsonb")),
    "lab_arena_append_journal_entry": (("p_round_id", "text"), ("p_entry", "jsonb")),
    "lab_arena_register_submission": (("p_round_id", "text"), ("p_submission_id", "text"), ("p_miner_hotkey", "text"), ("p_doc", "jsonb")),
    "lab_arena_update_submission": (("p_round_id", "text"), ("p_submission_id", "text"), ("p_expected_status", "text"), ("p_next_status", "text"), ("p_patch", "jsonb")),
    "lab_arena_upsert_account_credential": (("p_miner_hotkey", "text"), ("p_ciphertext", "text"), ("p_key_hash", "text"), ("p_preflight", "jsonb")),
    "lab_arena_record_preflight": (("p_miner_hotkey", "text"), ("p_preflight", "jsonb")),
    "lab_arena_credit_deposit": (("p_miner_hotkey", "text"), ("p_payment_reference", "text"), ("p_amount_microusd", "bigint"), ("p_deposit_doc", "jsonb")),
    "lab_arena_open_stage": (("p_round_id", "text"), ("p_stage", "smallint"), ("p_participants", "jsonb"), ("p_icp_positions", "integer[]"), ("p_icp_hashes", "text[]"), ("p_per_icp_cap_microusd", "bigint")),
    "lab_arena_claim_assignment": (("p_round_id", "text"), ("p_runner_hotkey", "text"), ("p_declared_parallelism", "integer"), ("p_slot_ceiling", "integer"), ("p_excluded_miner_hotkeys", "text[]"), ("p_request_id", "text"), ("p_request_hash", "text"), ("p_lease_token_hash", "text"), ("p_lease_ttl_seconds", "integer")),
    "lab_arena_reserve_call": (("p_run_id", "text"), ("p_lease_token_hash", "text"), ("p_call_identity", "text"), ("p_operation_id", "text"), ("p_provider", "text"), ("p_funding_source", "text"), ("p_amount_microusd", "bigint"), ("p_call_doc", "jsonb"), ("p_lease_ttl_seconds", "integer")),
    "lab_arena_mark_dispatched": (("p_run_id", "text"), ("p_lease_token_hash", "text"), ("p_call_identity", "text")),
    "lab_arena_settle_call": (("p_run_id", "text"), ("p_lease_token_hash", "text"), ("p_call_identity", "text"), ("p_actual_microusd", "bigint"), ("p_terminal_response", "jsonb"), ("p_event", "jsonb"), ("p_lease_ttl_seconds", "integer")),
    "lab_arena_mark_uncertain": (("p_run_id", "text"), ("p_lease_token_hash", "text"), ("p_call_identity", "text"), ("p_call_doc", "jsonb"), ("p_event", "jsonb"), ("p_lease_ttl_seconds", "integer")),
    "lab_arena_append_events": (("p_run_id", "text"), ("p_lease_token_hash", "text"), ("p_events", "jsonb"), ("p_lease_ttl_seconds", "integer")),
    "lab_arena_complete_attempt": (("p_run_id", "text"), ("p_lease_token_hash", "text"), ("p_receipt", "jsonb"), ("p_receipt_hash", "text"), ("p_terminal_cause", "text"), ("p_output_hash", "text"), ("p_output_ref", "text"), ("p_provider_call_root", "text"), ("p_private_event_root", "text"), ("p_cost_root", "text")),
    "lab_arena_expire_leases": (("p_round_id", "text"),),
    "lab_arena_close_stage": (("p_round_id", "text"), ("p_stage", "smallint")),
    "lab_arena_cancel_round": (("p_round_id", "text"), ("p_reason", "text")),
    "lab_arena_record_run_scores": (("p_round_id", "text"), ("p_stage", "smallint"), ("p_scores", "jsonb")),
}

TABLES = (
    "lab_arena_rounds",
    "lab_arena_submissions",
    "lab_arena_runs",
    "lab_arena_events",
    "lab_arena_accounts",
    "lab_arena_ledger",
)


DEADLOCK_SQLSTATE = "40P01"
DEADLOCK_RETRIES = 3


class ArenaStoreError(RuntimeError):
    """A durable-state operation failed. Messages never carry credentials."""


class ArenaRoleError(ArenaStoreError):
    """The database identity is not the least-privilege Arena service role."""


def new_lease_token() -> str:
    return secrets.token_hex(32)


def hash_lease_token(token: str) -> str:
    return "sha256:" + hashlib.sha256(str(token).encode("utf-8")).hexdigest()


def _check_filter_value(value: Any) -> str:
    text = str(value)
    if any(ch in text for ch in ",.()\"'\n\r\t") or len(text) > 200:
        raise ArenaStoreError("filter value contains reserved characters")
    return text


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------


class StoreTransport:
    def rpc(self, function: str, params: Mapping[str, Any]) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    def select(
        self,
        table: str,
        *,
        filters: Optional[Mapping[str, Any]] = None,
        order: Optional[str] = None,
        descending: bool = False,
        limit: Optional[int] = None,
        columns: str = "*",
    ) -> List[Dict[str, Any]]:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        return None


def create_http1_client(timeout_seconds: float) -> httpx.Client:
    """The HTTP/1.1-pinned construction copied from ``gateway/db/client.py``.

    postgrest-py enables HTTP/2 by default, but its shared HPACK encoder is
    not safe when multiple threads encode headers concurrently. HTTP/1.1 keeps
    connection pooling and parallel requests without that shared table.
    """

    return httpx.Client(
        http1=True,
        http2=False,
        timeout=httpx.Timeout(float(timeout_seconds)),
        follow_redirects=True,
    )


class PostgrestTransport(StoreTransport):
    """PostgREST over HTTP/1.1 with the operator-minted service JWT."""

    def __init__(
        self,
        base_url: str,
        *,
        anon_key: str,
        service_jwt: str,
        timeout_seconds: float = 8.0,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        if not base_url.startswith("https://") and not base_url.startswith("http://127.0.0.1") and not base_url.startswith("http://localhost"):
            raise ArenaStoreError("PostgREST base URL must be https (or loopback for tests)")
        if not anon_key or not service_jwt:
            raise ArenaStoreError("PostgREST anon key and service JWT are required")
        if service_jwt.count(".") != 2:
            raise ArenaStoreError("service JWT has an invalid shape")
        self._base_url = base_url.rstrip("/")
        self._headers = {
            "apikey": anon_key,
            "Authorization": "Bearer " + service_jwt,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self._client = http_client or create_http1_client(timeout_seconds)
        self.deadlock_retries = 0

    def __repr__(self) -> str:  # never expose the token
        return "PostgrestTransport(%r)" % self._base_url

    def _raise_for_status(self, response: httpx.Response, context: str) -> None:
        if 200 <= response.status_code < 300:
            return
        detail = ""
        try:
            body = response.json()
            if isinstance(body, dict):
                detail = " code=%s message=%s" % (body.get("code"), str(body.get("message") or "")[:200])
        except ValueError:
            detail = ""
        raise ArenaStoreError("%s failed: HTTP %d%s" % (context, response.status_code, detail))

    def rpc(self, function: str, params: Mapping[str, Any]) -> Any:
        if function not in FUNCTION_SIGNATURES:
            raise ArenaStoreError("unknown Arena function")
        content = canonical_json(dict(params)).encode("utf-8")
        for attempt in range(DEADLOCK_RETRIES + 1):
            try:
                response = self._client.post("%s/rest/v1/rpc/%s" % (self._base_url, function), headers=self._headers, content=content)
            except httpx.HTTPError as exc:
                raise ArenaStoreError("rpc %s transport failure: %s" % (function, type(exc).__name__)) from exc
            if response.status_code >= 400 and attempt < DEADLOCK_RETRIES:
                try:
                    code = response.json().get("code")
                except ValueError:
                    code = None
                if code == DEADLOCK_SQLSTATE:
                    self.deadlock_retries += 1
                    time.sleep(0.01 * (attempt + 1))
                    continue
            self._raise_for_status(response, "rpc %s" % function)
            try:
                return response.json()
            except ValueError as exc:
                raise ArenaStoreError("rpc %s returned non-JSON" % function) from exc
        raise ArenaStoreError("rpc %s failed after deadlock retries" % function)

    def select(self, table, *, filters=None, order=None, descending=False, limit=None, columns="*"):
        if table not in TABLES:
            raise ArenaStoreError("unknown Arena table")
        query: List[tuple] = [("select", columns)]
        for key, value in (filters or {}).items():
            query.append((key, "eq." + _check_filter_value(value)))
        if order:
            query.append(("order", "%s.%s" % (order, "desc" if descending else "asc")))
        if limit is not None:
            query.append(("limit", str(int(limit))))
        try:
            response = self._client.get(
                "%s/rest/v1/%s" % (self._base_url, table),
                headers=self._headers,
                params=query,
            )
        except httpx.HTTPError as exc:
            raise ArenaStoreError("select %s transport failure: %s" % (table, type(exc).__name__)) from exc
        self._raise_for_status(response, "select %s" % table)
        rows = response.json()
        if not isinstance(rows, list):
            raise ArenaStoreError("select %s returned a non-list" % table)
        return rows

    def close(self) -> None:
        self._client.close()


class PsycopgTransport(StoreTransport):
    """Test/local transport calling the same functions through psycopg2.

    Connections come from a bounded pool so concurrency tests that model
    separate service instances never exhaust the server. ``role`` is applied
    with ``SET ROLE`` on every connection so the least-privilege grants are
    exercised.
    """

    def __init__(self, connect: Callable[[], Any], *, role: Optional[str] = SERVICE_ROLE_NAME, pool_size: int = 6) -> None:
        import queue

        self._connect = connect
        self._role = role
        self._pool_size = max(1, int(pool_size))
        self._idle: "queue.LifoQueue[Any]" = queue.LifoQueue()
        self._created = 0
        self._lock = threading.Lock()
        self._connections: List[Any] = []
        self._closed = False
        self.deadlock_retries = 0
        self.last_deadlock_detail = ""

    def _acquire(self):
        import queue

        with self._lock:
            if self._closed:
                raise ArenaStoreError("transport is closed")
        try:
            return self._idle.get_nowait()
        except queue.Empty:
            pass
        with self._lock:
            if self._created < self._pool_size:
                self._created += 1
                create = True
            else:
                create = False
        if create:
            try:
                connection = self._connect()
                connection.autocommit = True
                if self._role:
                    with connection.cursor() as cursor:
                        cursor.execute("SET ROLE %s" % self._role)
            except Exception:
                with self._lock:
                    self._created -= 1
                raise
            with self._lock:
                self._connections.append(connection)
            return connection
        return self._idle.get(timeout=120)

    def _release(self, connection: Any) -> None:
        if getattr(connection, "closed", 0):
            with self._lock:
                self._created -= 1
            return
        self._idle.put(connection)

    def rpc(self, function: str, params: Mapping[str, Any]) -> Any:
        signature = FUNCTION_SIGNATURES.get(function)
        if signature is None:
            raise ArenaStoreError("unknown Arena function")
        names = [name for name, _ in signature]
        unknown = set(params) - set(names)
        if unknown:
            raise ArenaStoreError("unknown parameters for %s: %s" % (function, sorted(unknown)))
        placeholders = []
        values = []
        for name, cast in signature:
            if name not in params:
                continue
            value = params[name]
            if cast == "jsonb":
                value = json.dumps(value, sort_keys=True) if value is not None else None
            placeholders.append("%s => %%s::%s" % (name, cast))
            values.append(value)
        sql = "SELECT public.%s(%s)" % (function, ", ".join(placeholders))
        for attempt in range(DEADLOCK_RETRIES + 1):
            connection = self._acquire()
            try:
                try:
                    with connection.cursor() as cursor:
                        cursor.execute(sql, values)
                        row = cursor.fetchone()
                    return row[0] if row else None
                except Exception as exc:  # psycopg2 errors carry no secrets here
                    if getattr(exc, "pgcode", None) == DEADLOCK_SQLSTATE and attempt < DEADLOCK_RETRIES:
                        # The aborted call changed nothing; every Arena function is
                        # idempotent by identity, so one bounded retry is safe.
                        self.deadlock_retries += 1
                        self.last_deadlock_detail = (getattr(getattr(exc, "diag", None), "message_detail", None) or "")[:2000]
                        time.sleep(0.01 * (attempt + 1))
                        continue
                    diag = getattr(exc, "diag", None)
                    detail = (getattr(diag, "message_detail", None) or "")[:2000]
                    context = (getattr(diag, "context", None) or "")[:600]
                    raise ArenaStoreError("rpc %s failed: %s%s%s" % (function, str(exc).splitlines()[0][:200], (" [" + detail + "]") if detail else "", (" {" + context + "}") if context else "")) from exc
            finally:
                self._release(connection)
        raise ArenaStoreError("rpc %s failed after deadlock retries" % function)

    def select(self, table, *, filters=None, order=None, descending=False, limit=None, columns="*"):
        if table not in TABLES:
            raise ArenaStoreError("unknown Arena table")
        clauses = []
        values: List[Any] = []
        for key, value in (filters or {}).items():
            if not key.replace("_", "").isalnum():
                raise ArenaStoreError("invalid filter column")
            clauses.append("%s = %%s" % key)
            values.append(value)
        sql = "SELECT row_to_json(t) FROM (SELECT %s FROM public.%s" % (columns, table)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        if order:
            if not order.replace("_", "").isalnum():
                raise ArenaStoreError("invalid order column")
            sql += " ORDER BY %s %s" % (order, "DESC" if descending else "ASC")
        if limit is not None:
            sql += " LIMIT %d" % int(limit)
        sql += ") t"
        connection = self._acquire()
        try:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(sql, values)
                    return [row[0] for row in cursor.fetchall()]
            except Exception as exc:
                raise ArenaStoreError("select %s failed: %s" % (table, str(exc).splitlines()[0][:200])) from exc
        finally:
            self._release(connection)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            connections = list(self._connections)
            self._connections = []
        for connection in connections:
            try:
                connection.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def _require_mapping(value: Any, context: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArenaStoreError("%s returned a non-object result" % context)
    return dict(value)


class ArenaStore:
    """Typed wrapper over the Arena functions and reads (section 15.1)."""

    def __init__(self, transport: StoreTransport, *, lease_ttl_seconds: int = LEASE_TTL_SECONDS) -> None:
        self._transport = transport
        self._lease_ttl_seconds = int(lease_ttl_seconds)

    # -- identity ---------------------------------------------------------

    def whoami(self) -> Dict[str, Any]:
        return _require_mapping(self._transport.rpc("lab_arena_whoami", {}), "whoami")

    def require_service_role(self) -> Dict[str, Any]:
        """Refuse to run unless the role is ``lab_arena_service`` without superuser/BYPASSRLS."""

        identity = self.whoami()
        if identity.get("schema_version") != WHOAMI_SCHEMA_VERSION:
            raise ArenaRoleError("whoami schema mismatch")
        if identity.get("current_user") != SERVICE_ROLE_NAME:
            raise ArenaRoleError("database role is not %s" % SERVICE_ROLE_NAME)
        if identity.get("rolsuper") is not False or identity.get("rolbypassrls") is not False:
            raise ArenaRoleError("database role must not be superuser or BYPASSRLS")
        if identity.get("rolcanlogin") is not False:
            raise ArenaRoleError("database role must be NOLOGIN")
        return identity

    # -- rounds -----------------------------------------------------------

    def create_round(self, round_id: str, configuration: Mapping[str, Any]) -> Dict[str, Any]:
        configuration_hash = require_sha256(configuration.get("configuration_hash"), "configuration_hash")
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_create_round",
                {"p_round_id": round_id, "p_configuration_hash": configuration_hash, "p_configuration_doc": dict(configuration)},
            ),
            "create_round",
        )

    def transition_round(self, round_id: str, expected_status: str, next_status: str, patch: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_transition_round",
                {"p_round_id": round_id, "p_expected_status": expected_status, "p_next_status": next_status, "p_patch": dict(patch or {})},
            ),
            "transition_round",
        )

    def append_journal_entry(self, round_id: str, entry: Mapping[str, Any]) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc("lab_arena_append_journal_entry", {"p_round_id": round_id, "p_entry": dict(entry)}),
            "append_journal_entry",
        )

    def get_round(self, round_id: str) -> Optional[Dict[str, Any]]:
        rows = self._transport.select("lab_arena_rounds", filters={"round_id": round_id}, limit=1)
        return rows[0] if rows else None

    def list_rounds(self, *, status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        filters = {"status": status} if status else None
        return self._transport.select("lab_arena_rounds", filters=filters, order="created_at", descending=True, limit=limit)

    def published_reward_bases(self, *, limit: int = 200) -> List[Dict[str, Any]]:
        rows = self._transport.select(
            "lab_arena_rounds",
            filters={"status": "published"},
            order="effective_reward_epoch",
            descending=True,
            limit=limit,
            columns="round_id,effective_reward_epoch,king_outcome,king_hotkey,king_start_epoch,reward_basis_hash,reward_basis_doc,result_bundle_hash,published_at",
        )
        return rows

    # -- submissions ------------------------------------------------------

    def register_submission(self, round_id: str, submission_id: str, miner_hotkey: str, doc: Mapping[str, Any]) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_register_submission",
                {"p_round_id": round_id, "p_submission_id": submission_id, "p_miner_hotkey": miner_hotkey, "p_doc": dict(doc)},
            ),
            "register_submission",
        )

    def update_submission(self, round_id: str, submission_id: str, expected_status: str, next_status: str, patch: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_update_submission",
                {"p_round_id": round_id, "p_submission_id": submission_id, "p_expected_status": expected_status, "p_next_status": next_status, "p_patch": dict(patch or {})},
            ),
            "update_submission",
        )

    def get_submission(self, submission_id: str) -> Optional[Dict[str, Any]]:
        rows = self._transport.select("lab_arena_submissions", filters={"submission_id": submission_id}, limit=1)
        return rows[0] if rows else None

    def list_submissions(self, round_id: str, *, status: Optional[str] = None) -> List[Dict[str, Any]]:
        filters: Dict[str, Any] = {"round_id": round_id}
        if status:
            filters["status"] = status
        return self._transport.select("lab_arena_submissions", filters=filters, order="created_at")

    # -- accounts and deposits --------------------------------------------

    def upsert_account_credential(self, miner_hotkey: str, ciphertext: str, key_hash: str, preflight: Mapping[str, Any]) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_upsert_account_credential",
                {"p_miner_hotkey": miner_hotkey, "p_ciphertext": ciphertext, "p_key_hash": key_hash, "p_preflight": dict(preflight)},
            ),
            "upsert_account_credential",
        )

    def record_preflight(self, miner_hotkey: str, preflight: Mapping[str, Any]) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc("lab_arena_record_preflight", {"p_miner_hotkey": miner_hotkey, "p_preflight": dict(preflight)}),
            "record_preflight",
        )

    def credit_deposit(self, *, miner_hotkey: str, payment_reference: str, amount_microusd: int, deposit_doc: Mapping[str, Any]) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_credit_deposit",
                {"p_miner_hotkey": miner_hotkey, "p_payment_reference": payment_reference, "p_amount_microusd": int(amount_microusd), "p_deposit_doc": dict(deposit_doc)},
            ),
            "credit_deposit",
        )

    def get_account(self, miner_hotkey: str) -> Optional[Dict[str, Any]]:
        rows = self._transport.select("lab_arena_accounts", filters={"miner_hotkey": miner_hotkey}, limit=1)
        return rows[0] if rows else None

    # -- stages and assignments -------------------------------------------

    def open_stage(self, round_id: str, stage: int, participants: Sequence[Mapping[str, Any]], icp_positions: Sequence[int], icp_hashes: Sequence[str], per_icp_cap_microusd: int) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_open_stage",
                {
                    "p_round_id": round_id,
                    "p_stage": int(stage),
                    "p_participants": [dict(item) for item in participants],
                    "p_icp_positions": [int(item) for item in icp_positions],
                    "p_icp_hashes": [str(item) for item in icp_hashes],
                    "p_per_icp_cap_microusd": int(per_icp_cap_microusd),
                },
            ),
            "open_stage",
        )

    def claim_assignment(
        self,
        *,
        round_id: str,
        runner_hotkey: str,
        declared_parallelism: int,
        slot_ceiling: int,
        excluded_miner_hotkeys: Sequence[str],
        request_id: str,
        request_hash: str,
        lease_token_hash: str,
        lease_ttl_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_claim_assignment",
                {
                    "p_round_id": round_id,
                    "p_runner_hotkey": runner_hotkey,
                    "p_declared_parallelism": int(declared_parallelism),
                    "p_slot_ceiling": int(slot_ceiling),
                    "p_excluded_miner_hotkeys": list(excluded_miner_hotkeys),
                    "p_request_id": request_id,
                    "p_request_hash": request_hash,
                    "p_lease_token_hash": lease_token_hash,
                    "p_lease_ttl_seconds": int(lease_ttl_seconds or self._lease_ttl_seconds),
                },
            ),
            "claim_assignment",
        )

    def reserve_call(self, *, run_id: str, lease_token_hash: str, call_identity: str, operation_id: str, provider: str, funding_source: str, amount_microusd: int, call_doc: Mapping[str, Any], lease_ttl_seconds: Optional[int] = None) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_reserve_call",
                {
                    "p_run_id": run_id,
                    "p_lease_token_hash": lease_token_hash,
                    "p_call_identity": call_identity,
                    "p_operation_id": operation_id,
                    "p_provider": provider,
                    "p_funding_source": funding_source,
                    "p_amount_microusd": int(amount_microusd),
                    "p_call_doc": dict(call_doc),
                    "p_lease_ttl_seconds": int(lease_ttl_seconds or self._lease_ttl_seconds),
                },
            ),
            "reserve_call",
        )

    def mark_dispatched(self, *, run_id: str, lease_token_hash: str, call_identity: str) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_mark_dispatched",
                {"p_run_id": run_id, "p_lease_token_hash": lease_token_hash, "p_call_identity": call_identity},
            ),
            "mark_dispatched",
        )

    def settle_call(self, *, run_id: str, lease_token_hash: str, call_identity: str, actual_microusd: int, terminal_response: Mapping[str, Any], event: Optional[Mapping[str, Any]], lease_ttl_seconds: Optional[int] = None) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_settle_call",
                {
                    "p_run_id": run_id,
                    "p_lease_token_hash": lease_token_hash,
                    "p_call_identity": call_identity,
                    "p_actual_microusd": int(actual_microusd),
                    "p_terminal_response": dict(terminal_response),
                    "p_event": dict(event) if event is not None else None,
                    "p_lease_ttl_seconds": int(lease_ttl_seconds or self._lease_ttl_seconds),
                },
            ),
            "settle_call",
        )

    def mark_uncertain(self, *, run_id: str, lease_token_hash: str, call_identity: str, call_doc: Mapping[str, Any], event: Optional[Mapping[str, Any]], lease_ttl_seconds: Optional[int] = None) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_mark_uncertain",
                {
                    "p_run_id": run_id,
                    "p_lease_token_hash": lease_token_hash,
                    "p_call_identity": call_identity,
                    "p_call_doc": dict(call_doc),
                    "p_event": dict(event) if event is not None else None,
                    "p_lease_ttl_seconds": int(lease_ttl_seconds or self._lease_ttl_seconds),
                },
            ),
            "mark_uncertain",
        )

    def append_events(self, *, run_id: str, lease_token_hash: str, events: Sequence[Mapping[str, Any]], lease_ttl_seconds: Optional[int] = None) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_append_events",
                {
                    "p_run_id": run_id,
                    "p_lease_token_hash": lease_token_hash,
                    "p_events": [dict(item) for item in events],
                    "p_lease_ttl_seconds": int(lease_ttl_seconds or self._lease_ttl_seconds),
                },
            ),
            "append_events",
        )

    def complete_attempt(self, *, run_id: str, lease_token_hash: str, receipt: Mapping[str, Any], receipt_hash: str, terminal_cause: str, output_hash: str, output_ref: str, provider_call_root: str, private_event_root: str, cost_root: str) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_complete_attempt",
                {
                    "p_run_id": run_id,
                    "p_lease_token_hash": lease_token_hash,
                    "p_receipt": dict(receipt),
                    "p_receipt_hash": receipt_hash,
                    "p_terminal_cause": terminal_cause,
                    "p_output_hash": output_hash,
                    "p_output_ref": output_ref,
                    "p_provider_call_root": provider_call_root,
                    "p_private_event_root": private_event_root,
                    "p_cost_root": cost_root,
                },
            ),
            "complete_attempt",
        )

    def expire_leases(self, round_id: str) -> Dict[str, Any]:
        return _require_mapping(self._transport.rpc("lab_arena_expire_leases", {"p_round_id": round_id}), "expire_leases")

    def close_stage(self, round_id: str, stage: int) -> Dict[str, Any]:
        return _require_mapping(self._transport.rpc("lab_arena_close_stage", {"p_round_id": round_id, "p_stage": int(stage)}), "close_stage")

    def cancel_round(self, round_id: str, reason: str) -> Dict[str, Any]:
        return _require_mapping(self._transport.rpc("lab_arena_cancel_round", {"p_round_id": round_id, "p_reason": reason}), "cancel_round")

    def record_run_scores(self, round_id: str, stage: int, scores: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_record_run_scores",
                {"p_round_id": round_id, "p_stage": int(stage), "p_scores": [dict(item) for item in scores]},
            ),
            "record_run_scores",
        )

    # -- reads ------------------------------------------------------------

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        rows = self._transport.select("lab_arena_runs", filters={"run_id": run_id}, limit=1)
        return rows[0] if rows else None

    def list_runs(self, round_id: str, *, stage: Optional[int] = None, status: Optional[str] = None, submission_id: Optional[str] = None) -> List[Dict[str, Any]]:
        filters: Dict[str, Any] = {"round_id": round_id}
        if stage is not None:
            filters["stage"] = int(stage)
        if status:
            filters["status"] = status
        if submission_id:
            filters["submission_id"] = submission_id
        return self._transport.select("lab_arena_runs", filters=filters, order="run_id")

    def list_events(self, run_id: str) -> List[Dict[str, Any]]:
        return self._transport.select("lab_arena_events", filters={"run_id": run_id}, order="sequence")

    def list_ledger(self, *, run_id: Optional[str] = None, call_identity: Optional[str] = None, miner_hotkey: Optional[str] = None) -> List[Dict[str, Any]]:
        filters: Dict[str, Any] = {}
        if run_id:
            filters["run_id"] = run_id
        if call_identity:
            filters["call_identity"] = call_identity
        if miner_hotkey:
            filters["miner_hotkey"] = miner_hotkey
        return self._transport.select("lab_arena_ledger", filters=filters or None, order="entry_id")

    def close(self) -> None:
        self._transport.close()
