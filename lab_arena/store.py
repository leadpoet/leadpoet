"""Arena durable-state access (labarena.md sections 11 and 15.1).

The service reaches the four ``lab_arena_*`` tables only through the
dedicated ``lab_arena_service`` role over PostgREST: every write is one of
the SECURITY DEFINER functions in ``scripts/179-lab-arena-v1.sql`` and reads
are plain selects. The HTTP/1.1-pinned client construction is copied from
``gateway/db/client.py`` (never imported: the validator enclave image copies
that file). The preferred scoped ``sb_secret_`` API key is sent only as
``apikey``. Legacy parity environments can still use an operator-minted JWT.

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
from urllib.parse import urlsplit

import httpx

from lab_arena.contracts import (
    ArenaContractError,
    LEASE_TTL_SECONDS,
    canonical_json,
)

WHOAMI_SCHEMA_VERSION = "leadpoet.lab_arena.whoami.v1"
SERVICE_ROLE_NAME = "lab_arena_service"

# Parameter order and PostgreSQL casts for every service-callable function.
# PostgREST matches JSON keys to parameter names; the psycopg transport uses
# named notation with explicit casts so both transports share this table.
SCORE_BATCH_SIZE = 500

FUNCTION_SIGNATURES: Dict[str, Sequence[tuple]] = {
    "lab_arena_whoami": (),
    "lab_arena_schema_version_v1": (),
    "lab_arena_weight_state_schema_v1": (),
    "lab_arena_current_daily_icp_set": (("p_set_id", "bigint"),),
    "lab_arena_commit_round_v2": (
        ("p_round_id", "text"),
        ("p_participants", "jsonb"),
        ("p_benchmark_ref", "text"),
        ("p_evaluation_date", "text"),
        ("p_icp_set_date", "date"),
        ("p_scorer_image_digest", "text"),
        ("p_scorer_image_reference", "text"),
    ),
    "lab_arena_create_round": (("p_round_id", "text"), ("p_configuration_doc", "jsonb")),
    "lab_arena_transition_round": (("p_round_id", "text"), ("p_expected_status", "text"), ("p_next_status", "text"), ("p_patch", "jsonb")),
    "lab_arena_activate_reward": (("p_round_id", "text"), ("p_reward_basis", "jsonb"), ("p_signing_key_doc", "jsonb")),
    "lab_arena_prepare_promotion": (("p_round_id", "text"), ("p_plan", "jsonb")),
    "lab_arena_complete_promotion": (("p_round_id", "text"), ("p_plan", "jsonb")),
    "lab_arena_register_submission": (("p_round_id", "text"), ("p_submission_id", "text"), ("p_miner_hotkey", "text"), ("p_doc", "jsonb")),
    "lab_arena_update_submission": (("p_round_id", "text"), ("p_submission_id", "text"), ("p_expected_status", "text"), ("p_next_status", "text"), ("p_patch", "jsonb")),
    "lab_arena_accept_submission_with_credentials": (("p_round_id", "text"), ("p_submission_id", "text"), ("p_miner_hotkey", "text"), ("p_credentials", "jsonb")),
    "lab_arena_get_submission_credential": (("p_submission_id", "text"), ("p_miner_hotkey", "text"), ("p_provider", "text")),
    "lab_arena_open_stage": (("p_round_id", "text"), ("p_stage", "smallint"), ("p_participants", "jsonb"), ("p_icp_positions", "integer[]")),
    "lab_arena_claim_assignment": (("p_round_id", "text"), ("p_runner_hotkey", "text"), ("p_declared_parallelism", "integer"), ("p_slot_ceiling", "integer"), ("p_excluded_miner_hotkeys", "text[]"), ("p_request_id", "text"), ("p_request_hash", "text"), ("p_lease_token_hash", "text"), ("p_lease_ttl_seconds", "integer")),
    "lab_arena_reserve_call": (("p_run_id", "text"), ("p_lease_token_hash", "text"), ("p_call_identity", "text"), ("p_operation_id", "text"), ("p_provider", "text"), ("p_funding_source", "text"), ("p_amount_microusd", "bigint"), ("p_call_doc", "jsonb"), ("p_lease_ttl_seconds", "integer")),
    "lab_arena_mark_dispatched": (("p_run_id", "text"), ("p_lease_token_hash", "text"), ("p_call_identity", "text")),
    "lab_arena_settle_call": (("p_run_id", "text"), ("p_lease_token_hash", "text"), ("p_call_identity", "text"), ("p_actual_microusd", "bigint"), ("p_terminal_response", "jsonb"), ("p_lease_ttl_seconds", "integer")),
    "lab_arena_mark_uncertain": (("p_run_id", "text"), ("p_lease_token_hash", "text"), ("p_call_identity", "text"), ("p_call_doc", "jsonb"), ("p_lease_ttl_seconds", "integer")),
    "lab_arena_complete_attempt": (("p_run_id", "text"), ("p_lease_token_hash", "text"), ("p_result", "jsonb"), ("p_terminal_cause", "text"), ("p_output_ref", "text")),
    "lab_arena_expire_leases": (("p_round_id", "text"),),
    "lab_arena_close_stage": (("p_round_id", "text"), ("p_stage", "smallint")),
    "lab_arena_open_scoring": (("p_round_id", "text"), ("p_stage", "smallint"), ("p_work_items", "jsonb")),
    "lab_arena_close_scoring": (("p_round_id", "text"), ("p_stage", "smallint")),
    "lab_arena_cancel_round": (("p_round_id", "text"), ("p_reason", "text")),
    "lab_arena_record_run_scores": (("p_round_id", "text"), ("p_stage", "smallint"), ("p_scores", "jsonb")),
    "lab_arena_publish_weight_state_v1": (("p_network", "text"), ("p_netuid", "integer"), ("p_epoch", "bigint"), ("p_state_hash", "text"), ("p_state_doc", "jsonb")),
    "lab_arena_record_chain_outcome_v1": (("p_network", "text"), ("p_netuid", "integer"), ("p_epoch", "bigint"), ("p_validator_hotkey", "text"), ("p_request_id", "text"), ("p_extrinsic_hash", "text"), ("p_outcome_doc", "jsonb")),
}

TABLES = (
    "lab_arena_rounds",
    "lab_arena_submissions",
    "lab_arena_runs",
    "lab_arena_ledger",
    "lab_arena_accepted_weight_states",
    "lab_arena_chain_outcomes",
)
ROUND_MODE_FILTER = "configuration_doc->>mode"
PROMOTION_OUTCOME_FILTER = "publication_doc->king_decision->>outcome"
ROUND_NETWORK_COLUMN = "arena_network_name"
ROUND_NETUID_COLUMN = "arena_netuid"


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
        offset: Optional[int] = None,
        status_in: Optional[Sequence[str]] = None,
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
        follow_redirects=False,
        trust_env=False,
    )


class PostgrestTransport(StoreTransport):
    """PostgREST over HTTP/1.1 with one least-privilege credential."""

    def __init__(
        self,
        base_url: str,
        *,
        anon_key: str,
        service_jwt: str = "",
        service_key: str = "",
        timeout_seconds: float = 8.0,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        parsed = urlsplit(base_url)
        is_https = parsed.scheme == "https" and bool(parsed.hostname)
        is_loopback_http = parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if (
            not (is_https or is_loopback_http)
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ArenaStoreError("PostgREST base URL must be https (or loopback for tests)")
        if bool(service_key) == bool(service_jwt):
            raise ArenaStoreError("exactly one PostgREST service credential is required")
        if service_key and not service_key.startswith("sb_secret_"):
            raise ArenaStoreError("scoped service key has an invalid shape")
        if service_jwt and (not anon_key or service_jwt.count(".") != 2):
            raise ArenaStoreError("anon key and valid service JWT are required")
        self._base_url = base_url.rstrip("/")
        self._headers = {
            "apikey": service_key or anon_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if service_jwt:
            self._headers["Authorization"] = "Bearer " + service_jwt
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

    def select(
        self,
        table,
        *,
        filters=None,
        order=None,
        descending=False,
        limit=None,
        offset=None,
        status_in=None,
        columns="*",
    ):
        if table not in TABLES:
            raise ArenaStoreError("unknown Arena table")
        query: List[tuple] = [("select", columns)]
        for key, value in (filters or {}).items():
            if key not in (ROUND_MODE_FILTER, PROMOTION_OUTCOME_FILTER) and not key.replace("_", "").isalnum():
                raise ArenaStoreError("invalid filter column")
            query.append((key, "is.null" if value is None else "eq." + _check_filter_value(value)))
        if status_in is not None:
            if table != "lab_arena_rounds" or "status" in (filters or {}):
                raise ArenaStoreError("status inclusion filter is invalid")
            statuses = tuple(_check_filter_value(value) for value in status_in)
            if not statuses:
                raise ArenaStoreError("status inclusion filter is empty")
            query.append(("status", "in.(%s)" % ",".join(statuses)))
        if order:
            query.append(("order", "%s.%s" % (order, "desc" if descending else "asc")))
        if limit is not None:
            query.append(("limit", str(int(limit))))
        if offset is not None:
            offset = int(offset)
            if offset < 0:
                raise ArenaStoreError("offset must be nonnegative")
            query.append(("offset", str(offset)))
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

    def select(
        self,
        table,
        *,
        filters=None,
        order=None,
        descending=False,
        limit=None,
        offset=None,
        status_in=None,
        columns="*",
    ):
        if table not in TABLES:
            raise ArenaStoreError("unknown Arena table")
        clauses = []
        values: List[Any] = []
        for key, value in (filters or {}).items():
            if key == ROUND_MODE_FILTER:
                expression = "configuration_doc ->> 'mode'"
            elif key == PROMOTION_OUTCOME_FILTER:
                expression = "publication_doc #>> '{king_decision,outcome}'"
            elif key.replace("_", "").isalnum():
                expression = key
            else:
                raise ArenaStoreError("invalid filter column")
            if value is None:
                clauses.append("%s IS NULL" % expression)
            else:
                clauses.append("%s = %%s" % expression)
                values.append(value)
        if status_in is not None:
            if table != "lab_arena_rounds" or "status" in (filters or {}):
                raise ArenaStoreError("status inclusion filter is invalid")
            statuses = tuple(_check_filter_value(value) for value in status_in)
            if not statuses:
                raise ArenaStoreError("status inclusion filter is empty")
            clauses.append("status = ANY(%s)")
            values.append(list(statuses))
        sql = "SELECT row_to_json(t) FROM (SELECT %s FROM public.%s" % (columns, table)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        if order:
            if not order.replace("_", "").isalnum():
                raise ArenaStoreError("invalid order column")
            sql += " ORDER BY %s %s" % (order, "DESC" if descending else "ASC")
        if limit is not None:
            sql += " LIMIT %d" % int(limit)
        if offset is not None:
            offset = int(offset)
            if offset < 0:
                raise ArenaStoreError("offset must be nonnegative")
            sql += " OFFSET %d" % offset
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

    def current_daily_icp_set(self, set_id: int) -> Dict[str, Any]:
        """Read only the active UTC-day ICP set exposed to the Arena."""

        return _require_mapping(
            self._transport.rpc(
                "lab_arena_current_daily_icp_set", {"p_set_id": int(set_id)}
            ),
            "current_daily_icp_set",
        )

    # -- accepted weight state ------------------------------------------

    def get_weight_state(self, network: str, netuid: int, epoch: int) -> Optional[Dict[str, Any]]:
        rows = self._transport.select(
            "lab_arena_accepted_weight_states",
            filters={"network": str(network), "netuid": int(netuid), "epoch": int(epoch)},
            limit=2,
            columns="network,netuid,epoch,state_hash,state_doc,created_at",
        )
        if len(rows) > 1:
            raise ArenaStoreError("multiple accepted weight states exist for one epoch")
        return rows[0] if rows else None

    def publish_weight_state(self, network: str, netuid: int, epoch: int, state_hash: str, state_doc: Mapping[str, Any]) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_publish_weight_state_v1",
                {"p_network": str(network), "p_netuid": int(netuid), "p_epoch": int(epoch), "p_state_hash": str(state_hash), "p_state_doc": dict(state_doc)},
            ),
            "publish_weight_state",
        )

    def record_chain_outcome(self, *, network: str, netuid: int, epoch: int, validator_hotkey: str, request_id: str, extrinsic_hash: str, outcome_doc: Mapping[str, Any]) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_record_chain_outcome_v1",
                {"p_network": str(network), "p_netuid": int(netuid), "p_epoch": int(epoch), "p_validator_hotkey": str(validator_hotkey), "p_request_id": str(request_id), "p_extrinsic_hash": str(extrinsic_hash), "p_outcome_doc": dict(outcome_doc)},
            ),
            "record_chain_outcome",
        )

    def list_chain_outcomes(self, network: str, netuid: int, epoch: int) -> List[Dict[str, Any]]:
        return self._transport.select(
            "lab_arena_chain_outcomes",
            filters={"network": str(network), "netuid": int(netuid), "epoch": int(epoch)},
            order="created_at",
            descending=False,
            limit=200,
            columns="network,netuid,epoch,validator_hotkey,request_id,extrinsic_hash,outcome_doc,created_at",
        )

    def get_chain_outcome(self, network: str, netuid: int, epoch: int, validator_hotkey: str, request_id: str) -> Optional[Dict[str, Any]]:
        rows = self._transport.select(
            "lab_arena_chain_outcomes",
            filters={"network": str(network), "netuid": int(netuid), "epoch": int(epoch), "validator_hotkey": str(validator_hotkey), "request_id": str(request_id)},
            limit=1,
            columns="network,netuid,epoch,validator_hotkey,request_id,extrinsic_hash,outcome_doc,created_at",
        )
        return rows[0] if rows else None

    # -- rounds -----------------------------------------------------------

    def create_round(self, round_id: str, configuration: Mapping[str, Any]) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_create_round",
                {"p_round_id": round_id, "p_configuration_doc": dict(configuration)},
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

    def commit_round_v2(
        self,
        round_id: str,
        *,
        participants: Sequence[Mapping[str, Any]],
        benchmark_ref: str,
        evaluation_date: str,
        icp_set_date: str,
        scorer_image_digest: str,
        scorer_image_reference: str,
    ) -> Dict[str, Any]:
        """Atomically commit an explicit Day 0 bank date for new-policy rounds."""

        return _require_mapping(
            self._transport.rpc(
                "lab_arena_commit_round_v2",
                {
                    "p_round_id": round_id,
                    "p_participants": [dict(item) for item in participants],
                    "p_benchmark_ref": benchmark_ref,
                    "p_evaluation_date": evaluation_date,
                    "p_icp_set_date": icp_set_date,
                    "p_scorer_image_digest": scorer_image_digest,
                    "p_scorer_image_reference": scorer_image_reference,
                },
            ),
            "commit_round_v2",
        )

    def activate_reward(self, round_id: str, reward_basis: Mapping[str, Any], signing_key_doc: Mapping[str, Any]) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_activate_reward",
                {"p_round_id": round_id, "p_reward_basis": dict(reward_basis), "p_signing_key_doc": dict(signing_key_doc)},
            ),
            "activate_reward",
        )

    def prepare_promotion(self, round_id: str, plan: Mapping[str, Any]) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_prepare_promotion",
                {"p_round_id": round_id, "p_plan": dict(plan)},
            ),
            "prepare_promotion",
        )

    def complete_promotion(self, round_id: str, plan: Mapping[str, Any]) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_complete_promotion",
                {"p_round_id": round_id, "p_plan": dict(plan)},
            ),
            "complete_promotion",
        )

    def pending_promotions(
        self, *, pinned_round_id: Optional[str] = None,
        network_name: Optional[str] = None, netuid: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        if (network_name is None) != (netuid is None):
            raise ArenaStoreError("round network filters must be supplied together")
        filters: Dict[str, Any] = {
            "status": "published",
            ROUND_MODE_FILTER: "live",
            "promotion_required": True,
            "baseline_promoted_at": None,
            PROMOTION_OUTCOME_FILTER: "crowned",
        }
        if pinned_round_id is not None:
            filters["round_id"] = pinned_round_id
        if network_name is not None:
            filters[ROUND_NETWORK_COLUMN] = str(network_name)
            filters[ROUND_NETUID_COLUMN] = int(netuid)
        rows = self._transport.select(
            "lab_arena_rounds",
            filters=filters,
            order="created_at",
            descending=False,
            limit=limit,
            columns="round_id,publication_doc,promotion_doc,published_at,created_at",
        )
        return rows

    def get_round(self, round_id: str) -> Optional[Dict[str, Any]]:
        rows = self._transport.select("lab_arena_rounds", filters={"round_id": round_id}, limit=1)
        return rows[0] if rows else None

    def list_rounds(
        self,
        *,
        status: Optional[str] = None,
        statuses: Optional[Sequence[str]] = None,
        mode: Optional[str] = None,
        network_name: Optional[str] = None,
        netuid: Optional[int] = None,
        limit: int = 100,
        offset: Optional[int] = None,
        columns: str = "*",
    ) -> List[Dict[str, Any]]:
        if status is not None and statuses is not None:
            raise ArenaStoreError("round status filters are mutually exclusive")
        if (network_name is None) != (netuid is None):
            raise ArenaStoreError("round network filters must be supplied together")
        filters: Dict[str, Any] = {}
        if status is not None:
            filters["status"] = status
        if mode is not None:
            filters[ROUND_MODE_FILTER] = mode
        if network_name is not None:
            filters[ROUND_NETWORK_COLUMN] = str(network_name)
            filters[ROUND_NETUID_COLUMN] = int(netuid)
        return self._transport.select(
            "lab_arena_rounds",
            filters=filters or None,
            status_in=statuses,
            order="created_at",
            descending=True,
            limit=limit,
            offset=offset,
            columns=columns,
        )

    def published_reward_bases(
        self, *, mode: Optional[str] = None,
        network_name: Optional[str] = None, netuid: Optional[int] = None,
        limit: int = 200
    ) -> List[Dict[str, Any]]:
        if (network_name is None) != (netuid is None):
            raise ArenaStoreError("round network filters must be supplied together")
        filters: Dict[str, Any] = {"status": "published"}
        if mode is not None:
            filters[ROUND_MODE_FILTER] = mode
        if network_name is not None:
            filters[ROUND_NETWORK_COLUMN] = str(network_name)
            filters[ROUND_NETUID_COLUMN] = int(netuid)
        rows = self._transport.select(
            "lab_arena_rounds",
            filters=filters,
            order="effective_reward_epoch",
            descending=True,
            limit=limit,
            columns="round_id,effective_reward_epoch,king_outcome,king_hotkey,king_start_epoch,reward_basis_hash,reward_basis_doc,reward_activated_at,published_at,configuration_doc",
        )
        return [row for row in rows if row.get("reward_activated_at") and row.get("reward_basis_doc")]

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

    def accept_submission_with_credentials(
        self,
        round_id: str,
        submission_id: str,
        miner_hotkey: str,
        encrypted_credentials: Mapping[str, str],
    ) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_accept_submission_with_credentials",
                {
                    "p_round_id": round_id,
                    "p_submission_id": submission_id,
                    "p_miner_hotkey": miner_hotkey,
                    "p_credentials": dict(encrypted_credentials),
                },
            ),
            "accept_submission_with_credentials",
        )

    def get_submission_credential(
        self, submission_id: str, miner_hotkey: str, provider: str
    ) -> Optional[Dict[str, Any]]:
        result = _require_mapping(
            self._transport.rpc(
                "lab_arena_get_submission_credential",
                {
                    "p_submission_id": submission_id,
                    "p_miner_hotkey": miner_hotkey,
                    "p_provider": provider,
                },
            ),
            "get_submission_credential",
        )
        return result if result.get("status") == "available" else None

    def get_submission(self, submission_id: str) -> Optional[Dict[str, Any]]:
        rows = self._transport.select("lab_arena_submissions", filters={"submission_id": submission_id}, limit=1)
        return rows[0] if rows else None

    def list_submissions(
        self,
        round_id: str,
        *,
        status: Optional[str] = None,
        columns: str = "*",
    ) -> List[Dict[str, Any]]:
        filters: Dict[str, Any] = {"round_id": round_id}
        if status:
            filters["status"] = status
        return self._transport.select(
            "lab_arena_submissions",
            filters=filters,
            order="created_at",
            columns=columns,
        )

    # -- stages and assignments -------------------------------------------

    def open_stage(self, round_id: str, stage: int, participants: Sequence[Mapping[str, Any]], icp_positions: Sequence[int]) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_open_stage",
                {
                    "p_round_id": round_id,
                    "p_stage": int(stage),
                    "p_participants": [dict(item) for item in participants],
                    "p_icp_positions": [int(item) for item in icp_positions],
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

    def settle_call(self, *, run_id: str, lease_token_hash: str, call_identity: str, actual_microusd: int, terminal_response: Mapping[str, Any], lease_ttl_seconds: Optional[int] = None) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_settle_call",
                {
                    "p_run_id": run_id,
                    "p_lease_token_hash": lease_token_hash,
                    "p_call_identity": call_identity,
                    "p_actual_microusd": int(actual_microusd),
                    "p_terminal_response": dict(terminal_response),
                    "p_lease_ttl_seconds": int(lease_ttl_seconds or self._lease_ttl_seconds),
                },
            ),
            "settle_call",
        )

    def mark_uncertain(self, *, run_id: str, lease_token_hash: str, call_identity: str, call_doc: Mapping[str, Any], lease_ttl_seconds: Optional[int] = None) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_mark_uncertain",
                {
                    "p_run_id": run_id,
                    "p_lease_token_hash": lease_token_hash,
                    "p_call_identity": call_identity,
                    "p_call_doc": dict(call_doc),
                    "p_lease_ttl_seconds": int(lease_ttl_seconds or self._lease_ttl_seconds),
                },
            ),
            "mark_uncertain",
        )

    def complete_attempt(self, *, run_id: str, lease_token_hash: str, result: Mapping[str, Any], terminal_cause: str, output_ref: str) -> Dict[str, Any]:
        return _require_mapping(
            self._transport.rpc(
                "lab_arena_complete_attempt",
                {
                    "p_run_id": run_id,
                    "p_lease_token_hash": lease_token_hash,
                    "p_result": dict(result),
                    "p_terminal_cause": terminal_cause,
                    "p_output_ref": output_ref,
                },
            ),
            "complete_attempt",
        )

    def expire_leases(self, round_id: str) -> Dict[str, Any]:
        return _require_mapping(self._transport.rpc("lab_arena_expire_leases", {"p_round_id": round_id}), "expire_leases")

    def close_stage(self, round_id: str, stage: int) -> Dict[str, Any]:
        return _require_mapping(self._transport.rpc("lab_arena_close_stage", {"p_round_id": round_id, "p_stage": int(stage)}), "close_stage")

    def open_scoring(self, round_id: str, stage: int, work_items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        """Turn the committed scoring plan into claimable scoring assignments (one per work item)."""

        return _require_mapping(
            self._transport.rpc("lab_arena_open_scoring", {"p_round_id": round_id, "p_stage": int(stage), "p_work_items": [dict(item) for item in work_items]}),
            "open_scoring",
        )

    def close_scoring(self, round_id: str, stage: int) -> Dict[str, Any]:
        return _require_mapping(self._transport.rpc("lab_arena_close_scoring", {"p_round_id": round_id, "p_stage": int(stage)}), "close_scoring")

    def cancel_round(self, round_id: str, reason: str) -> Dict[str, Any]:
        return _require_mapping(self._transport.rpc("lab_arena_cancel_round", {"p_round_id": round_id, "p_reason": reason}), "cancel_round")

    def record_run_scores(self, round_id: str, stage: int, scores: Sequence[Mapping[str, Any]], *, batch_size: int = SCORE_BATCH_SIZE) -> Dict[str, Any]:
        """Record per-run scores in bounded batches.

        The SQL function is idempotent per run (an equal score counts as
        existing, a different one is refused), so a stage of thousands of
        runs is written in batches that stay within request-size limits and a
        retry after a partial write completes the remainder.
        """

        items = [dict(item) for item in scores]
        if batch_size < 1:
            raise ArenaStoreError("score batch size must be positive")
        totals = {"status": "ok", "recorded": 0, "existing": 0, "batches": 0}
        for start in range(0, len(items), batch_size) or (0,):
            result = _require_mapping(
                self._transport.rpc(
                    "lab_arena_record_run_scores",
                    {"p_round_id": round_id, "p_stage": int(stage), "p_scores": items[start:start + batch_size]},
                ),
                "record_run_scores",
            )
            if result.get("status") != "ok":
                return dict(result)
            totals["recorded"] += int(result.get("recorded") or 0)
            totals["existing"] += int(result.get("existing") or 0)
            totals["batches"] += 1
        return totals

    # -- reads ------------------------------------------------------------

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        rows = self._transport.select("lab_arena_runs", filters={"run_id": run_id}, limit=1)
        return rows[0] if rows else None

    def list_runs(self, round_id: str, *, stage: Optional[int] = None, status: Optional[str] = None, submission_id: Optional[str] = None, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        filters: Dict[str, Any] = {"round_id": round_id}
        if stage is not None:
            filters["stage"] = int(stage)
        if status:
            filters["status"] = status
        if submission_id:
            filters["submission_id"] = submission_id
        if kind:
            filters["kind"] = kind
        return self._transport.select("lab_arena_runs", filters=filters, order="run_id")

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
