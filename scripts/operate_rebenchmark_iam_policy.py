#!/usr/bin/env python3
"""Operate bounded Leadpoet IAM policy-document changes through the gateway.

The local process never consults an AWS credential chain. It streams this
exact ``origin/main`` source and the exact gateway credential reader to the
gateway over SSH, and the remote process returns only a redacted receipt.
AWS IAM does not expose a conditional policy-document write, so writes use a
bounded optimistic protocol with exact pre/post hashes and a guarded rollback;
the receipt deliberately does not claim native compare-and-swap semantics.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
from fnmatch import fnmatchcase
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import select
import shlex
import socket
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
OPERATOR_PATH = "scripts/operate_rebenchmark_iam_policy.py"
SETUP_PATH = "scripts/setup_production_parity_staging.py"
GATEWAY_HOST = "ec2-user@52.91.135.79"
SSH_KEY = Path("/Users/pranav/Downloads/leadpoet-2026-07-28.pem")
SSH_KNOWN_HOSTS = Path("/Users/pranav/.ssh/known_hosts")
GIT_BIN = "/usr/bin/git"
SSH_BIN = "/usr/bin/ssh"
EXPECTED_ORIGIN_URL = "https://github.com/leadpoet/leadpoet.git"
EXPECTED_REGION = "us-east-1"
EXPECTED_ACCOUNT_ID = "493765492819"
EXPECTED_CALLER_ARN = "arn:aws:iam::493765492819:user/pranav-main"
AUTHORIZATION_ENV = "LEADPOET_OVERNIGHT_REBENCHMARK_AUTHORIZED"
DEFAULT_LEDGER = (
    Path.home()
    / ".codex"
    / "state"
    / "leadpoet-overnight-rebenchmark-validation.json"
)
REQUEST_SCHEMA = "leadpoet.rebenchmark_iam_policy_document_change.v1"
PLAN_REQUEST_SCHEMA = "leadpoet.rebenchmark_iam_policy_document_plan_request.v1"
PLAN_RECEIPT_SCHEMA = "leadpoet.rebenchmark_iam_policy_document_plan.v1"
TASK_SCOPE_SCHEMA = "leadpoet.rebenchmark_iam_task_scope.v1"
AUTHORITY_SCHEMA = "leadpoet.rebenchmark_iam_authority.v1"
RECEIPT_SCHEMA = "leadpoet.rebenchmark_iam_policy_document_receipt.v1"
LEDGER_SCHEMA = "leadpoet.overnight_rebenchmark_validation.v1"
NEVER_PAUSE_SCHEMA = "leadpoet.rebenchmark_iam_never_pause.v1"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
ROLE_NAME_RE = re.compile(r"^leadpoet-[A-Za-z0-9+=,.@_-]{1,56}$")
POLICY_NAME_RE = re.compile(
    r"^(?:Leadpoet[A-Za-z0-9+=,.@_-]{1,120}|leadpoet-[A-Za-z0-9+=,.@_-]{1,119})$"
)
CHANGE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,119}$")
MAX_REQUEST_BYTES = 256 * 1024
MAX_LEDGER_BYTES = 8 * 1024 * 1024
MAX_POLICY_BYTES = 6144
MAX_DELTA_ITEMS = 128
CONCURRENCY_MODEL = "bounded_optimistic_pre_post_hash_checks"
READBACK_ATTEMPTS = 6
READBACK_SLEEP_SECONDS = 0.2
FD_READ_TIMEOUT_SECONDS = 30.0

INLINE_TARGET_ALLOWLIST = frozenset(
    {
        (
            "leadpoet-gateway-s3-cloudwatch-role",
            "leadpoet-gateway-env-secretsmanager",
        ),
        (
            "leadpoet-production-parity-controller",
            "LeadpoetProductionParityRevokeOlderSessions",
        ),
        (
            "leadpoet-production-parity-runner",
            "LeadpoetProductionParityRunnerRevokeOlderSessions",
        ),
    }
)
MANAGED_POLICY_NAMES = frozenset(
    {
        "LeadpoetParityControllerEc2Launch",
        "LeadpoetParityControllerLifecycle",
        "LeadpoetParityControllerCloudFront",
        "LeadpoetParityControllerData",
    }
)
MANAGED_TARGET_ALLOWLIST = frozenset(
    f"arn:aws:iam::{EXPECTED_ACCOUNT_ID}:policy/leadpoet/production-parity/{name}"
    for name in MANAGED_POLICY_NAMES
)
PRINCIPAL_ROLE_ALLOWLIST = frozenset(
    {
        "leadpoet-gateway-s3-cloudwatch-role",
        "leadpoet-validator-s3-cloudwatch-role",
        "leadpoet-production-parity-controller",
        "leadpoet-production-parity-runner",
        "leadpoet-production-parity-static-bootstrap",
    }
)
FORBIDDEN_ACTION_SERVICES = frozenset(
    {"account", "iam", "organizations", "sts"}
)
GLOBAL_RESOURCE_ACTION_ALLOWLIST = frozenset(
    {
        "cloudfront:CreateDistribution",
        "cloudfront:ListCachePolicies",
        "cloudfront:ListDistributions",
        "cloudfront:ListOriginRequestPolicies",
        "ecr:GetAuthorizationToken",
        "ec2:DescribeInstances",
        "s3:ListAllMyBuckets",
        "secretsmanager:CreateSecret",
        "secretsmanager:ListSecrets",
        "ssm:DescribeInstanceInformation",
        "ssm:GetCommandInvocation",
        "sts:GetCallerIdentity",
    }
)
GLOBAL_ACTION_DECOYS = {
    "cloudfront": "cloudfront:DeleteDistribution",
    "ecr": "ecr:DeleteRepository",
    "ec2": "ec2:TerminateInstances",
    "s3": "s3:DeleteBucket",
    "secretsmanager": "secretsmanager:DeleteSecret",
    "ssm": "ssm:SendCommand",
    "sts": "sts:AssumeRole",
}

_AWS_SELECTORS = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_CONFIG_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ROLE_ARN",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_IAM",
        "AWS_ENDPOINT_URL_STS",
        "AWS_CA_BUNDLE",
        "AWS_DATA_PATH",
        "AWS_DEFAULT_REGION",
        "AWS_REGION",
        "AWS_SDK_LOAD_CONFIG",
        "AWS_STS_REGIONAL_ENDPOINTS",
        "AWS_USE_DUALSTACK_ENDPOINT",
        "AWS_USE_FIPS_ENDPOINT",
        "BOTO_CONFIG",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "https_proxy",
        "http_proxy",
        "all_proxy",
    }
)


class OperationError(RuntimeError):
    """A policy authority, identity, concurrency, or readback gate failed."""


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_json(value).encode("utf-8"))


def _policy_document(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(unquote(value))
        except (ValueError, TypeError) as exc:
            raise OperationError("IAM policy document is invalid") from exc
        if isinstance(decoded, Mapping):
            return decoded
    raise OperationError("IAM policy document is invalid")


def _canonical_condition(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_condition(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list):
        normalized = [_canonical_condition(item) for item in value]
        return sorted(normalized, key=_json)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise OperationError("IAM policy condition value is invalid")


def _canonical_string_set(value: Any, *, label: str) -> list[str]:
    values = [value] if isinstance(value, str) else value
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(item, str) or not item for item in values)
    ):
        raise OperationError(f"IAM policy {label} is invalid")
    if len(values) != len(set(values)):
        raise OperationError(f"IAM policy {label} contains duplicates")
    return sorted(values)


def _canonical_policy(value: Any) -> dict[str, Any]:
    document = dict(_policy_document(value))
    if set(document) != {"Version", "Statement"}:
        raise OperationError("IAM policy top-level fields are invalid")
    if document.get("Version") != "2012-10-17":
        raise OperationError("IAM policy version is invalid")
    statements = document.get("Statement")
    if isinstance(statements, Mapping):
        statements = [statements]
    if not isinstance(statements, list) or not 1 <= len(statements) <= 100:
        raise OperationError("IAM policy statements are invalid")
    normalized: list[dict[str, Any]] = []
    allowed_keys = {
        "Sid",
        "Effect",
        "Action",
        "Resource",
        "Condition",
    }
    for statement in statements:
        if not isinstance(statement, Mapping) or not set(statement).issubset(
            allowed_keys
        ):
            raise OperationError("IAM identity-policy statement is invalid")
        if statement.get("Effect") not in {"Allow", "Deny"}:
            raise OperationError("IAM policy effect is invalid")
        if "Action" not in statement:
            raise OperationError("IAM policy action selector is invalid")
        if "Resource" not in statement:
            raise OperationError("IAM policy resource selector is invalid")
        item: dict[str, Any] = {"Effect": str(statement["Effect"])}
        if "Sid" in statement:
            sid = statement["Sid"]
            if not isinstance(sid, str) or not re.fullmatch(
                r"[A-Za-z0-9]{1,128}", sid
            ):
                raise OperationError("IAM policy Sid is invalid")
            item["Sid"] = sid
        for key in ("Action", "Resource"):
            if key in statement:
                item[key] = _canonical_string_set(statement[key], label=key)
        if "Condition" in statement:
            if not isinstance(statement["Condition"], Mapping):
                raise OperationError("IAM policy condition is invalid")
            item["Condition"] = _canonical_condition(statement["Condition"])
        normalized.append(item)
    normalized.sort(key=_json)
    result = {"Version": "2012-10-17", "Statement": normalized}
    if len(_json(result).encode("utf-8")) > MAX_POLICY_BYTES:
        raise OperationError("IAM policy exceeds the managed-policy size bound")
    return result


def _policy_hash(value: Any) -> str:
    return _sha256_json(_canonical_policy(value))


def _structural_delta(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Return a statement-granularity, hash-only policy delta."""
    before_rows = before["Statement"]
    after_rows = after["Statement"]
    output: list[dict[str, Any]] = []
    common = min(len(before_rows), len(after_rows))
    for index in range(common):
        if before_rows[index] != after_rows[index]:
            output.append(
                {
                    "op": "replace",
                    "path": f"/Statement/{index}",
                    "before_hash": _sha256_json(before_rows[index]),
                    "after_hash": _sha256_json(after_rows[index]),
                }
            )
    for index in range(common, len(before_rows)):
        output.append(
            {
                "op": "remove",
                "path": f"/Statement/{index}",
                "before_hash": _sha256_json(before_rows[index]),
                "after_hash": None,
            }
        )
    for index in range(common, len(after_rows)):
        output.append(
            {
                "op": "add",
                "path": f"/Statement/{index}",
                "before_hash": None,
                "after_hash": _sha256_json(after_rows[index]),
            }
        )
    return output


def _policy_delta(
    before: Mapping[str, Any] | None, after: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if before is not None:
        return _structural_delta(before, after)
    return [
        {
            "op": "add",
            "path": f"/Statement/{index}",
            "before_hash": None,
            "after_hash": _sha256_json(statement),
        }
        for index, statement in enumerate(after["Statement"])
    ]


def _statement_map(document: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if document is None:
        return {}
    output: dict[str, dict[str, Any]] = {}
    seen_sids: set[str] = set()
    for index, statement in enumerate(document["Statement"]):
        sid = statement.get("Sid")
        if sid is None:
            key = f"index:{index}"
        elif isinstance(sid, str) and sid and sid not in seen_sids:
            seen_sids.add(sid)
            key = f"sid:{sid}"
        else:
            raise OperationError(
                "scoped IAM policy changes require unique statement Sids"
            )
        output[key] = dict(statement)
    return output


def _scoped_action(value: str) -> str:
    if value in GLOBAL_RESOURCE_ACTION_ALLOWLIST:
        return value
    if (
        re.fullmatch(r"[a-z0-9-]+:[A-Za-z0-9]+", value) is None
        or value.split(":", 1)[0] in FORBIDDEN_ACTION_SERVICES
    ):
        raise OperationError("IAM task scope contains a forbidden action")
    return value


def _scoped_deny_action(value: str) -> str:
    if value == "*" or re.fullmatch(r"[a-z0-9-]+:[A-Za-z0-9*]+", value):
        return value
    raise OperationError("IAM task scope contains an invalid deny action")


def _scoped_resource(value: str) -> str:
    if value == "*" or not value.startswith("arn:aws:"):
        raise OperationError("IAM task scope contains an unscoped resource")
    parts = value.split(":", 5)
    if len(parts) != 6:
        raise OperationError("IAM task scope contains an invalid resource")
    _arn, _partition, service, region, account, resource = parts
    if not service or not resource or region not in {"", EXPECTED_REGION}:
        raise OperationError("IAM task scope contains an invalid resource")
    if service == "s3":
        if account or region or not resource.lower().startswith("leadpoet-"):
            raise OperationError("IAM task scope contains an unrelated S3 resource")
    elif account != EXPECTED_ACCOUNT_ID:
        raise OperationError("IAM task scope contains an unrelated account resource")
    return value


def _scoped_added_resource(
    value: str, *, effect: str, actions: Sequence[str]
) -> str:
    if effect == "Deny" and value == "*":
        return value
    if value == "*":
        if actions and all(
            action in GLOBAL_RESOURCE_ACTION_ALLOWLIST for action in actions
        ):
            return value
        raise OperationError("IAM task scope contains an unpermitted global resource")
    return _scoped_resource(value)


def _computed_task_scope(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any],
    *,
    scope_id: str,
    target: Mapping[str, str],
) -> dict[str, Any]:
    before_rows = _statement_map(before)
    after_rows = _statement_map(after)
    changes: list[dict[str, Any]] = []
    for sid in sorted(set(before_rows) | set(after_rows)):
        prior = before_rows.get(sid)
        desired = after_rows.get(sid)
        if prior == desired:
            continue
        if prior is None:
            operation = "add"
        elif desired is None:
            operation = "remove"
        else:
            operation = "modify"
            if prior["Effect"] != desired["Effect"]:
                raise OperationError("IAM task scope cannot change a statement effect")
        prior_actions = set((prior or {}).get("Action", []))
        desired_actions = set((desired or {}).get("Action", []))
        prior_resources = set((prior or {}).get("Resource", []))
        desired_resources = set((desired or {}).get("Resource", []))
        added_actions = sorted(desired_actions - prior_actions)
        removed_actions = sorted(prior_actions - desired_actions)
        added_resources = sorted(desired_resources - prior_resources)
        removed_resources = sorted(prior_resources - desired_resources)
        desired_effect = str((desired or prior or {})["Effect"])
        normalized_added_actions = [
            (
                _scoped_deny_action(action)
                if desired_effect == "Deny"
                else _scoped_action(action)
            )
            for action in added_actions
        ]
        if (
            desired_effect == "Allow"
            and "*" in desired_resources
            and any(
                action not in GLOBAL_RESOURCE_ACTION_ALLOWLIST
                for action in normalized_added_actions
            )
        ):
            raise OperationError(
                "IAM task scope adds an action with an unpermitted global resource"
            )
        prior_condition = (prior or {}).get("Condition")
        desired_condition = (desired or {}).get("Condition")
        condition_hashes: list[str] = []
        if prior_condition != desired_condition:
            condition_hashes = sorted(
                {
                    _sha256_json(condition)
                    for condition in (prior_condition, desired_condition)
                    if condition is not None
                }
            )
        changes.append(
            {
                "sid": sid,
                "operation": operation,
                "added_actions": normalized_added_actions,
                "removed_actions": removed_actions,
                "added_resources": [
                    _scoped_added_resource(
                        resource,
                        effect=desired_effect,
                        actions=sorted(desired_actions),
                    )
                    for resource in added_resources
                ],
                "removed_resources": removed_resources,
                "condition_hashes": condition_hashes,
            }
        )
    return {
        "schema_version": TASK_SCOPE_SCHEMA,
        "scope_id": scope_id,
        "target": dict(target),
        "statement_changes": changes,
    }


def _validate_task_scope(
    value: Any,
    *,
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any],
    change_id: str,
    target: Mapping[str, str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "scope_id",
        "target",
        "statement_changes",
    }:
        raise OperationError("IAM task scope fields are invalid")
    if value.get("schema_version") != TASK_SCOPE_SCHEMA:
        raise OperationError("IAM task scope schema is invalid")
    if value.get("scope_id") != change_id or _target(value.get("target")) != target:
        raise OperationError("IAM task scope identity differs")
    expected = _computed_task_scope(
        before, after, scope_id=change_id, target=target
    )
    if value != expected:
        raise OperationError("IAM task scope differs from the exact statement changes")
    return expected


def _validate_expected_delta(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > MAX_DELTA_ITEMS:
        raise OperationError("expected IAM policy delta is invalid")
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {
            "op",
            "path",
            "before_hash",
            "after_hash",
        }:
            raise OperationError("expected IAM policy delta item is invalid")
        op = item.get("op")
        path = item.get("path")
        before_hash = item.get("before_hash")
        after_hash = item.get("after_hash")
        if (
            op not in {"add", "remove", "replace"}
            or not isinstance(path, str)
            or re.fullmatch(r"/Statement/(?:0|[1-9][0-9]*)", path) is None
            or (before_hash is not None and not HASH_RE.fullmatch(str(before_hash)))
            or (after_hash is not None and not HASH_RE.fullmatch(str(after_hash)))
            or (op == "add" and (before_hash is not None or after_hash is None))
            or (op == "remove" and (before_hash is None or after_hash is not None))
            or (op == "replace" and (before_hash is None or after_hash is None))
        ):
            raise OperationError("expected IAM policy delta item is invalid")
        normalized.append(
            {
                "op": str(op),
                "path": path,
                "before_hash": before_hash,
                "after_hash": after_hash,
            }
        )
    return normalized


def _target(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise OperationError("IAM policy target is invalid")
    kind = value.get("kind")
    if kind == "inline_role":
        if set(value) != {"kind", "role_name", "policy_name"}:
            raise OperationError("inline IAM policy target is invalid")
        role = str(value.get("role_name") or "")
        name = str(value.get("policy_name") or "")
        if (
            ROLE_NAME_RE.fullmatch(role) is None
            or POLICY_NAME_RE.fullmatch(name) is None
            or (role, name) not in INLINE_TARGET_ALLOWLIST
        ):
            raise OperationError("inline IAM policy target is outside Leadpoet scope")
        return {"kind": kind, "role_name": role, "policy_name": name}
    if kind == "managed":
        if set(value) != {"kind", "policy_arn"}:
            raise OperationError("managed IAM policy target is invalid")
        arn = str(value.get("policy_arn") or "")
        prefix = f"arn:aws:iam::{EXPECTED_ACCOUNT_ID}:policy/"
        suffix = arn.removeprefix(prefix)
        name = suffix.rsplit("/", 1)[-1]
        if (
            not arn.startswith(prefix)
            or not suffix
            or POLICY_NAME_RE.fullmatch(name) is None
            or arn not in MANAGED_TARGET_ALLOWLIST
        ):
            raise OperationError("managed IAM policy target is outside Leadpoet scope")
        return {"kind": kind, "policy_arn": arn}
    raise OperationError("IAM policy target kind is unsupported")


def _validate_simulations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not 1 <= len(value) <= 32:
        raise OperationError("IAM policy simulation cases are invalid")
    output: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {
            "name",
            "action",
            "resources",
            "context",
            "expected",
        }:
            raise OperationError("IAM policy simulation case is invalid")
        name = str(item.get("name") or "")
        action = str(item.get("action") or "")
        resources = item.get("resources")
        context = item.get("context")
        expected = item.get("expected")
        if (
            CHANGE_ID_RE.fullmatch(name) is None
            or not re.fullmatch(r"[a-z0-9-]+:[A-Za-z0-9]+", action)
            or not isinstance(resources, list)
            or not resources
            or len(resources) > 16
            or any(not isinstance(resource, str) or not resource for resource in resources)
            or not isinstance(context, Mapping)
            or expected not in {"allowed", "implicitDeny", "explicitDeny"}
        ):
            raise OperationError("IAM policy simulation case is invalid")
        normalized_context: dict[str, dict[str, Any]] = {}
        for key, raw_entry in context.items():
            if not isinstance(key, str) or not key:
                raise OperationError("IAM simulation context is invalid")
            if isinstance(raw_entry, Mapping):
                if set(raw_entry) != {"type", "values"}:
                    raise OperationError("IAM simulation context is invalid")
                value_type = str(raw_entry.get("type") or "")
                values = raw_entry.get("values")
            else:
                value_type = "string"
                values = raw_entry
            values = [values] if isinstance(values, str) else values
            if (
                value_type
                not in {
                    "string",
                    "stringList",
                    "numeric",
                    "numericList",
                    "boolean",
                    "booleanList",
                    "ip",
                    "ipList",
                    "binary",
                    "binaryList",
                    "date",
                    "dateList",
                }
                or not isinstance(values, list)
                or not values
                or any(not isinstance(entry, str) for entry in values)
            ):
                raise OperationError("IAM simulation context is invalid")
            normalized_context[key] = {
                "type": value_type,
                "values": list(values),
            }
        output.append(
            {
                "name": name,
                "action": action,
                "resources": list(resources),
                "context": normalized_context,
                "expected": expected,
            }
        )
    return output


def _validate_plan_request(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "change_id",
        "target",
        "desired_document",
        "task_scope",
        "simulations",
        "prune_managed_version",
    }:
        raise OperationError("IAM policy plan request fields are invalid")
    if value.get("schema_version") != PLAN_REQUEST_SCHEMA:
        raise OperationError("IAM policy plan request schema is invalid")
    change_id = str(value.get("change_id") or "")
    if CHANGE_ID_RE.fullmatch(change_id) is None:
        raise OperationError("IAM policy plan request identity is invalid")
    prune = value.get("prune_managed_version")
    if prune is not None:
        raise OperationError("pre-existing managed IAM policy versions cannot be pruned")
    if not isinstance(value.get("task_scope"), Mapping):
        raise OperationError("IAM task scope is invalid")
    return {
        "schema_version": PLAN_REQUEST_SCHEMA,
        "change_id": change_id,
        "target": _target(value["target"]),
        "desired_document": _canonical_policy(value["desired_document"]),
        "task_scope": dict(value["task_scope"]),
        "simulations": _validate_simulations(value["simulations"]),
        "prune_managed_version": prune,
    }


def _validate_plan_receipt(
    value: Any,
    *,
    commit: str | None = None,
    source_hash: str | None = None,
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "status",
        "change_id",
        "target",
        "origin_main_sha",
        "bridge_source_hash",
        "account_id",
        "caller_arn",
        "route",
        "local_chain",
        "prior_document_hash",
        "inventory_hash",
        "desired_document_hash",
        "expected_delta",
        "expected_delta_hash",
        "task_scope_hash",
        "simulation_case_count",
        "fixed_decoy_case_count",
        "principal_arns",
        "target_present",
        "managed_default_version_id",
        "managed_prior_versions",
        "managed_stable_hash",
        "concurrency_model",
        "aws_native_compare_and_swap",
        "secret_values_printed",
        "policy_material_printed",
        "plan_hash",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise OperationError("gateway IAM plan receipt fields differ")
    target = _target(value.get("target"))
    change_id = str(value.get("change_id") or "")
    delta = _validate_expected_delta(value.get("expected_delta"))
    principal_arns = value.get("principal_arns")
    versions = value.get("managed_prior_versions")
    if (
        value.get("schema_version") != PLAN_RECEIPT_SCHEMA
        or value.get("status") != "planned"
        or CHANGE_ID_RE.fullmatch(change_id) is None
        or value.get("account_id") != EXPECTED_ACCOUNT_ID
        or value.get("caller_arn") != EXPECTED_CALLER_ARN
        or value.get("route") != "gateway_bridge"
        or value.get("local_chain") != "ignored_non_authority"
        or value.get("concurrency_model") != CONCURRENCY_MODEL
        or value.get("aws_native_compare_and_swap") is not False
        or value.get("secret_values_printed") is not False
        or value.get("policy_material_printed") is not False
        or (commit is not None and value.get("origin_main_sha") != commit)
        or (source_hash is not None and value.get("bridge_source_hash") != source_hash)
        or SHA_RE.fullmatch(str(value.get("origin_main_sha") or "")) is None
        or HASH_RE.fullmatch(str(value.get("bridge_source_hash") or "")) is None
        or any(
            HASH_RE.fullmatch(str(value.get(name) or "")) is None
            for name in (
                "prior_document_hash",
                "inventory_hash",
                "desired_document_hash",
                "expected_delta_hash",
                "task_scope_hash",
            )
        )
        or value.get("expected_delta_hash") != _sha256_json(delta)
        or not isinstance(value.get("simulation_case_count"), int)
        or isinstance(value.get("simulation_case_count"), bool)
        or not 1 <= int(value["simulation_case_count"]) <= 32
        or not isinstance(value.get("fixed_decoy_case_count"), int)
        or isinstance(value.get("fixed_decoy_case_count"), bool)
        or not 1 <= int(value["fixed_decoy_case_count"]) <= 512
        or not isinstance(principal_arns, list)
        or not principal_arns
        or len(principal_arns) != len(set(principal_arns))
        or not isinstance(value.get("target_present"), bool)
        or not isinstance(versions, list)
    ):
        raise OperationError("gateway IAM plan receipt differs")
    for arn in principal_arns:
        prefix = f"arn:aws:iam::{EXPECTED_ACCOUNT_ID}:role/"
        role = str(arn).removeprefix(prefix)
        if not str(arn).startswith(prefix) or role not in PRINCIPAL_ROLE_ALLOWLIST:
            raise OperationError("gateway IAM plan principal differs")
    normalized_versions: list[dict[str, Any]] = []
    for row in versions:
        if not isinstance(row, Mapping) or set(row) != {
            "version_id",
            "is_default",
            "document_hash",
        }:
            raise OperationError("gateway IAM plan version inventory differs")
        if (
            re.fullmatch(r"v[1-9][0-9]*", str(row.get("version_id") or "")) is None
            or not isinstance(row.get("is_default"), bool)
            or HASH_RE.fullmatch(str(row.get("document_hash") or "")) is None
        ):
            raise OperationError("gateway IAM plan version inventory differs")
        normalized_versions.append(dict(row))
    if target["kind"] == "managed":
        if (
            re.fullmatch(
                r"v[1-9][0-9]*", str(value.get("managed_default_version_id") or "")
            )
            is None
            or not normalized_versions
            or len(normalized_versions) > 5
            or len(
                {row["version_id"] for row in normalized_versions}
            )
            != len(normalized_versions)
            or sum(bool(row["is_default"]) for row in normalized_versions) != 1
            or not any(
                row["is_default"]
                and row["version_id"] == value.get("managed_default_version_id")
                for row in normalized_versions
            )
            or value.get("target_present") is not True
            or HASH_RE.fullmatch(str(value.get("managed_stable_hash") or "")) is None
        ):
            raise OperationError("gateway managed IAM plan receipt differs")
    elif (
        value.get("managed_default_version_id") is not None
        or normalized_versions
        or value.get("managed_stable_hash") is not None
    ):
        raise OperationError("gateway inline IAM plan receipt differs")
    normalized = dict(value)
    normalized["target"] = target
    normalized["expected_delta"] = delta
    normalized["managed_prior_versions"] = normalized_versions
    plan_material = dict(normalized)
    plan_hash = str(plan_material.pop("plan_hash") or "")
    if HASH_RE.fullmatch(plan_hash) is None or plan_hash != _sha256_json(plan_material):
        raise OperationError("gateway IAM plan hash differs")
    normalized["plan_hash"] = plan_hash
    return normalized


def _validate_request(
    value: Any,
    *,
    commit: str | None = None,
    source_hash: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "change_id",
        "target",
        "desired_document",
        "task_scope",
        "simulations",
        "plan",
        "prune_managed_version",
    }:
        raise OperationError("IAM policy change request fields are invalid")
    if value.get("schema_version") != REQUEST_SCHEMA:
        raise OperationError("IAM policy change request schema is invalid")
    change_id = str(value.get("change_id") or "")
    target = _target(value.get("target"))
    desired = _canonical_policy(value.get("desired_document"))
    simulations = _validate_simulations(value.get("simulations"))
    plan = _validate_plan_receipt(
        value.get("plan"), commit=commit, source_hash=source_hash
    )
    if (
        CHANGE_ID_RE.fullmatch(change_id) is None
        or plan["change_id"] != change_id
        or plan["target"] != target
        or plan["desired_document_hash"] != _sha256_json(desired)
        or plan["task_scope_hash"] != _sha256_json(value.get("task_scope"))
        or plan["simulation_case_count"] != len(simulations)
    ):
        raise OperationError("IAM policy change request plan binding differs")
    if value.get("prune_managed_version") is not None:
        raise OperationError("pre-existing managed IAM policy versions cannot be pruned")
    return {
        "schema_version": REQUEST_SCHEMA,
        "change_id": change_id,
        "target": target,
        "expected_prior_document_hash": plan["prior_document_hash"],
        "expected_inventory_hash": plan["inventory_hash"],
        "desired_document": desired,
        "expected_delta": plan["expected_delta"],
        "task_scope": dict(value.get("task_scope") or {}),
        "simulations": simulations,
        "plan": plan,
        "prune_managed_version": None,
    }


def _page(iam: Any, method: str, key: str, **kwargs: Any) -> list[Any]:
    output: list[Any] = []
    marker: str | None = None
    while True:
        call = dict(kwargs)
        if marker:
            call["Marker"] = marker
        response = getattr(iam, method)(**call)
        values = response.get(key, [])
        if not isinstance(values, list):
            raise OperationError("IAM inventory response is invalid")
        output.extend(values)
        if not response.get("IsTruncated"):
            return output
        marker = str(response.get("Marker") or "")
        if not marker:
            raise OperationError("IAM inventory pagination is invalid")


def _role_inventory(iam: Any, role_name: str) -> tuple[dict[str, Any], Mapping[str, Any]]:
    role = iam.get_role(RoleName=role_name).get("Role", {})
    expected_arn = f"arn:aws:iam::{EXPECTED_ACCOUNT_ID}:role/{role_name}"
    if str(role.get("Arn") or "") != expected_arn:
        raise OperationError("IAM role identity differs")
    inline_names = sorted(
        str(name)
        for name in _page(iam, "list_role_policies", "PolicyNames", RoleName=role_name)
    )
    attached = sorted(
        str(item.get("PolicyArn") or "")
        for item in _page(
            iam, "list_attached_role_policies", "AttachedPolicies", RoleName=role_name
        )
    )
    profiles = sorted(
        str(item.get("Arn") or "")
        for item in _page(
            iam, "list_instance_profiles_for_role", "InstanceProfiles", RoleName=role_name
        )
    )
    tags = sorted(
        (str(item.get("Key") or ""), str(item.get("Value") or ""))
        for item in _page(iam, "list_role_tags", "Tags", RoleName=role_name)
    )
    inventory = {
        "kind": "inline_role",
        "role_arn": expected_arn,
        "path": str(role.get("Path") or ""),
        "permissions_boundary": role.get("PermissionsBoundary") or None,
        "max_session_duration": int(role.get("MaxSessionDuration") or 0),
        "trust_hash": _sha256_json(
            _policy_document(role.get("AssumeRolePolicyDocument"))
        ),
        "inline_policy_names": inline_names,
        "attached_policy_arns": attached,
        "instance_profile_arns": profiles,
        "tag_hash": _sha256_json(tags),
    }
    return inventory, role


def _inline_state(iam: Any, target: Mapping[str, str]) -> dict[str, Any]:
    role_name = target["role_name"]
    policy_name = target["policy_name"]
    inventory, _role = _role_inventory(iam, role_name)
    target_present = policy_name in inventory["inline_policy_names"]
    surrounding_inventory = dict(inventory)
    surrounding_inventory["inline_policy_names"] = [
        name for name in inventory["inline_policy_names"] if name != policy_name
    ]
    canonical = None
    if target_present:
        document = iam.get_role_policy(
            RoleName=role_name, PolicyName=policy_name
        ).get("PolicyDocument")
        canonical = _canonical_policy(document)
    return {
        "document": canonical,
        "document_hash": _sha256_json(canonical),
        "target_present": target_present,
        "inventory": surrounding_inventory,
        "inventory_hash": _sha256_json(surrounding_inventory),
    }


def _managed_entities(iam: Any, arn: str) -> dict[str, list[str]]:
    roles: list[str] = []
    users: list[str] = []
    groups: list[str] = []
    marker: str | None = None
    while True:
        call = {"PolicyArn": arn}
        if marker:
            call["Marker"] = marker
        response = iam.list_entities_for_policy(**call)
        roles.extend(str(item.get("RoleName") or "") for item in response.get("PolicyRoles", []))
        users.extend(str(item.get("UserName") or "") for item in response.get("PolicyUsers", []))
        groups.extend(str(item.get("GroupName") or "") for item in response.get("PolicyGroups", []))
        if not response.get("IsTruncated"):
            break
        marker = str(response.get("Marker") or "")
        if not marker:
            raise OperationError("managed IAM policy entity pagination is invalid")
    return {"roles": sorted(roles), "users": sorted(users), "groups": sorted(groups)}


def _managed_state(iam: Any, target: Mapping[str, str]) -> dict[str, Any]:
    arn = target["policy_arn"]
    policy = iam.get_policy(PolicyArn=arn).get("Policy", {})
    if str(policy.get("Arn") or "") != arn:
        raise OperationError("managed IAM policy identity differs")
    versions = iam.list_policy_versions(PolicyArn=arn).get("Versions", [])
    if (
        not isinstance(versions, list)
        or not 1 <= len(versions) <= 5
        or len({str(item.get("VersionId") or "") for item in versions}) != len(versions)
    ):
        raise OperationError("managed IAM policy version inventory is invalid")
    version_rows: list[dict[str, Any]] = []
    for item in versions:
        version_id = str(item.get("VersionId") or "")
        if not version_id:
            raise OperationError("managed IAM policy version identity is invalid")
        document = iam.get_policy_version(
            PolicyArn=arn, VersionId=version_id
        ).get("PolicyVersion", {}).get("Document")
        version_rows.append(
            {
                "version_id": version_id,
                "is_default": item.get("IsDefaultVersion") is True,
                "document_hash": _policy_hash(document),
            }
        )
    version_rows.sort(key=lambda row: row["version_id"])
    defaults = [row for row in version_rows if row["is_default"]]
    default_id = str(policy.get("DefaultVersionId") or "")
    if len(defaults) != 1 or defaults[0]["version_id"] != default_id:
        raise OperationError("managed IAM policy default version differs")
    tags = sorted(
        (str(item.get("Key") or ""), str(item.get("Value") or ""))
        for item in _page(iam, "list_policy_tags", "Tags", PolicyArn=arn)
    )
    stable = {
        "kind": "managed",
        "arn": arn,
        "name": str(policy.get("PolicyName") or ""),
        "path": str(policy.get("Path") or ""),
        "description": str(policy.get("Description") or ""),
        "entities": _managed_entities(iam, arn),
        "tag_hash": _sha256_json(tags),
    }
    inventory = {"stable": stable, "versions": version_rows}
    default_document = iam.get_policy_version(
        PolicyArn=arn, VersionId=default_id
    ).get("PolicyVersion", {}).get("Document")
    canonical = _canonical_policy(default_document)
    return {
        "document": canonical,
        "document_hash": _sha256_json(canonical),
        "default_version_id": default_id,
        "inventory": inventory,
        "inventory_hash": _sha256_json(inventory),
    }


def _context_entries(context: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "ContextKeyName": key,
            "ContextKeyValues": list(entry["values"]),
            "ContextKeyType": entry["type"],
        }
        for key, entry in sorted(context.items())
    ]


def _check_simulation_response(
    response: Mapping[str, Any], case: Mapping[str, Any], *, label: str
) -> None:
    if response.get("IsTruncated"):
        raise OperationError(f"IAM {label} simulation was truncated")
    results = response.get("EvaluationResults")
    if not isinstance(results, list) or not results:
        raise OperationError(f"IAM {label} simulation response is invalid")
    decisions: dict[str, str] = {}
    missing: set[str] = set()
    for result in results:
        if not isinstance(result, Mapping) or result.get("EvalActionName") != case["action"]:
            raise OperationError(f"IAM {label} simulation response is invalid")
        missing.update(str(value) for value in result.get("MissingContextValues", []))
        specific = result.get("ResourceSpecificResults") or []
        if specific:
            for item in specific:
                resource = str(item.get("EvalResourceName") or "")
                decision = str(item.get("EvalResourceDecision") or "")
                if resource in decisions:
                    raise OperationError(f"IAM {label} simulation response is ambiguous")
                decisions[resource] = decision
        else:
            resource = str(result.get("EvalResourceName") or "")
            decision = str(result.get("EvalDecision") or "")
            if resource in decisions:
                raise OperationError(f"IAM {label} simulation response is ambiguous")
            decisions[resource] = decision
    if (
        missing
        or set(decisions) != set(case["resources"])
        or len(case["resources"]) != len(set(case["resources"]))
        or set(decisions.values()) != {case["expected"]}
    ):
        raise OperationError(f"IAM {label} simulation {case['name']} differs")


def _simulate_custom(
    iam: Any, document: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]
) -> None:
    for case in cases:
        response = iam.simulate_custom_policy(
            PolicyInputList=[_json(document)],
            ActionNames=[case["action"]],
            ResourceArns=list(case["resources"]),
            ContextEntries=_context_entries(case["context"]),
        )
        _check_simulation_response(response, case, label="custom-policy")


def _simulate_principals(
    iam: Any,
    principal_arns: Sequence[str],
    cases: Sequence[Mapping[str, Any]],
) -> None:
    for principal_arn in principal_arns:
        for case in cases:
            response = iam.simulate_principal_policy(
                PolicySourceArn=principal_arn,
                ActionNames=[case["action"]],
                ResourceArns=list(case["resources"]),
                ContextEntries=_context_entries(case["context"]),
            )
            _check_simulation_response(response, case, label="principal-policy")


def _decoy_resource(resource: str, *, action: str) -> str:
    if resource == "*":
        service = action.split(":", 1)[0]
        digest = hashlib.sha256(action.encode("utf-8")).hexdigest()[:16]
        return (
            f"arn:aws:{service}:{EXPECTED_REGION}:{EXPECTED_ACCOUNT_ID}:"
            f"leadpoet-iam-decoy/{digest}"
        )
    parts = resource.split(":", 5)
    if len(parts) != 6:
        raise OperationError("IAM simulation resource cannot produce a fixed decoy")
    service = parts[2]
    digest = hashlib.sha256(resource.encode("utf-8")).hexdigest()[:16]
    if service == "s3":
        return (
            f"arn:aws:s3:::leadpoet-iam-decoy-{EXPECTED_ACCOUNT_ID}/"
            f"{digest}"
        )
    region = parts[3] or EXPECTED_REGION
    return (
        f"arn:aws:{service}:{region}:{EXPECTED_ACCOUNT_ID}:"
        f"leadpoet-iam-decoy/{digest}"
    )


def _action_matches(pattern: str, action: str) -> bool:
    return fnmatchcase(action.lower(), pattern.lower())


def _document_matches_resource(
    document: Mapping[str, Any], *, action: str, resource: str
) -> bool:
    for statement in document["Statement"]:
        if statement["Effect"] != "Allow":
            continue
        if not any(_action_matches(pattern, action) for pattern in statement["Action"]):
            continue
        if any(fnmatchcase(resource, pattern) for pattern in statement["Resource"]):
            return True
    return False


def _fixed_decoy_cases(
    document: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: dict[tuple[str, str, str], str] = {}
    for case in cases:
        for resource in case["resources"]:
            decoy = _decoy_resource(resource, action=case["action"])
            decoy_action = case["action"]
            expected = (
                "implicitDeny" if case["expected"] == "allowed" else case["expected"]
            )
            if case["expected"] == "allowed" and _document_matches_resource(
                document, action=case["action"], resource=decoy
            ):
                if case["action"] not in GLOBAL_RESOURCE_ACTION_ALLOWLIST:
                    raise OperationError(
                        "IAM desired policy is too broad for the fixed decoy check"
                    )
                service = case["action"].split(":", 1)[0]
                decoy_action = GLOBAL_ACTION_DECOYS[service]
                if _document_matches_resource(
                    document, action=decoy_action, resource=decoy
                ):
                    raise OperationError(
                        "IAM desired policy grants the fixed global-action decoy"
                    )
            key = (decoy_action, decoy, _sha256_json(case["context"]))
            if key in seen and seen[key] != expected:
                raise OperationError("IAM fixed decoy expectations conflict")
            if key in seen:
                continue
            seen[key] = expected
            output.append(
                {
                    "name": f"fixed-decoy-{len(output) + 1}",
                    "action": decoy_action,
                    "resources": [decoy],
                    "context": dict(case["context"]),
                    "expected": expected,
                }
            )
    if not output:
        raise OperationError("IAM policy plan requires a fixed same-service decoy")
    return output


def _principal_arns(
    state: Mapping[str, Any], target: Mapping[str, str]
) -> list[str]:
    if target["kind"] == "inline_role":
        roles = [target["role_name"]]
    else:
        entities = state["inventory"]["stable"]["entities"]
        roles = list(entities["roles"])
        if entities["users"] or entities["groups"]:
            raise OperationError(
                "managed IAM policy is attached to a user or group outside scope"
            )
    if (
        not roles
        or len(roles) != len(set(roles))
        or any(role not in PRINCIPAL_ROLE_ALLOWLIST for role in roles)
    ):
        raise OperationError("IAM effective-policy principal is outside scope")
    return [f"arn:aws:iam::{EXPECTED_ACCOUNT_ID}:role/{role}" for role in sorted(roles)]


def _simulate_before_write(
    iam: Any,
    document: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    decoys = _fixed_decoy_cases(document, cases)
    _simulate_custom(iam, document, list(cases) + decoys)
    return decoys


def _simulate_after_write(
    iam: Any,
    document: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    decoys: Sequence[Mapping[str, Any]],
    principal_arns: Sequence[str],
) -> None:
    combined = list(cases) + list(decoys)
    _simulate_custom(iam, document, combined)
    _simulate_principals(iam, principal_arns, combined)


def _wait_for_state(
    loader: Any,
    accepted: Any,
    *,
    label: str,
) -> Mapping[str, Any]:
    last_state: Mapping[str, Any] | None = None
    last_error: Exception | None = None
    for attempt in range(READBACK_ATTEMPTS):
        try:
            last_state = loader()
            if accepted(last_state):
                return last_state
        except Exception as exc:  # AWS may acknowledge a write before response loss.
            last_error = exc
        if attempt + 1 < READBACK_ATTEMPTS:
            time.sleep(READBACK_SLEEP_SECONDS * (2**attempt))
    if last_state is not None:
        return last_state
    raise OperationError(f"IAM {label} readback was unavailable") from last_error


def _simulate(iam: Any, document: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> None:
    """Backward-compatible custom-policy helper for focused callers/tests."""
    _simulate_custom(iam, document, cases)


def _plan_receipt(
    iam: Any,
    request: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
    source_hash: str,
    commit: str,
    account_id: str,
    caller_arn: str,
) -> dict[str, Any]:
    scope = _validate_task_scope(
        request["task_scope"],
        before=state["document"],
        after=request["desired_document"],
        change_id=request["change_id"],
        target=request["target"],
    )
    delta = _policy_delta(state["document"], request["desired_document"])
    if len(delta) > MAX_DELTA_ITEMS:
        raise OperationError("IAM policy structural delta is too large")
    decoys = _simulate_before_write(
        iam, request["desired_document"], request["simulations"]
    )
    principals = _principal_arns(state, request["target"])
    managed = request["target"]["kind"] == "managed"
    if (
        managed
        and state["document_hash"] != _sha256_json(request["desired_document"])
        and len(state["inventory"]["versions"]) >= 5
    ):
        raise OperationError(
            "managed IAM policy capacity cannot preserve an exact rollback version"
        )
    receipt: dict[str, Any] = {
        "schema_version": PLAN_RECEIPT_SCHEMA,
        "status": "planned",
        "change_id": request["change_id"],
        "target": dict(request["target"]),
        "origin_main_sha": commit,
        "bridge_source_hash": source_hash,
        "account_id": account_id,
        "caller_arn": caller_arn,
        "route": "gateway_bridge",
        "local_chain": "ignored_non_authority",
        "prior_document_hash": state["document_hash"],
        "inventory_hash": state["inventory_hash"],
        "desired_document_hash": _sha256_json(request["desired_document"]),
        "expected_delta": delta,
        "expected_delta_hash": _sha256_json(delta),
        "task_scope_hash": _sha256_json(scope),
        "simulation_case_count": len(request["simulations"]),
        "fixed_decoy_case_count": len(decoys),
        "principal_arns": principals,
        "target_present": bool(state.get("target_present", True)),
        "managed_default_version_id": (
            state["default_version_id"] if managed else None
        ),
        "managed_prior_versions": (
            list(state["inventory"]["versions"]) if managed else []
        ),
        "managed_stable_hash": (
            _sha256_json(state["inventory"]["stable"]) if managed else None
        ),
        "concurrency_model": CONCURRENCY_MODEL,
        "aws_native_compare_and_swap": False,
        "secret_values_printed": False,
        "policy_material_printed": False,
    }
    receipt["plan_hash"] = _sha256_json(receipt)
    return receipt


def _check_preconditions(state: Mapping[str, Any], request: Mapping[str, Any]) -> None:
    if state["document_hash"] != request["expected_prior_document_hash"]:
        raise OperationError("IAM policy prior document hash differs")
    if state["inventory_hash"] != request["expected_inventory_hash"]:
        raise OperationError("IAM policy surrounding inventory hash differs")
    actual_delta = _policy_delta(state["document"], request["desired_document"])
    if len(actual_delta) > MAX_DELTA_ITEMS or actual_delta != request["expected_delta"]:
        raise OperationError("IAM policy structural delta differs")
    scope = _validate_task_scope(
        request["task_scope"],
        before=state["document"],
        after=request["desired_document"],
        change_id=request["change_id"],
        target=request["target"],
    )
    plan = request["plan"]
    if (
        plan["task_scope_hash"] != _sha256_json(scope)
        or plan["expected_delta_hash"] != _sha256_json(actual_delta)
        or plan["target_present"] != bool(state.get("target_present", True))
        or plan["principal_arns"] != _principal_arns(state, request["target"])
    ):
        raise OperationError("IAM policy plan no longer matches live state")
    if request["target"]["kind"] == "managed" and (
        plan["managed_default_version_id"] != state["default_version_id"]
        or plan["managed_prior_versions"] != state["inventory"]["versions"]
        or plan["managed_stable_hash"]
        != _sha256_json(state["inventory"]["stable"])
    ):
        raise OperationError("managed IAM policy plan inventory differs")


def _receipt_base(
    *,
    request: Mapping[str, Any],
    source_hash: str,
    commit: str,
    account_id: str,
    caller_arn: str,
) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA,
        "change_id": request["change_id"],
        "target": dict(request["target"]),
        "origin_main_sha": commit,
        "bridge_source_hash": source_hash,
        "account_id": account_id,
        "caller_arn": caller_arn,
        "route": "gateway_bridge",
        "local_chain": "ignored_non_authority",
        "prior_document_hash": request["expected_prior_document_hash"],
        "desired_document_hash": _sha256_json(request["desired_document"]),
        "expected_delta_hash": _sha256_json(request["expected_delta"]),
        "task_scope_hash": _sha256_json(request["task_scope"]),
        "plan_hash": request["plan"]["plan_hash"],
        "simulation_case_count": len(request["simulations"]),
        "fixed_decoy_case_count": request["plan"]["fixed_decoy_case_count"],
        "principal_simulation_count": (
            len(request["plan"]["principal_arns"])
            * (
                len(request["simulations"])
                + request["plan"]["fixed_decoy_case_count"]
            )
        ),
        "concurrency_model": CONCURRENCY_MODEL,
        "aws_native_compare_and_swap": False,
        "secret_values_printed": False,
        "policy_material_printed": False,
    }


def _apply_inline(
    iam: Any,
    request: Mapping[str, Any],
    *,
    source_hash: str,
    commit: str,
    account_id: str,
    caller_arn: str,
) -> dict[str, Any]:
    target = request["target"]
    before = _inline_state(iam, target)
    desired_hash = _sha256_json(request["desired_document"])

    def is_planned_before(state: Mapping[str, Any]) -> bool:
        return (
            state["document_hash"] == request["expected_prior_document_hash"]
            and state["inventory_hash"] == request["expected_inventory_hash"]
            and state["target_present"] == request["plan"]["target_present"]
        )

    def is_desired(state: Mapping[str, Any]) -> bool:
        return (
            state["document_hash"] == desired_hash
            and state["inventory_hash"] == request["expected_inventory_hash"]
            and state["target_present"] is True
        )

    reconciled = False
    if not is_planned_before(before):
        if is_desired(before):
            reconciled = True
        else:
            raise OperationError("IAM inline policy plan no longer matches live state")
    if not reconciled:
        _check_preconditions(before, request)
    decoys = _simulate_before_write(
        iam, request["desired_document"], request["simulations"]
    )
    if len(decoys) != request["plan"]["fixed_decoy_case_count"]:
        raise OperationError("IAM fixed decoy plan differs")
    principal_arns = request["plan"]["principal_arns"]
    if before["document_hash"] == desired_hash:
        _simulate_after_write(
            iam,
            request["desired_document"],
            request["simulations"],
            decoys,
            principal_arns,
        )
        final = _wait_for_state(
            lambda: _inline_state(iam, target),
            is_desired,
            label="inline final",
        )
        if not is_desired(final):
            raise OperationError("IAM inline policy changed during final verification")
        receipt = _receipt_base(
            request=request,
            source_hash=source_hash,
            commit=commit,
            account_id=account_id,
            caller_arn=caller_arn,
        )
        receipt.update(
            status="reconciled" if reconciled else "unchanged",
            readback_document_hash=final["document_hash"],
        )
        return receipt
    immediate = _inline_state(iam, target)
    if not is_planned_before(immediate):
        raise OperationError("IAM inline policy changed before write")
    write_error: Exception | None = None
    try:
        iam.put_role_policy(
            RoleName=target["role_name"],
            PolicyName=target["policy_name"],
            PolicyDocument=_json(request["desired_document"]),
        )
    except Exception as exc:
        write_error = exc
    after = _wait_for_state(
        lambda: _inline_state(iam, target),
        is_desired,
        label="inline post-write",
    )
    if not is_desired(after):
        if is_planned_before(after):
            raise OperationError("IAM inline policy write did not take effect") from write_error
        raise OperationError("IAM inline policy entered an unexpected concurrent state")
    try:
        _simulate_after_write(
            iam,
            request["desired_document"],
            request["simulations"],
            decoys,
            principal_arns,
        )
    except Exception as exc:
        live = _wait_for_state(
            lambda: _inline_state(iam, target),
            lambda state: is_desired(state) or is_planned_before(state),
            label="inline rollback guard",
        )
        if not is_desired(live):
            raise OperationError(
                "IAM inline policy post-write simulation found a third state"
            ) from exc
        rollback_error: Exception | None = None
        for _attempt in range(2):
            try:
                if request["plan"]["target_present"]:
                    iam.put_role_policy(
                        RoleName=target["role_name"],
                        PolicyName=target["policy_name"],
                        PolicyDocument=_json(before["document"]),
                    )
                else:
                    iam.delete_role_policy(
                        RoleName=target["role_name"], PolicyName=target["policy_name"]
                    )
            except Exception as rollback_exc:
                rollback_error = rollback_exc
            rolled_back = _wait_for_state(
                lambda: _inline_state(iam, target),
                is_planned_before,
                label="inline rollback",
            )
            if is_planned_before(rolled_back):
                break
            if not is_desired(rolled_back):
                raise OperationError("IAM inline policy rollback found a third state") from exc
        else:
            raise OperationError("IAM inline policy rollback readback differs") from exc
        raise OperationError("IAM inline policy post-write simulation rolled back") from exc
    final = _wait_for_state(
        lambda: _inline_state(iam, target),
        is_desired,
        label="inline final",
    )
    if not is_desired(final):
        raise OperationError("IAM inline policy changed during final verification")
    receipt = _receipt_base(
        request=request,
        source_hash=source_hash,
        commit=commit,
        account_id=account_id,
        caller_arn=caller_arn,
    )
    receipt.update(
        status="reconciled" if write_error is not None else "updated",
        readback_document_hash=final["document_hash"],
    )
    return receipt


def _version_map(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["version_id"]): dict(item)
        for item in state["inventory"]["versions"]
    }


def _managed_inventory_matches_addition(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    new_version: str,
    new_hash: str,
    default_version: str,
) -> bool:
    if before["inventory"]["stable"] != after["inventory"]["stable"]:
        return False
    expected = _version_map(before)
    for row in expected.values():
        row["is_default"] = row["version_id"] == default_version
    expected[new_version] = {
        "version_id": new_version,
        "is_default": new_version == default_version,
        "document_hash": new_hash,
    }
    return expected == _version_map(after)


def _managed_matches_plan_base(
    state: Mapping[str, Any], plan: Mapping[str, Any]
) -> bool:
    return (
        state["default_version_id"] == plan["managed_default_version_id"]
        and _sha256_json(state["inventory"]["stable"])
        == plan["managed_stable_hash"]
        and list(state["inventory"]["versions"]) == plan["managed_prior_versions"]
        and state["inventory_hash"] == plan["inventory_hash"]
        and state["document_hash"] == plan["prior_document_hash"]
    )


def _managed_plan_addition(
    state: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    desired_hash: str,
    default_new: bool,
) -> str | None:
    if _sha256_json(state["inventory"]["stable"]) != plan["managed_stable_hash"]:
        return None
    expected = {
        str(item["version_id"]): dict(item)
        for item in plan["managed_prior_versions"]
    }
    current = _version_map(state)
    additions = sorted(set(current) - set(expected))
    if len(additions) != 1 or set(expected) - set(current):
        return None
    new_version = additions[0]
    expected_default = new_version if default_new else plan["managed_default_version_id"]
    for row in expected.values():
        row["is_default"] = row["version_id"] == expected_default
    expected[new_version] = {
        "version_id": new_version,
        "is_default": default_new,
        "document_hash": desired_hash,
    }
    if current != expected or state["default_version_id"] != expected_default:
        return None
    return new_version


def _apply_managed(
    iam: Any,
    request: Mapping[str, Any],
    *,
    source_hash: str,
    commit: str,
    account_id: str,
    caller_arn: str,
) -> dict[str, Any]:
    target = request["target"]
    arn = target["policy_arn"]
    plan = request["plan"]
    desired_hash = _sha256_json(request["desired_document"])
    before = _managed_state(iam, target)
    base_matches = _managed_matches_plan_base(before, plan)
    staged_version = _managed_plan_addition(
        before, plan, desired_hash=desired_hash, default_new=False
    )
    resumed_staged = staged_version is not None
    active_version = _managed_plan_addition(
        before, plan, desired_hash=desired_hash, default_new=True
    )
    if not base_matches and staged_version is None and active_version is None:
        raise OperationError("managed IAM policy plan no longer matches live state")
    if base_matches:
        _check_preconditions(before, request)
    decoys = _simulate_before_write(
        iam, request["desired_document"], request["simulations"]
    )
    if len(decoys) != plan["fixed_decoy_case_count"]:
        raise OperationError("IAM fixed decoy plan differs")
    principal_arns = plan["principal_arns"]

    def is_base(state: Mapping[str, Any]) -> bool:
        return _managed_matches_plan_base(state, plan)

    def active_addition(state: Mapping[str, Any]) -> str | None:
        return _managed_plan_addition(
            state, plan, desired_hash=desired_hash, default_new=True
        )

    def staged_addition(state: Mapping[str, Any]) -> str | None:
        return _managed_plan_addition(
            state, plan, desired_hash=desired_hash, default_new=False
        )

    def clean_staged_addition(version_id: str, *, label: str) -> None:
        last_error: Exception | None = None
        for _attempt in range(2):
            live = _wait_for_state(
                lambda: _managed_state(iam, target),
                lambda state: is_base(state)
                or staged_addition(state) == version_id,
                label=f"{label} guard",
            )
            if is_base(live):
                return
            if staged_addition(live) != version_id:
                raise OperationError(
                    "managed IAM policy staged cleanup found a third state"
                )
            try:
                iam.delete_policy_version(PolicyArn=arn, VersionId=version_id)
            except Exception as exc:
                last_error = exc
            checked = _wait_for_state(
                lambda: _managed_state(iam, target),
                lambda state: is_base(state)
                or staged_addition(state) == version_id,
                label=label,
            )
            if is_base(checked):
                return
            if staged_addition(checked) != version_id:
                raise OperationError(
                    "managed IAM policy staged cleanup found a third state"
                )
        raise OperationError("managed IAM policy staged cleanup differs") from last_error

    if active_version is not None or (base_matches and before["document_hash"] == desired_hash):
        _simulate_after_write(
            iam,
            request["desired_document"],
            request["simulations"],
            decoys,
            principal_arns,
        )
        final = _wait_for_state(
            lambda: _managed_state(iam, target),
            lambda state: (
                active_addition(state) is not None
                if active_version is not None
                else is_base(state)
            ),
            label="managed final",
        )
        final_version = active_addition(final)
        if active_version is not None and final_version is None:
            raise OperationError("managed IAM policy changed during final verification")
        if active_version is None and not is_base(final):
            raise OperationError("managed IAM policy changed during final verification")
        receipt = _receipt_base(
            request=request,
            source_hash=source_hash,
            commit=commit,
            account_id=account_id,
            caller_arn=caller_arn,
        )
        receipt.update(
            status="reconciled" if active_version is not None else "unchanged",
            readback_document_hash=final["document_hash"],
            **(
                {
                    "managed_version_id": final_version,
                    "pruned_managed_version_id": None,
                }
                if active_version is not None
                else {}
            ),
        )
        return receipt
    if staged_version is None:
        immediate = _managed_state(iam, target)
        if not is_base(immediate):
            raise OperationError("managed IAM policy changed before write")
        if len(_version_map(immediate)) >= 5:
            raise OperationError(
                "managed IAM policy capacity cannot preserve an exact rollback version"
            )
        create_error: Exception | None = None
        returned_version = ""
        try:
            created = iam.create_policy_version(
                PolicyArn=arn,
                PolicyDocument=_json(request["desired_document"]),
                SetAsDefault=False,
            ).get("PolicyVersion", {})
            returned_version = str(created.get("VersionId") or "")
        except Exception as exc:
            create_error = exc
        staged = _wait_for_state(
            lambda: _managed_state(iam, target),
            lambda state: staged_addition(state) is not None,
            label="managed staged",
        )
        staged_version = staged_addition(staged)
        if staged_version is None:
            if is_base(staged):
                raise OperationError(
                    "managed IAM policy version write did not take effect"
                ) from create_error
            raise OperationError("managed IAM policy staged version entered a third state")
        if returned_version and returned_version != staged_version:
            raise OperationError("managed IAM policy created version identity differs")
    new_version = staged_version
    set_default_error: Exception | None = None
    after: Mapping[str, Any] | None = None
    for _attempt in range(2):
        guarded = _wait_for_state(
            lambda: _managed_state(iam, target),
            lambda state: active_addition(state) == new_version
            or staged_addition(state) == new_version,
            label="managed default guard",
        )
        if active_addition(guarded) == new_version:
            after = guarded
            break
        if staged_addition(guarded) != new_version:
            raise OperationError(
                "managed IAM policy default guard found a third state"
            )
        try:
            iam.set_default_policy_version(PolicyArn=arn, VersionId=new_version)
        except Exception as exc:
            set_default_error = exc
        after = _wait_for_state(
            lambda: _managed_state(iam, target),
            lambda state: active_addition(state) == new_version,
            label="managed post-write",
        )
        if active_addition(after) == new_version:
            break
        if staged_addition(after) != new_version:
            raise OperationError(
                "managed IAM policy entered an unexpected concurrent state"
            )
    else:
        clean_staged_addition(new_version, label="managed staged cleanup")
        raise OperationError(
            "managed IAM policy default write did not take effect"
        ) from set_default_error
    assert after is not None and active_addition(after) == new_version
    try:
        _simulate_after_write(
            iam,
            request["desired_document"],
            request["simulations"],
            decoys,
            principal_arns,
        )
    except Exception as exc:
        live = _wait_for_state(
            lambda: _managed_state(iam, target),
            lambda state: active_addition(state) == new_version,
            label="managed rollback guard",
        )
        if active_addition(live) != new_version:
            raise OperationError(
                "managed IAM policy post-write simulation found a third state"
            ) from exc
        rollback_error: Exception | None = None
        reverted: Mapping[str, Any] | None = None
        for _attempt in range(2):
            guarded = _wait_for_state(
                lambda: _managed_state(iam, target),
                lambda state: active_addition(state) == new_version
                or staged_addition(state) == new_version,
                label="managed rollback default guard",
            )
            if staged_addition(guarded) == new_version:
                reverted = guarded
                break
            if active_addition(guarded) != new_version:
                raise OperationError(
                    "managed IAM policy rollback default found a third state"
                ) from exc
            try:
                iam.set_default_policy_version(
                    PolicyArn=arn, VersionId=plan["managed_default_version_id"]
                )
            except Exception as rollback_exc:
                rollback_error = rollback_exc
            reverted = _wait_for_state(
                lambda: _managed_state(iam, target),
                lambda state: staged_addition(state) == new_version,
                label="managed rollback default",
            )
            if staged_addition(reverted) == new_version:
                break
            if active_addition(reverted) != new_version:
                raise OperationError(
                    "managed IAM policy rollback default found a third state"
                ) from exc
        else:
            raise OperationError(
                "managed IAM policy rollback default differs"
            ) from rollback_error
        assert reverted is not None and staged_addition(reverted) == new_version
        try:
            clean_staged_addition(new_version, label="managed rollback cleanup")
        except OperationError as cleanup_exc:
            raise OperationError(
                "managed IAM policy rollback readback differs"
            ) from cleanup_exc
        raise OperationError("managed IAM policy post-write simulation rolled back") from exc
    final = _wait_for_state(
        lambda: _managed_state(iam, target),
        lambda state: active_addition(state) == new_version,
        label="managed final",
    )
    if active_addition(final) != new_version:
        raise OperationError("managed IAM policy changed during final verification")
    receipt = _receipt_base(
        request=request,
        source_hash=source_hash,
        commit=commit,
        account_id=account_id,
        caller_arn=caller_arn,
    )
    receipt.update(
        status=(
            "reconciled"
            if resumed_staged or create_error is not None or set_default_error is not None
            else "updated"
        ),
        readback_document_hash=final["document_hash"],
        managed_version_id=new_version,
        pruned_managed_version_id=None,
    )
    return receipt


def _gateway_clients(setup: Mapping[str, Any]) -> tuple[Any, Any, str, str]:
    sts, iam, account_id = setup["_iam_clients"](EXPECTED_REGION)
    identity = sts.get_caller_identity()
    caller_account = str(identity.get("Account") or "")
    caller_arn = str(identity.get("Arn") or "")
    if account_id != EXPECTED_ACCOUNT_ID or caller_account != EXPECTED_ACCOUNT_ID:
        raise OperationError("gateway IAM caller account differs")
    if caller_arn != EXPECTED_CALLER_ARN:
        raise OperationError("gateway IAM caller principal differs")
    return sts, iam, account_id, caller_arn


def _remote_entry(
    operation: str,
    request: Any,
    setup: Mapping[str, Any],
    *,
    source_hash: str,
    commit: str,
) -> dict[str, Any]:
    sts, iam, account_id, caller_arn = _gateway_clients(setup)
    try:
        if operation == "probe":
            return {
                "schema_version": AUTHORITY_SCHEMA,
                "status": "authority_ready",
                "origin_main_sha": commit,
                "bridge_source_hash": source_hash,
                "account_id": account_id,
                "caller_arn": caller_arn,
                "route": "gateway_bridge",
                "local_chain": "ignored_non_authority",
                "secret_values_printed": False,
                "policy_material_printed": False,
            }
        if operation == "plan":
            validated_plan = _validate_plan_request(request)
            if validated_plan["target"]["kind"] == "inline_role":
                state = _inline_state(iam, validated_plan["target"])
            else:
                state = _managed_state(iam, validated_plan["target"])
            return _plan_receipt(
                iam,
                validated_plan,
                state=state,
                source_hash=source_hash,
                commit=commit,
                account_id=account_id,
                caller_arn=caller_arn,
            )
        if operation != "apply":
            raise OperationError("IAM policy operation is unsupported")
        validated = _validate_request(
            request, commit=commit, source_hash=source_hash
        )
        if validated["target"]["kind"] == "inline_role":
            return _apply_inline(
                iam,
                validated,
                source_hash=source_hash,
                commit=commit,
                account_id=account_id,
                caller_arn=caller_arn,
            )
        return _apply_managed(
            iam,
            validated,
            source_hash=source_hash,
            commit=commit,
            account_id=account_id,
            caller_arn=caller_arn,
        )
    finally:
        for client in (iam, sts):
            close = getattr(client, "close", None)
            if close is not None:
                with contextlib.suppress(Exception):
                    close()


REMOTE_LOADER = r'''
import base64, contextlib, hashlib, io, json, os, sys
try:
    for name in (
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN", "AWS_PROFILE", "AWS_DEFAULT_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE", "AWS_CONFIG_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_ROLE_ARN",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "AWS_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_IAM", "AWS_ENDPOINT_URL_STS", "AWS_CA_BUNDLE",
        "AWS_DATA_PATH", "AWS_DEFAULT_REGION", "AWS_REGION",
        "AWS_SDK_LOAD_CONFIG", "AWS_STS_REGIONAL_ENDPOINTS",
        "AWS_USE_DUALSTACK_ENDPOINT", "AWS_USE_FIPS_ENDPOINT", "BOTO_CONFIG",
        "REQUESTS_CA_BUNDLE", "SSL_CERT_DIR", "SSL_CERT_FILE",
        "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
        "https_proxy", "http_proxy", "all_proxy",
    ):
        os.environ.pop(name, None)
    payload = json.loads(sys.stdin.buffer.read(1024 * 1024))
    sources = {}
    for item in payload["sources"]:
        source = base64.b64decode(item["source"], validate=True)
        if hashlib.sha256(source).hexdigest() != item["sha256"]:
            raise RuntimeError("source hash")
        sources[item["path"]] = source
    required = {
        "scripts/setup_production_parity_staging.py",
        "scripts/operate_rebenchmark_iam_policy.py",
    }
    if set(sources) != required:
        raise RuntimeError("source inventory")
    setup = {"__name__": "__leadpoet_iam_setup__",
             "__file__": "scripts/setup_production_parity_staging.py"}
    exec(compile(sources["scripts/setup_production_parity_staging.py"],
                 "scripts/setup_production_parity_staging.py", "exec"), setup)
    operator = {"__name__": "__leadpoet_iam_operator__",
                "__file__": "scripts/operate_rebenchmark_iam_policy.py"}
    exec(compile(sources["scripts/operate_rebenchmark_iam_policy.py"],
                 "scripts/operate_rebenchmark_iam_policy.py", "exec"), operator)
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        receipt = operator["_remote_entry"](
            payload["operation"], payload.get("request"), setup,
            source_hash=payload["bridge_source_hash"], commit=payload["commit"])
    if stdout.getvalue() or stderr.getvalue():
        raise RuntimeError("unexpected output")
    sys.stdout.write(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
except Exception:
    sys.stderr.write("REMOTE_REBENCHMARK_IAM_ERROR\n")
    raise SystemExit(1)
'''.strip()


def _run(*args: str, input_value: bytes | None = None, timeout: int = 120) -> bytes:
    environment = {
        name: os.environ[name]
        for name in ("PATH", "HOME", "LANG", "LC_ALL")
        if os.environ.get(name)
    }
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
        }
    )
    if set(environment) & _AWS_SELECTORS:
        raise OperationError("local AWS selectors reached the gateway bridge")
    try:
        result = subprocess.run(
            list(args),
            cwd=ROOT,
            env=environment,
            input=input_value,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
            start_new_session=True,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OperationError(f"command failed: {args[0]}") from exc
    if result.returncode != 0:
        raise OperationError(f"command failed: {args[0]}")
    return result.stdout


def _git(*args: str) -> bytes:
    return _run(
        GIT_BIN,
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "credential.helper=",
        "-c",
        "http.proxy=",
        "-c",
        "remote.origin.proxy=",
        *args,
    )


def _validate_ssh_key() -> None:
    try:
        before = SSH_KEY.lstat()
    except OSError as exc:
        raise OperationError("SSH key is unavailable") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or SSH_KEY.is_symlink()
        or before.st_uid != os.geteuid()
        or before.st_nlink != 1
        or before.st_mode & 0o077
    ):
        raise OperationError("SSH key metadata is unsafe")


def _exact_sources() -> tuple[str, list[dict[str, str]], str]:
    configured_url = _git(
        "config", "--local", "--get", "remote.origin.url"
    ).decode().strip()
    resolved_url = _git("remote", "get-url", "origin").decode().strip()
    if configured_url != EXPECTED_ORIGIN_URL or resolved_url != EXPECTED_ORIGIN_URL:
        raise OperationError("IAM bridge origin repository identity differs")
    _git(
        "fetch",
        "--no-tags",
        EXPECTED_ORIGIN_URL,
        "refs/heads/main:refs/remotes/origin/main",
    )
    commit = _git("rev-parse", "origin/main").decode().strip()
    head = _git("rev-parse", "HEAD").decode().strip()
    if not SHA_RE.fullmatch(commit) or head != commit:
        raise OperationError("IAM bridge checkout is not exact current origin/main")
    _git("diff", "--exit-code", commit, "--")
    if _git("status", "--porcelain=v1", "--untracked-files=all").strip():
        raise OperationError("IAM bridge worktree is not pristine")
    sources: list[dict[str, str]] = []
    commitments: dict[str, str] = {}
    for path in (SETUP_PATH, OPERATOR_PATH):
        source = _git("show", f"{commit}:{path}")
        local = (ROOT / path).read_bytes()
        if source != local:
            raise OperationError(f"local IAM bridge source differs: {path}")
        digest = hashlib.sha256(source).hexdigest()
        commitments[path] = "sha256:" + digest
        sources.append(
            {
                "path": path,
                "sha256": digest,
                "source": base64.b64encode(source).decode("ascii"),
            }
        )
    return commit, sources, _sha256_json(commitments)


def _read_fd(descriptor: int, *, limit: int) -> Any:
    if descriptor < 3:
        raise OperationError("request descriptor must be an inherited pipe or socket")
    try:
        mode = os.fstat(descriptor).st_mode
    except OSError as exc:
        raise OperationError("request descriptor is unavailable") from exc
    if not (stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode)) or os.isatty(descriptor):
        raise OperationError("request descriptor must be an inherited pipe or socket")
    if stat.S_ISSOCK(mode):
        duplicate = os.dup(descriptor)
        try:
            inherited_socket = socket.socket(fileno=duplicate)
            if inherited_socket.family != socket.AF_UNIX:
                raise OperationError("request socket must be local AF_UNIX")
        finally:
            try:
                inherited_socket.close()
            except UnboundLocalError:
                os.close(duplicate)
    chunks: list[bytes] = []
    total = 0
    deadline = time.monotonic() + FD_READ_TIMEOUT_SECONDS
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([descriptor], [], [], remaining)[0]:
            raise OperationError("IAM policy request pipe did not reach EOF")
        try:
            chunk = os.read(descriptor, 65536)
        except OSError as exc:
            raise OperationError("IAM policy request pipe failed") from exc
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise OperationError("IAM policy request is too large")
        chunks.append(chunk)
    try:
        return json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise OperationError("IAM policy request is invalid") from exc


@contextlib.contextmanager
def _active_ledger_lock(path: Path):
    lock_path = path.parent / f".{path.name}.lock"
    lock_descriptor: int | None = None
    try:
        lock_descriptor = os.open(
            lock_path,
            os.O_CREAT
            | os.O_RDWR
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        lock_metadata = os.fstat(lock_descriptor)
        if (
            not stat.S_ISREG(lock_metadata.st_mode)
            or lock_metadata.st_uid != os.geteuid()
            or lock_metadata.st_nlink != 1
        ):
            raise OperationError("active rebenchmark ledger lock metadata is unsafe")
        os.fchmod(lock_descriptor, 0o600)
        fcntl.flock(lock_descriptor, fcntl.LOCK_SH)
        yield
    except OSError as exc:
        raise OperationError("active rebenchmark ledger lock is unavailable") from exc
    finally:
        if lock_descriptor is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)


def _validate_active_ledger(
    path: Path,
    *,
    commit: str,
    source_hash: str,
    required_plan_hash: str | None = None,
    _lock_held: bool = False,
) -> None:
    if not _lock_held:
        with _active_ledger_lock(path):
            return _validate_active_ledger(
                path,
                commit=commit,
                source_hash=source_hash,
                required_plan_hash=required_plan_hash,
                _lock_held=True,
            )
    try:
        before = path.lstat()
    except OSError as exc:
        raise OperationError("active rebenchmark ledger is unavailable") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or path.is_symlink()
        or before.st_uid != os.geteuid()
        or before.st_nlink != 1
        or before.st_mode & 0o077
        or not 1 <= before.st_size <= MAX_LEDGER_BYTES
    ):
        raise OperationError("active rebenchmark ledger metadata is unsafe")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
                or opened.st_size != before.st_size
                or opened.st_uid != os.geteuid()
                or opened.st_nlink != 1
                or opened.st_mode & 0o077
            ):
                raise OperationError("active rebenchmark ledger changed during open")
            raw = bytearray()
            while len(raw) <= MAX_LEDGER_BYTES:
                chunk = os.read(descriptor, 65536)
                if not chunk:
                    break
                raw.extend(chunk)
            if len(raw) != opened.st_size:
                raise OperationError("active rebenchmark ledger read differs")
        finally:
            os.close(descriptor)
        value = json.loads(raw.decode("utf-8"))
        raw[:] = b"\0" * len(raw)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise OperationError("active rebenchmark ledger is unavailable") from exc
    if not isinstance(value, Mapping):
        raise OperationError("active rebenchmark ledger is invalid")
    route = value.get("iam_authority_route")
    invariant = value.get("iam_never_pause_invariant")
    routes = value.get("iam_authority_routes")
    plans = value.get("iam_policy_plans")
    plan_history = value.get("iam_policy_plan_history", [])
    changes = value.get("iam_policy_changes")
    if (
        value.get("schema_version") != LEDGER_SCHEMA
        or value.get("status") != "running"
        or not isinstance(route, Mapping)
        or route.get("schema_version") != AUTHORITY_SCHEMA
        or route.get("status") != "authority_ready"
        or route.get("origin_main_sha") != commit
        or route.get("bridge_source_hash") != source_hash
        or route.get("account_id") != EXPECTED_ACCOUNT_ID
        or route.get("caller_arn") != EXPECTED_CALLER_ARN
        or route.get("route") != "gateway_bridge"
        or route.get("local_chain") != "ignored_non_authority"
        or route.get("secret_values_printed") is not False
        or route.get("policy_material_printed") is not False
        or not isinstance(invariant, Mapping)
        or invariant.get("schema_version") != NEVER_PAUSE_SCHEMA
        or invariant.get("status") != "enforced"
        or invariant.get("local_chain_failure_disposition")
        != "ignored_non_authority"
        or invariant.get("blocks_recovery") is not False
        or invariant.get("operator_iam_request_allowed") is not False
        or not isinstance(value.get("generation"), int)
        or isinstance(value.get("generation"), bool)
        or value.get("generation") < 0
        or not isinstance(value.get("repo"), str)
        or not value.get("repo")
        or not isinstance(value.get("started_at"), str)
        or not isinstance(value.get("updated_at"), str)
        or not isinstance(value.get("stages"), list)
        or not isinstance(routes, list)
        or not routes
        or routes[-1] != route
        or not isinstance(plan_history, list)
        or not isinstance(plans, list)
        or not isinstance(changes, list)
    ):
        raise OperationError("typed IAM authority ledger gate is not satisfied")
    if required_plan_hash is not None:
        matching = [
            item
            for item in plans
            if isinstance(item, Mapping) and item.get("plan_hash") == required_plan_hash
        ]
        if len(matching) != 1:
            raise OperationError("typed IAM policy plan ledger gate is not satisfied")
        _validate_plan_receipt(
            matching[0], commit=commit, source_hash=source_hash
        )
        if any(
            isinstance(item, Mapping)
            and item.get("plan_hash") == required_plan_hash
            for item in changes
        ):
            raise OperationError("typed IAM policy plan is already completed")


def _remote_call(
    operation: str,
    request: Any,
    *,
    commit: str,
    sources: Sequence[Mapping[str, str]],
    source_hash: str,
) -> dict[str, Any]:
    _validate_ssh_key()
    payload = {
        "operation": operation,
        "request": request,
        "commit": commit,
        "sources": list(sources),
        "bridge_source_hash": source_hash,
    }
    command = "python3 -I -c " + shlex.quote(REMOTE_LOADER)
    output = _run(
        SSH_BIN,
        "-i",
        str(SSH_KEY),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "UserKnownHostsFile=" + str(SSH_KNOWN_HOSTS),
        "-o",
        "GlobalKnownHostsFile=/etc/ssh/ssh_known_hosts",
        "-F",
        "/dev/null",
        "-o",
        "ConnectTimeout=15",
        GATEWAY_HOST,
        command,
        input_value=_json(payload).encode("utf-8"),
        timeout=180,
    )
    try:
        value = json.loads(output)
    except (UnicodeDecodeError, ValueError) as exc:
        raise OperationError("gateway IAM receipt is invalid") from exc
    if not isinstance(value, dict):
        raise OperationError("gateway IAM receipt is invalid")
    return _validate_remote_receipt(
        operation,
        value,
        request=request,
        commit=commit,
        source_hash=source_hash,
    )


def _validate_remote_receipt(
    operation: str,
    value: Mapping[str, Any],
    *,
    request: Any,
    commit: str,
    source_hash: str,
) -> dict[str, Any]:
    authority_keys = {
        "schema_version",
        "status",
        "origin_main_sha",
        "bridge_source_hash",
        "account_id",
        "caller_arn",
        "route",
        "local_chain",
        "secret_values_printed",
        "policy_material_printed",
    }
    fixed = (
        value.get("origin_main_sha") == commit
        and value.get("bridge_source_hash") == source_hash
        and value.get("account_id") == EXPECTED_ACCOUNT_ID
        and value.get("caller_arn") == EXPECTED_CALLER_ARN
        and value.get("route") == "gateway_bridge"
        and value.get("local_chain") == "ignored_non_authority"
        and value.get("secret_values_printed") is False
        and value.get("policy_material_printed") is False
    )
    if operation == "probe":
        if (
            set(value) != authority_keys
            or value.get("schema_version") != AUTHORITY_SCHEMA
            or value.get("status") != "authority_ready"
            or not fixed
        ):
            raise OperationError("gateway IAM authority receipt differs")
        return dict(value)
    if operation == "plan":
        if not isinstance(request, Mapping):
            raise OperationError("gateway IAM plan receipt request is invalid")
        plan = _validate_plan_receipt(
            value, commit=commit, source_hash=source_hash
        )
        if (
            plan["change_id"] != request["change_id"]
            or plan["target"] != request["target"]
            or plan["desired_document_hash"]
            != _sha256_json(request["desired_document"])
            or plan["task_scope_hash"] != _sha256_json(request["task_scope"])
            or plan["simulation_case_count"] != len(request["simulations"])
        ):
            raise OperationError("gateway IAM plan receipt request binding differs")
        return plan
    if not isinstance(request, Mapping):
        raise OperationError("gateway IAM policy receipt request is invalid")
    common_keys = {
        "schema_version",
        "change_id",
        "target",
        "origin_main_sha",
        "bridge_source_hash",
        "account_id",
        "caller_arn",
        "route",
        "local_chain",
        "prior_document_hash",
        "desired_document_hash",
        "expected_delta_hash",
        "task_scope_hash",
        "plan_hash",
        "simulation_case_count",
        "fixed_decoy_case_count",
        "principal_simulation_count",
        "concurrency_model",
        "aws_native_compare_and_swap",
        "secret_values_printed",
        "policy_material_printed",
        "status",
        "readback_document_hash",
    }
    managed_keys = {"managed_version_id", "pruned_managed_version_id"}
    status = value.get("status")
    managed_updated = (
        request["target"]["kind"] == "managed"
        and status in {"updated", "reconciled"}
    )
    if (
        not common_keys.issubset(value)
        or set(value) - common_keys - managed_keys
        or (set(value) & managed_keys) != (managed_keys if managed_updated else set())
        or value.get("schema_version") != RECEIPT_SCHEMA
        or value.get("change_id") != request["change_id"]
        or value.get("target") != request["target"]
        or value.get("prior_document_hash")
        != request["expected_prior_document_hash"]
        or value.get("desired_document_hash")
        != _sha256_json(request["desired_document"])
        or value.get("expected_delta_hash")
        != _sha256_json(request["expected_delta"])
        or value.get("task_scope_hash") != _sha256_json(request["task_scope"])
        or value.get("plan_hash") != request["plan"]["plan_hash"]
        or value.get("simulation_case_count") != len(request["simulations"])
        or value.get("fixed_decoy_case_count")
        != request["plan"]["fixed_decoy_case_count"]
        or value.get("principal_simulation_count")
        != len(request["plan"]["principal_arns"])
        * (len(request["simulations"]) + request["plan"]["fixed_decoy_case_count"])
        or isinstance(value.get("principal_simulation_count"), bool)
        or value.get("concurrency_model") != CONCURRENCY_MODEL
        or value.get("aws_native_compare_and_swap") is not False
        or status not in {"updated", "unchanged", "reconciled"}
        or value.get("readback_document_hash")
        != value.get("desired_document_hash")
        or (status == "unchanged" and value.get("prior_document_hash")
            != value.get("desired_document_hash"))
        or not fixed
    ):
        raise OperationError("gateway IAM policy receipt differs")
    if managed_updated and (
        re.fullmatch(r"v[1-9][0-9]*", str(value.get("managed_version_id") or ""))
        is None
        or value.get("pruned_managed_version_id") is not None
    ):
        raise OperationError("gateway managed IAM policy receipt differs")
    return dict(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("probe")
    plan = commands.add_parser("plan")
    plan.add_argument("--request-fd", type=int, required=True)
    plan.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    apply = commands.add_parser("apply")
    apply.add_argument("--request-fd", type=int, required=True)
    apply.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        commit, sources, source_hash = _exact_sources()
        if args.command == "probe":
            request = None
            receipt = _remote_call(
                args.command,
                request,
                commit=commit,
                sources=sources,
                source_hash=source_hash,
            )
        else:
            if os.environ.get(AUTHORIZATION_ENV) != "1":
                raise OperationError("active overnight rebenchmark authorization is required")
            if args.ledger.expanduser().resolve() != DEFAULT_LEDGER.resolve():
                raise OperationError("active overnight rebenchmark ledger path differs")
            raw_request = _read_fd(args.request_fd, limit=MAX_REQUEST_BYTES)
            # Re-fetch after the bounded pipe read so a held-open writer cannot
            # turn an old origin/main and ledger check into later write authority.
            refreshed_commit, refreshed_sources, refreshed_hash = _exact_sources()
            if (
                refreshed_commit != commit
                or refreshed_hash != source_hash
                or refreshed_sources != sources
            ):
                raise OperationError("IAM bridge origin changed during request admission")
            if args.command == "plan":
                request = _validate_plan_request(raw_request)
                required_plan_hash = None
            else:
                request = _validate_request(
                    raw_request, commit=commit, source_hash=source_hash
                )
                required_plan_hash = request["plan"]["plan_hash"]
            with _active_ledger_lock(args.ledger):
                _validate_active_ledger(
                    args.ledger,
                    commit=commit,
                    source_hash=source_hash,
                    required_plan_hash=required_plan_hash,
                    _lock_held=True,
                )
                receipt = _remote_call(
                    args.command,
                    request,
                    commit=commit,
                    sources=sources,
                    source_hash=source_hash,
                )
    except OperationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(_json(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
