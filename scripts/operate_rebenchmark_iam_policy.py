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
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
OPERATOR_PATH = "scripts/operate_rebenchmark_iam_policy.py"
SETUP_PATH = "scripts/setup_production_parity_staging.py"
GATEWAY_HOST = "ec2-user@52.91.135.79"
SSH_KEY = Path("/Users/pranav/Downloads/leadpoet-2026-07-28.pem")
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
        "NotAction",
        "Resource",
        "NotResource",
        "Condition",
    }
    for statement in statements:
        if not isinstance(statement, Mapping) or not set(statement).issubset(
            allowed_keys
        ):
            raise OperationError("IAM identity-policy statement is invalid")
        if statement.get("Effect") not in {"Allow", "Deny"}:
            raise OperationError("IAM policy effect is invalid")
        if ("Action" in statement) == ("NotAction" in statement):
            raise OperationError("IAM policy action selector is invalid")
        if ("Resource" in statement) == ("NotResource" in statement):
            raise OperationError("IAM policy resource selector is invalid")
        item: dict[str, Any] = {"Effect": str(statement["Effect"])}
        if "Sid" in statement:
            sid = statement["Sid"]
            if not isinstance(sid, str) or not re.fullmatch(
                r"[A-Za-z0-9]{1,128}", sid
            ):
                raise OperationError("IAM policy Sid is invalid")
            item["Sid"] = sid
        for key in ("Action", "NotAction", "Resource", "NotResource"):
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


def _pointer(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _structural_delta(before: Any, after: Any, path: str = "") -> list[dict[str, Any]]:
    if before == after:
        return []
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        output: list[dict[str, Any]] = []
        for key in sorted(set(before) | set(after), key=str):
            child = path + "/" + _pointer(str(key))
            if key not in before:
                output.append(
                    {
                        "op": "add",
                        "path": child,
                        "before_hash": None,
                        "after_hash": _sha256_json(after[key]),
                    }
                )
            elif key not in after:
                output.append(
                    {
                        "op": "remove",
                        "path": child,
                        "before_hash": _sha256_json(before[key]),
                        "after_hash": None,
                    }
                )
            else:
                output.extend(_structural_delta(before[key], after[key], child))
        return output
    if isinstance(before, list) and isinstance(after, list):
        output = []
        common = min(len(before), len(after))
        for index in range(common):
            output.extend(
                _structural_delta(before[index], after[index], f"{path}/{index}")
            )
        for index in range(common, len(before)):
            output.append(
                {
                    "op": "remove",
                    "path": f"{path}/{index}",
                    "before_hash": _sha256_json(before[index]),
                    "after_hash": None,
                }
            )
        for index in range(common, len(after)):
            output.append(
                {
                    "op": "add",
                    "path": f"{path}/{index}",
                    "before_hash": None,
                    "after_hash": _sha256_json(after[index]),
                }
            )
        return output
    return [
        {
            "op": "replace",
            "path": path or "/",
            "before_hash": _sha256_json(before),
            "after_hash": _sha256_json(after),
        }
    ]


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
            or not path.startswith("/Statement/")
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
        if ROLE_NAME_RE.fullmatch(role) is None or POLICY_NAME_RE.fullmatch(name) is None:
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
            or not re.fullmatch(r"[a-z0-9-]+:[A-Za-z0-9*]+", action)
            or not isinstance(resources, list)
            or not resources
            or len(resources) > 16
            or any(not isinstance(resource, str) or not resource for resource in resources)
            or not isinstance(context, Mapping)
            or expected not in {"allowed", "implicitDeny", "explicitDeny"}
        ):
            raise OperationError("IAM policy simulation case is invalid")
        normalized_context: dict[str, list[str]] = {}
        for key, values in context.items():
            if not isinstance(key, str) or not key:
                raise OperationError("IAM simulation context is invalid")
            values = [values] if isinstance(values, str) else values
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(entry, str) for entry in values)
            ):
                raise OperationError("IAM simulation context is invalid")
            normalized_context[key] = list(values)
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


def _validate_request(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "change_id",
        "target",
        "expected_prior_document_hash",
        "expected_inventory_hash",
        "desired_document",
        "expected_delta",
        "simulations",
        "prune_managed_version",
    }:
        raise OperationError("IAM policy change request fields are invalid")
    if value.get("schema_version") != REQUEST_SCHEMA:
        raise OperationError("IAM policy change request schema is invalid")
    change_id = str(value.get("change_id") or "")
    prior_hash = str(value.get("expected_prior_document_hash") or "")
    inventory_hash = str(value.get("expected_inventory_hash") or "")
    if (
        CHANGE_ID_RE.fullmatch(change_id) is None
        or HASH_RE.fullmatch(prior_hash) is None
        or HASH_RE.fullmatch(inventory_hash) is None
    ):
        raise OperationError("IAM policy change request identity is invalid")
    prune = value.get("prune_managed_version")
    if prune is not None:
        raise OperationError("pre-existing managed IAM policy versions cannot be pruned")
    return {
        "schema_version": REQUEST_SCHEMA,
        "change_id": change_id,
        "target": _target(value["target"]),
        "expected_prior_document_hash": prior_hash,
        "expected_inventory_hash": inventory_hash,
        "desired_document": _canonical_policy(value["desired_document"]),
        "expected_delta": _validate_expected_delta(value["expected_delta"]),
        "simulations": _validate_simulations(value["simulations"]),
        "prune_managed_version": prune,
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


def _context_entries(context: Mapping[str, Sequence[str]]) -> list[dict[str, Any]]:
    return [
        {
            "ContextKeyName": key,
            "ContextKeyValues": list(values),
            "ContextKeyType": "string",
        }
        for key, values in sorted(context.items())
    ]


def _simulate(iam: Any, document: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> None:
    for case in cases:
        response = iam.simulate_custom_policy(
            PolicyInputList=[_json(document)],
            ActionNames=[case["action"]],
            ResourceArns=list(case["resources"]),
            ContextEntries=_context_entries(case["context"]),
        )
        if response.get("IsTruncated"):
            raise OperationError("IAM custom-policy simulation was truncated")
        results = response.get("EvaluationResults")
        if not isinstance(results, list) or not results:
            raise OperationError("IAM custom-policy simulation response is invalid")
        decisions: dict[str, str] = {}
        missing: set[str] = set()
        for result in results:
            if not isinstance(result, Mapping) or result.get("EvalActionName") != case["action"]:
                raise OperationError("IAM custom-policy simulation response is invalid")
            missing.update(str(value) for value in result.get("MissingContextValues", []))
            specific = result.get("ResourceSpecificResults") or []
            if specific:
                for item in specific:
                    resource = str(item.get("EvalResourceName") or "")
                    decision = str(item.get("EvalResourceDecision") or "")
                    decisions[resource] = decision
            else:
                resource = str(result.get("EvalResourceName") or "")
                decision = str(result.get("EvalDecision") or "")
                decisions[resource] = decision
        if (
            missing
            or set(decisions) != set(case["resources"])
            or set(decisions.values()) != {case["expected"]}
        ):
            raise OperationError(f"IAM policy simulation {case['name']} differs")


def _check_preconditions(state: Mapping[str, Any], request: Mapping[str, Any]) -> None:
    if state["document_hash"] != request["expected_prior_document_hash"]:
        raise OperationError("IAM policy prior document hash differs")
    if state["inventory_hash"] != request["expected_inventory_hash"]:
        raise OperationError("IAM policy surrounding inventory hash differs")
    actual_delta = _policy_delta(state["document"], request["desired_document"])
    if len(actual_delta) > MAX_DELTA_ITEMS or actual_delta != request["expected_delta"]:
        raise OperationError("IAM policy structural delta differs")


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
        "simulation_case_count": len(request["simulations"]),
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
    _check_preconditions(before, request)
    _simulate(iam, request["desired_document"], request["simulations"])
    if before["document_hash"] == _sha256_json(request["desired_document"]):
        receipt = _receipt_base(
            request=request,
            source_hash=source_hash,
            commit=commit,
            account_id=account_id,
            caller_arn=caller_arn,
        )
        receipt.update(status="unchanged", readback_document_hash=before["document_hash"])
        return receipt
    immediate = _inline_state(iam, target)
    if (
        immediate["document_hash"] != before["document_hash"]
        or immediate["target_present"] != before["target_present"]
        or immediate["inventory_hash"] != before["inventory_hash"]
    ):
        raise OperationError("IAM inline policy changed before write")
    iam.put_role_policy(
        RoleName=target["role_name"],
        PolicyName=target["policy_name"],
        PolicyDocument=_json(request["desired_document"]),
    )
    after = _inline_state(iam, target)
    desired_hash = _sha256_json(request["desired_document"])
    if (
        after["document_hash"] != desired_hash
        or after["target_present"] is not True
        or after["inventory_hash"] != before["inventory_hash"]
    ):
        raise OperationError("IAM inline policy entered an unexpected concurrent state")
    try:
        _simulate(iam, request["desired_document"], request["simulations"])
    except OperationError as exc:
        live = _inline_state(iam, target)
        if (
            live["document_hash"] != desired_hash
            or live["target_present"] is not True
            or live["inventory_hash"] != before["inventory_hash"]
        ):
            raise OperationError(
                "IAM inline policy post-write simulation found a third state"
            ) from exc
        if before["target_present"]:
            iam.put_role_policy(
                RoleName=target["role_name"],
                PolicyName=target["policy_name"],
                PolicyDocument=_json(before["document"]),
            )
        else:
            iam.delete_role_policy(
                RoleName=target["role_name"], PolicyName=target["policy_name"]
            )
        rolled_back = _inline_state(iam, target)
        if (
            rolled_back["document_hash"] != before["document_hash"]
            or rolled_back["target_present"] != before["target_present"]
            or rolled_back["inventory_hash"] != before["inventory_hash"]
        ):
            raise OperationError("IAM inline policy rollback readback differs") from exc
        raise OperationError("IAM inline policy post-write simulation rolled back") from exc
    receipt = _receipt_base(
        request=request,
        source_hash=source_hash,
        commit=commit,
        account_id=account_id,
        caller_arn=caller_arn,
    )
    receipt.update(status="updated", readback_document_hash=after["document_hash"])
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
    before = _managed_state(iam, target)
    _check_preconditions(before, request)
    _simulate(iam, request["desired_document"], request["simulations"])
    desired_hash = _sha256_json(request["desired_document"])
    if before["document_hash"] == desired_hash:
        receipt = _receipt_base(
            request=request,
            source_hash=source_hash,
            commit=commit,
            account_id=account_id,
            caller_arn=caller_arn,
        )
        receipt.update(status="unchanged", readback_document_hash=before["document_hash"])
        return receipt
    immediate = _managed_state(iam, target)
    if (
        immediate["document_hash"] != before["document_hash"]
        or immediate["inventory_hash"] != before["inventory_hash"]
    ):
        raise OperationError("managed IAM policy changed before write")
    versions = _version_map(immediate)
    if len(versions) >= 5:
        raise OperationError(
            "managed IAM policy capacity cannot preserve an exact rollback version"
        )
    created = iam.create_policy_version(
        PolicyArn=arn,
        PolicyDocument=_json(request["desired_document"]),
        SetAsDefault=False,
    ).get("PolicyVersion", {})
    new_version = str(created.get("VersionId") or "")
    if not re.fullmatch(r"v[1-9][0-9]*", new_version):
        raise OperationError("managed IAM policy created version is invalid")
    staged = _managed_state(iam, target)
    if (
        staged["default_version_id"] != before["default_version_id"]
        or not _managed_inventory_matches_addition(
            before,
            staged,
            new_version=new_version,
            new_hash=desired_hash,
            default_version=before["default_version_id"],
        )
    ):
        current = _managed_state(iam, target)
        if (
            current["default_version_id"] == before["default_version_id"]
            and _managed_inventory_matches_addition(
                before,
                current,
                new_version=new_version,
                new_hash=desired_hash,
                default_version=before["default_version_id"],
            )
        ):
            iam.delete_policy_version(PolicyArn=arn, VersionId=new_version)
            cleaned = _managed_state(iam, target)
            expected_versions = _version_map(before)
            if (
                cleaned["document_hash"] != before["document_hash"]
                or cleaned["inventory"]["stable"] != before["inventory"]["stable"]
                or _version_map(cleaned) != expected_versions
            ):
                raise OperationError("managed IAM policy staged cleanup differs")
            raise OperationError("managed IAM policy staged verification cleaned up")
        raise OperationError("managed IAM policy staged version entered a third state")
    iam.set_default_policy_version(PolicyArn=arn, VersionId=new_version)
    after = _managed_state(iam, target)
    if (
        after["default_version_id"] != new_version
        or after["document_hash"] != desired_hash
        or not _managed_inventory_matches_addition(
            before,
            after,
            new_version=new_version,
            new_hash=desired_hash,
            default_version=new_version,
        )
    ):
        raise OperationError("managed IAM policy entered an unexpected concurrent state")
    try:
        _simulate(iam, request["desired_document"], request["simulations"])
    except OperationError as exc:
        live = _managed_state(iam, target)
        if (
            live["default_version_id"] != new_version
            or live["document_hash"] != desired_hash
            or not _managed_inventory_matches_addition(
                before,
                live,
                new_version=new_version,
                new_hash=desired_hash,
                default_version=new_version,
            )
        ):
            raise OperationError(
                "managed IAM policy post-write simulation found a third state"
            ) from exc
        iam.set_default_policy_version(
            PolicyArn=arn, VersionId=before["default_version_id"]
        )
        reverted = _managed_state(iam, target)
        if reverted["default_version_id"] != before["default_version_id"]:
            raise OperationError("managed IAM policy rollback default differs") from exc
        iam.delete_policy_version(PolicyArn=arn, VersionId=new_version)
        rolled_back = _managed_state(iam, target)
        expected_versions = _version_map(before)
        if (
            rolled_back["document_hash"] != before["document_hash"]
            or rolled_back["inventory"]["stable"] != before["inventory"]["stable"]
            or _version_map(rolled_back) != expected_versions
        ):
            raise OperationError("managed IAM policy rollback readback differs") from exc
        raise OperationError("managed IAM policy post-write simulation rolled back") from exc
    receipt = _receipt_base(
        request=request,
        source_hash=source_hash,
        commit=commit,
        account_id=account_id,
        caller_arn=caller_arn,
    )
    receipt.update(
        status="updated",
        readback_document_hash=after["document_hash"],
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
        if operation != "apply":
            raise OperationError("IAM policy operation is unsupported")
        validated = _validate_request(request)
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
    for name in operator["_AWS_SELECTORS"]:
        os.environ.pop(name, None)
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
    _run(
        "git",
        "fetch",
        "--no-tags",
        "origin",
        "refs/heads/main:refs/remotes/origin/main",
    )
    commit = _run("git", "rev-parse", "origin/main").decode().strip()
    head = _run("git", "rev-parse", "HEAD").decode().strip()
    if not SHA_RE.fullmatch(commit) or head != commit:
        raise OperationError("IAM bridge checkout is not exact current origin/main")
    _run("git", "diff", "--exit-code", commit, "--")
    sources: list[dict[str, str]] = []
    commitments: dict[str, str] = {}
    for path in (SETUP_PATH, OPERATOR_PATH):
        source = _run("git", "show", f"{commit}:{path}")
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
    mode = os.fstat(descriptor).st_mode
    if not (stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode)) or os.isatty(descriptor):
        raise OperationError("request descriptor must be an inherited pipe or socket")
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(descriptor, 65536)
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


def _validate_active_ledger(path: Path, *, commit: str, source_hash: str) -> None:
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
    ):
        raise OperationError("typed IAM authority ledger gate is not satisfied")


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
        "ssh",
        "-i",
        str(SSH_KEY),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
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
        "simulation_case_count",
        "concurrency_model",
        "aws_native_compare_and_swap",
        "secret_values_printed",
        "policy_material_printed",
        "status",
        "readback_document_hash",
    }
    managed_keys = {"managed_version_id", "pruned_managed_version_id"}
    status = value.get("status")
    managed_updated = request["target"]["kind"] == "managed" and status == "updated"
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
        or value.get("simulation_case_count") != len(request["simulations"])
        or value.get("concurrency_model") != CONCURRENCY_MODEL
        or value.get("aws_native_compare_and_swap") is not False
        or status not in {"updated", "unchanged"}
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
        else:
            if os.environ.get(AUTHORIZATION_ENV) != "1":
                raise OperationError("active overnight rebenchmark authorization is required")
            _validate_active_ledger(args.ledger, commit=commit, source_hash=source_hash)
            request = _validate_request(
                _read_fd(args.request_fd, limit=MAX_REQUEST_BYTES)
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
