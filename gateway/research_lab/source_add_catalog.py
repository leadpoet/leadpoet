"""SOURCE_ADD catalog/provisioning helpers."""

from __future__ import annotations

import base64
import binascii
import logging
import os
import re
from typing import Any, Mapping, Sequence

from gateway.db.client import get_write_client
from gateway.research_lab.key_vault import decrypt_source_add_credential
from research_lab.canonical import sha256_json
from research_lab.probe_catalog import ProviderProbeEndpoint, validate_probe_catalog

logger = logging.getLogger(__name__)

PROVISION_STATUS_APPROVED_PENDING = "approved_pending_provision"
PROVISION_STATUS_ELIGIBLE = "provisioned_autoresearch_eligible"
PROVISION_STATUS_DISABLED = "disabled"
PROVISION_STATUSES = {
    PROVISION_STATUS_APPROVED_PENDING,
    PROVISION_STATUS_ELIGIBLE,
    PROVISION_STATUS_DISABLED,
}

ALREADY_SUBMITTED_DETAIL = "Already submitted"
_SECRET_RE = re.compile(r"(?i)(sk-or-|sb_secret|service_role|raw_secret|password|api[_-]?key\s*=)")
_ENV_REF_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,127}$")
_TRUTHY = {"1", "true", "yes", "on"}


def source_add_env_ref_resolves(env_name: str) -> bool:
    ref = str(env_name or "").strip()
    if not ref or ref.startswith("encrypted_ref:") or not _ENV_REF_RE.fullmatch(ref):
        return False
    key_split = str(os.getenv("RESEARCH_LAB_PROVIDER_KEY_SPLIT", "") or "").strip().lower() in _TRUTHY
    if key_split and not ref.startswith("RESEARCH_LAB_"):
        return False
    return bool(str(os.getenv(ref) or "").strip())


def source_add_encrypted_envelope_valid(envelope: Mapping[str, Any]) -> bool:
    ciphertext = str(envelope.get("ciphertext_b64") or "").strip()
    kms_key_id = str(envelope.get("kms_key_id") or "").strip()
    credential_ref = str(envelope.get("credential_ref") or "").strip()
    if not ciphertext or not kms_key_id or not credential_ref.startswith("encrypted_ref:source_add:"):
        return False
    try:
        decoded = base64.b64decode(ciphertext, validate=True)
    except (binascii.Error, ValueError):
        return False
    return 8 <= len(decoded) <= 16_384


def reject_source_add_secret_text(value: Any, *, field_name: str = "value") -> None:
    text = str(value or "")
    if _SECRET_RE.search(text):
        raise ValueError(f"{field_name} contains secret-like material")


def sanitize_source_add_doc(doc: Mapping[str, Any]) -> dict[str, Any]:
    def clean(value: Any) -> Any:
        if isinstance(value, Mapping):
            out: dict[str, Any] = {}
            for key, item in value.items():
                key_text = str(key)[:80]
                lowered = key_text.lower()
                if lowered == "credential_ref":
                    out[key_text] = clean(item)
                elif any(marker in lowered for marker in ("password", "secret", "token", "api_key", "credential")):
                    out[key_text] = "[redacted]"
                else:
                    out[key_text] = clean(item)
            return out
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value[:50]]
        if isinstance(value, str):
            if _SECRET_RE.search(value):
                return "[redacted]"
            return value[:4000]
        return value

    return clean(dict(doc))


def provisioning_ref(adapter_id: str, seq: int) -> str:
    return "source_add_provision:" + sha256_json({"adapter_id": adapter_id, "seq": int(seq)}).split(":", 1)[1][:16]


def source_add_row_credential_ready(row: Mapping[str, Any]) -> bool:
    """True when an eligible SOURCE_ADD provider can authenticate right now."""

    doc = row.get("provision_doc") if isinstance(row.get("provision_doc"), Mapping) else {}
    raw_entry = doc.get("provider_registry_entry") if isinstance(doc.get("provider_registry_entry"), Mapping) else {}
    auth_kind = str(raw_entry.get("auth_kind") or "none").strip().lower()
    if auth_kind == "none":
        return True
    envelope = row.get("credential_envelope") if isinstance(row.get("credential_envelope"), Mapping) else {}
    if source_add_encrypted_envelope_valid(envelope):
        return True
    refs = raw_entry.get("credential_ref") or ()
    if isinstance(refs, str):
        refs = (refs,)
    for ref in refs:
        env_name = str(ref or "").strip()
        if source_add_env_ref_resolves(env_name):
            return True
    return False


def load_provisioned_source_rows_sync(*, raise_on_error: bool = False) -> list[dict[str, Any]]:
    """Best-effort sync loader for runtime catalog extension."""

    try:
        client = get_write_client()
        rows: list[dict[str, Any]] = []
        batch_size = 500
        max_rows = 5000
        for offset in range(0, max_rows, batch_size):
            response = (
                client.table("research_lab_source_add_provisioning_current")
                .select("*")
                .eq("provision_status", PROVISION_STATUS_ELIGIBLE)
                .range(offset, offset + batch_size - 1)
                .execute()
            )
            batch = [dict(row) for row in (getattr(response, "data", None) or [])]
            rows.extend(batch)
            if len(batch) < batch_size:
                break
        else:
            raise RuntimeError(f"provisioned SOURCE_ADD catalog exceeded max_rows={max_rows}")
        ready_rows = [row for row in rows if source_add_row_credential_ready(row)]
        omitted = len(rows) - len(ready_rows)
        if omitted:
            logger.warning(
                "source_add_provisioned_catalog_credentials_unresolved omitted_count=%d",
                omitted,
            )
        return ready_rows
    except Exception as exc:  # noqa: BLE001 - dynamic sources must not hide static providers
        if raise_on_error:
            raise
        logger.warning("source_add_provisioned_catalog_load_failed error=%s", str(exc)[:200])
        return []


def provider_registry_entries_from_provisioned_rows(rows: Sequence[Mapping[str, Any]]) -> list[Any]:
    from gateway.research_lab.provider_evidence_proxy import ProviderRegistryEntry

    entries: list[ProviderRegistryEntry] = []
    for row in rows:
        doc = row.get("provision_doc") if isinstance(row.get("provision_doc"), Mapping) else {}
        raw_entry = doc.get("provider_registry_entry") if isinstance(doc.get("provider_registry_entry"), Mapping) else {}
        if not raw_entry:
            continue
        entry_doc = dict(raw_entry)
        credential_envelope = row.get("credential_envelope") if isinstance(row.get("credential_envelope"), Mapping) else {}
        cost_model = dict(entry_doc.get("cost_model") or {})
        if credential_envelope:
            cost_model["source_add_credential_envelope"] = dict(credential_envelope)
            cost_model["source_add_miner_hotkey"] = str(row.get("miner_hotkey") or "")
            cost_model["source_add_adapter_ref"] = f"source_add:{str(row.get('adapter_id') or '')}"
        entry_doc["cost_model"] = cost_model
        entries.append(ProviderRegistryEntry.from_mapping(entry_doc))
    return entries


def probe_endpoints_from_provisioned_rows(rows: Sequence[Mapping[str, Any]]) -> list[ProviderProbeEndpoint]:
    endpoints: list[ProviderProbeEndpoint] = []
    for row in rows:
        doc = row.get("provision_doc") if isinstance(row.get("provision_doc"), Mapping) else {}
        raw = doc.get("probe_endpoints") if isinstance(doc.get("probe_endpoints"), list) else []
        for item in raw:
            if isinstance(item, Mapping):
                endpoints.append(ProviderProbeEndpoint.from_mapping(item))
    errors = validate_probe_catalog(endpoints) if endpoints else []
    if errors:
        logger.warning("source_add_provisioned_probe_catalog_invalid errors=%s", "; ".join(errors[:5]))
        return []
    return endpoints


def decrypt_source_add_registry_credential(entry: Any) -> tuple[str, str]:
    envelope = entry.cost_model.get("source_add_credential_envelope")
    if not isinstance(envelope, Mapping) or not envelope.get("ciphertext_b64"):
        return "", ""
    miner_hotkey = str(entry.cost_model.get("source_add_miner_hotkey") or "")
    adapter_ref = str(entry.cost_model.get("source_add_adapter_ref") or "")
    if not miner_hotkey or not adapter_ref:
        return "", ""
    value = decrypt_source_add_credential(
        ciphertext_b64=str(envelope.get("ciphertext_b64") or ""),
        miner_hotkey=miner_hotkey,
        adapter_ref=adapter_ref,
    )
    return value, str(envelope.get("credential_ref") or "encrypted_ref:source_add")
