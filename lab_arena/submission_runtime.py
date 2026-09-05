"""Select the payer from the accepted submission, never from model input."""

from typing import Any, Mapping

from lab_arena.broker import BrokerError, RunContext
from lab_arena.credentials import CredentialError


class SubmissionProviderKeys:
    """Resolve runtime keys only. Management keys have no provider route."""

    def __init__(self, *, store: Any, credentials: Any, organizer_keys: Mapping[str, str]) -> None:
        self._store = store
        self._credentials = credentials
        self._organizer_keys = dict(organizer_keys)

    def _submission(self, context: RunContext) -> Mapping[str, Any]:
        row = self._store.get_submission(context.submission_id)
        if (
            row is None
            or row.get("miner_hotkey") != context.miner_hotkey
            or row.get("status") not in ("accepted", "frozen")
        ):
            raise BrokerError("broker_unavailable")
        return row

    def _is_baseline(self, row: Mapping[str, Any]) -> bool:
        round_id = str(row.get("round_id") or "")
        if row.get("submission_id") != "baseline-" + round_id.removeprefix("arena-"):
            return False
        round_row = self._store.get_round(round_id)
        configured_hotkey = ((round_row or {}).get("configuration_doc") or {}).get("baseline_hotkey")
        return bool(configured_hotkey and row.get("miner_hotkey") == configured_hotkey and row.get("is_king"))

    def funding_source_for(self, context: RunContext) -> str:
        return "host" if self._is_baseline(self._submission(context)) else "miner_key"

    def credential_for(self, context: RunContext, provider: str) -> str:
        row = self._submission(context)
        if self._is_baseline(row):
            secret = self._organizer_keys.get(provider)
            if not secret:
                raise BrokerError("broker_unavailable")
            return secret
        if provider not in ("openrouter", "deepline"):
            raise BrokerError("miner_provider_not_configured")
        if self._credentials is None:
            raise BrokerError("miner_credentials_unavailable")
        encrypted = self._store.get_submission_credential(context.submission_id, context.miner_hotkey, provider)
        if (
            not encrypted
            or encrypted.get("submission_id") != context.submission_id
            or encrypted.get("miner_hotkey") != context.miner_hotkey
            or encrypted.get("provider") != provider
        ):
            raise BrokerError("miner_credentials_unavailable")
        try:
            return self._credentials.runtime_key(encrypted, provider)
        except CredentialError as exc:
            raise BrokerError("broker_unavailable" if exc.retryable else "miner_credentials_unavailable") from None
