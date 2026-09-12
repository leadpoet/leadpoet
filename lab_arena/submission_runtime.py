"""Select the immutable run payer, never from model input."""

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

    def _provider_funding(
        self, context: RunContext, provider: str
    ) -> Mapping[str, Any]:
        if provider not in ("openrouter", "deepline", "scrapingdog"):
            raise BrokerError("miner_provider_not_configured")
        try:
            funding = self._store.provider_funding(context.run_id, provider)
        except Exception as exc:
            raise BrokerError("broker_unavailable") from exc
        required = {
            "status",
            "funding_source",
            "champion_funding",
            "credential_submission_id",
            "credential_miner_hotkey",
            "restart_required",
        }
        if (
            not isinstance(funding, Mapping)
            or set(funding) != required
            or funding.get("status") != "available"
            or funding.get("funding_source") not in ("host", "miner_key")
            or not isinstance(funding.get("champion_funding"), bool)
            or not isinstance(funding.get("restart_required"), bool)
        ):
            raise BrokerError("broker_unavailable")
        if funding["funding_source"] == "host":
            if (
                funding.get("credential_submission_id") is not None
                or funding.get("credential_miner_hotkey") is not None
            ):
                raise BrokerError("broker_unavailable")
        elif not all(
            isinstance(funding.get(field), str) and funding.get(field)
            for field in (
                "credential_submission_id",
                "credential_miner_hotkey",
            )
        ):
            raise BrokerError("broker_unavailable")
        return funding

    def provider_funding_source_for(
        self, context: RunContext, provider: str
    ) -> str:
        """Return the immutable payer selected for this run and provider."""

        self._submission(context)
        return str(self._provider_funding(context, provider)["funding_source"])

    def retry_miner_credential_for(self, context: RunContext) -> bool:
        """Retry credentials only for the daily champion's execution."""

        return context.kind == "execute" and self._is_baseline(
            self._submission(context)
        )

    def provider_restart_required_for(
        self, context: RunContext, provider: str
    ) -> bool:
        """Stop an old miner-funded run after another ICP latched fallback."""

        self._submission(context)
        return self._provider_funding(context, provider)["restart_required"] is True

    def mark_provider_fallback(
        self,
        context: RunContext,
        provider: str,
        evidence: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        try:
            return self._store.mark_champion_provider_fallback(
                context.run_id,
                context.lease_token_hash,
                provider,
                evidence,
            )
        except Exception as exc:
            raise BrokerError("broker_unavailable") from exc

    def credential_for(self, context: RunContext, provider: str) -> str:
        row = self._submission(context)
        funding = self._provider_funding(context, provider)
        if funding["funding_source"] == "host":
            secret = self._organizer_keys.get(provider)
            if not secret:
                raise BrokerError("broker_unavailable")
            return secret
        return self._miner_key_for_identity(
            str(funding["credential_submission_id"]),
            str(funding["credential_miner_hotkey"]),
            provider,
        )

    def code_review_key(self, submission: Mapping[str, Any]) -> str:
        """Review only with the accepted source owner's stored OpenRouter key."""
        row = self._store.get_submission(str(submission["submission_id"]))
        if (
            row is None or row.get("status") not in ("accepted", "frozen")
            or row.get("miner_hotkey") != submission.get("miner_hotkey")
            or self._is_baseline(row)
        ):
            raise BrokerError("miner_credentials_unavailable")
        return self._miner_key(row, "openrouter")

    def _miner_key(self, row: Mapping[str, Any], provider: str) -> str:
        return self._miner_key_for_identity(
            str(row["submission_id"]), str(row["miner_hotkey"]), provider
        )

    def _miner_key_for_identity(
        self, submission_id: str, miner_hotkey: str, provider: str
    ) -> str:
        if provider not in ("openrouter", "deepline", "scrapingdog"):
            raise BrokerError("miner_provider_not_configured")
        if self._credentials is None:
            raise BrokerError("miner_credentials_unavailable")
        encrypted = self._store.get_submission_credential(
            submission_id, miner_hotkey, provider
        )
        if (
            not encrypted
            or encrypted.get("submission_id") != submission_id
            or encrypted.get("miner_hotkey") != miner_hotkey
            or encrypted.get("provider") != provider
        ):
            raise BrokerError("miner_credentials_unavailable")
        try:
            return self._credentials.runtime_key(encrypted, provider)
        except CredentialError as exc:
            raise BrokerError("broker_unavailable" if exc.retryable else "miner_credentials_unavailable") from None
