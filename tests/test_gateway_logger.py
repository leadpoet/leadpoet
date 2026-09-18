import pytest

from gateway.db import client as database_client
from gateway.utils import logger as gateway_logger
from gateway.utils import tee_client as tee_client_module


class _Coordinator:
    async def get_event_signing_identity(self):
        return {
            "purpose": "gateway_event_signing",
            "enclave_pubkey": "c" * 64,
            "attestation_document_b64": "attestation",
        }


def _deny_database_access(monkeypatch) -> None:
    def unexpected_database_access(*args, **kwargs):
        raise AssertionError("retired relational audit database was accessed")

    monkeypatch.setattr(
        database_client,
        "get_read_client",
        unexpected_database_access,
    )
    monkeypatch.setattr(
        database_client,
        "get_write_client",
        unexpected_database_access,
    )
    monkeypatch.setattr(
        database_client,
        "get_async_read_client",
        unexpected_database_access,
    )
    monkeypatch.setattr(
        database_client,
        "get_async_write_client",
        unexpected_database_access,
    )


@pytest.mark.asyncio
async def test_gateway_reads_measured_identity_without_relational_state(monkeypatch):
    coordinator = _Coordinator()
    _deny_database_access(monkeypatch)
    monkeypatch.setattr(tee_client_module, "tee_client", coordinator)

    identity = await gateway_logger.initialize_enclave_identity()

    assert identity["purpose"] == "gateway_event_signing"
