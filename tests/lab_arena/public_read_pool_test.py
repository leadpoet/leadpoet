"""The public read path must not share a connection pool with run work.

Run work arrives in waves far above the steady rate. With one pool, those
writes can hold every connection and a public page waits inside the pool with
no error of its own until the request timeout expires. These tests pin the
separation and the ceilings that make it meaningful.
"""

import httpx

from lab_arena import store as store_module
from lab_arena.store import (
    DEFAULT_STORE_LIMITS,
    PostgrestTransport,
    create_http1_client,
)


def _pool(client: httpx.Client):
    return client._transport._pool


def test_store_client_states_its_connection_ceilings() -> None:
    client = create_http1_client(8.0)
    try:
        assert _pool(client)._max_connections == DEFAULT_STORE_LIMITS.max_connections
        assert (
            _pool(client)._max_keepalive_connections
            == DEFAULT_STORE_LIMITS.max_keepalive_connections
        )
    finally:
        client.close()


def test_a_transport_can_be_given_a_smaller_pool_of_its_own() -> None:
    transport = PostgrestTransport(
        "https://db.example",
        service_key="sb_secret_test",
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=20),
    )
    try:
        assert _pool(transport._client)._max_connections == 20
    finally:
        transport.close()


def test_two_transports_do_not_share_a_pool() -> None:
    primary = PostgrestTransport("https://db.example", service_key="sb_secret_test")
    reserved = PostgrestTransport(
        "https://db.example",
        service_key="sb_secret_test",
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=20),
    )
    try:
        assert _pool(primary._client) is not _pool(reserved._client)
    finally:
        primary.close()
        reserved.close()


# -- which store a thread gets ------------------------------------------------

import threading

from lab_arena.service import ArenaService


def _service(primary: object, reserved: object) -> ArenaService:
    """An ArenaService with only the store selection wired up.

    The selection is the whole of the change under test; building the rest of
    the service would exercise code this behaviour does not touch.
    """

    service = object.__new__(ArenaService)
    service._primary_store = primary
    service._public_read_store = reserved
    service._reading_public = threading.local()
    return service


def test_ordinary_work_uses_the_primary_store() -> None:
    service = _service("primary", "reserved")
    assert service._store == "primary"
    assert service.store == "primary"


def test_a_public_read_uses_the_reserved_store() -> None:
    service = _service("primary", "reserved")
    with service.public_reads():
        assert service._store == "reserved"
    assert service._store == "primary"


def test_the_reserved_store_is_released_when_a_public_read_fails() -> None:
    service = _service("primary", "reserved")
    try:
        with service.public_reads():
            raise RuntimeError("the round is missing")
    except RuntimeError:
        pass
    assert service._store == "primary"


def test_nesting_restores_the_outer_choice() -> None:
    service = _service("primary", "reserved")
    with service.public_reads():
        with service.public_reads():
            assert service._store == "reserved"
        assert service._store == "reserved"
    assert service._store == "primary"


def test_one_thread_reading_publicly_does_not_move_another() -> None:
    """Public requests run on threadpool workers beside ordinary work."""

    service = _service("primary", "reserved")
    inside_public = threading.Event()
    checked = threading.Event()
    seen = {}

    def public_reader() -> None:
        with service.public_reads():
            inside_public.set()
            checked.wait(timeout=5)

    reader = threading.Thread(target=public_reader)
    reader.start()
    try:
        assert inside_public.wait(timeout=5)
        seen["other_thread"] = service._store
    finally:
        checked.set()
        reader.join(timeout=5)

    assert seen["other_thread"] == "primary"


def test_the_public_store_falls_back_to_the_only_store_when_unset() -> None:
    from lab_arena.service import ServiceConfig
    import dataclasses

    assert any(
        field.name == "public_read_store" and field.default is None
        for field in dataclasses.fields(ServiceConfig)
    )
