from __future__ import annotations

from io import BytesIO

import pytest

from lab_arena import wiring
from lab_arena.contracts import ArenaContractError
from lab_arena.service import S3ObjectStore


class _S3Error(RuntimeError):
    def __init__(self, status: int) -> None:
        super().__init__(str(status))
        self.response = {
            "Error": {"Code": "PreconditionFailed" if status == 412 else "ConditionalRequestConflict"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }


class _ConditionalS3:
    def __init__(self) -> None:
        self.objects = {}
        self.puts = []
        self.gets = []
        self.heads = []
        self.presigns = []

    def put_object(self, **request):
        self.puts.append(request)
        key = request["Key"]
        if key in self.objects:
            raise _S3Error(412)
        self.objects[key] = bytes(request["Body"])

    def get_object(self, **request):
        self.gets.append(request)
        return {"Body": BytesIO(self.objects[request["Key"]])}

    def head_object(self, **request):
        self.heads.append(request)
        return {"ContentLength": len(self.objects[request["Key"]])}

    def generate_presigned_url(self, operation, **request):
        self.presigns.append((operation, request))
        return "https://uploads.example/object"


def test_s3_object_store_is_idempotent_and_write_once():
    client = _ConditionalS3()
    store = S3ObjectStore("arena", client=client)

    store.put("round/output.json", b"one")
    store.put("round/output.json", b"one")

    assert client.objects["round/output.json"] == b"one"
    assert all(request["IfNoneMatch"] == "*" for request in client.puts)
    with pytest.raises(ArenaContractError, match="different bytes"):
        store.put("round/output.json", b"two")
    assert client.objects["round/output.json"] == b"one"


def test_empty_s3_key_prefix_preserves_logical_keys_for_every_operation():
    client = _ConditionalS3()
    store = S3ObjectStore("arena", client=client)
    logical_ref = "arena/round-1/source.tar.gz"

    store.put(logical_ref, b"source")
    assert store.get(logical_ref) == b"source"
    assert store.get_bounded(logical_ref, 10) == b"source"
    store.presign_put(
        logical_ref,
        size_bytes=6,
        content_type="application/gzip",
        expires_seconds=900,
    )

    assert client.puts[0]["Key"] == logical_ref
    assert all(request["Key"] == logical_ref for request in client.gets)
    assert client.heads == [{"Bucket": "arena", "Key": logical_ref}]
    assert client.presigns[0][1]["Params"]["Key"] == logical_ref
    assert "ContentMD5" not in client.presigns[0][1]["Params"]


def test_s3_key_prefix_is_applied_without_changing_the_logical_ref():
    client = _ConditionalS3()
    store = S3ObjectStore(
        "arena", client=client, prefix="testnet/session-2026-09-04"
    )
    logical_ref = "arena/round-1/source.tar.gz"
    physical_key = "testnet/session-2026-09-04/" + logical_ref

    store.put(logical_ref, b"source")
    assert store.get(logical_ref) == b"source"
    assert store.get_bounded(logical_ref, 10) == b"source"
    upload = store.presign_put(
        logical_ref,
        size_bytes=6,
        content_type="application/gzip",
        expires_seconds=900,
        source_content_md5="ycS5J9O2S8sMKVgsf+qGQw==",
    )

    assert client.objects == {physical_key: b"source"}
    assert client.puts[0]["Key"] == physical_key
    assert all(request["Key"] == physical_key for request in client.gets)
    assert client.heads == [{"Bucket": "arena", "Key": physical_key}]
    assert client.presigns[0][1]["Params"]["Key"] == physical_key
    assert client.presigns[0][1]["Params"]["ContentMD5"] == "ycS5J9O2S8sMKVgsf+qGQw=="
    assert upload["upload_headers"]["content-md5"] == "ycS5J9O2S8sMKVgsf+qGQw=="
    assert upload["upload_url"] == "https://uploads.example/object"


@pytest.mark.parametrize("checksum", ("not-base64", "c2hvcnQ=", "A" * 24))
def test_s3_presign_rejects_a_non_md5_transport_checksum(checksum):
    store = S3ObjectStore("arena", client=_ConditionalS3())
    with pytest.raises(ArenaContractError, match="source_content_md5"):
        store.presign_put(
            "arena/round-1/source.tar.gz",
            size_bytes=6,
            content_type="application/gzip",
            expires_seconds=900,
            source_content_md5=checksum,
        )


@pytest.mark.parametrize(
    "prefix",
    (
        "/testnet",
        "testnet/",
        "testnet//session",
        "testnet/./session",
        "testnet/../production",
        " testnet",
        "testnet ",
        "testnet/ session",
        "testnet/session ",
        "testnet\\session",
        "testnet\x00session",
    ),
)
def test_s3_key_prefix_rejects_ambiguous_or_traversing_segments(prefix):
    with pytest.raises(ArenaContractError, match="object key prefix is invalid"):
        S3ObjectStore("arena", client=_ConditionalS3(), prefix=prefix)


@pytest.mark.parametrize(
    ("configured_prefix", "expected_prefix"),
    ((None, ""), ("testnet/session-2026-09-04", "testnet/session-2026-09-04")),
)
def test_service_wiring_passes_the_optional_object_prefix(
    monkeypatch, configured_prefix, expected_prefix
):
    captured = {}

    monkeypatch.setenv("LAB_ARENA_SUPABASE_URL", "https://database.example")
    monkeypatch.setenv("LAB_ARENA_SUPABASE_ANON_KEY", "anon")
    monkeypatch.setenv("LAB_ARENA_SERVICE_KEY", "service")
    monkeypatch.setenv("LAB_ARENA_BUCKET", "arena")
    monkeypatch.setenv("LAB_ARENA_CHAIN_ENDPOINT", "wss://chain.example")
    if configured_prefix is None:
        monkeypatch.delenv("LAB_ARENA_OBJECT_PREFIX", raising=False)
    else:
        monkeypatch.setenv("LAB_ARENA_OBJECT_PREFIX", configured_prefix)
    monkeypatch.setattr(wiring, "PostgrestTransport", lambda *_args, **_kwargs: object())

    def object_store(bucket, **kwargs):
        captured.update(bucket=bucket, **kwargs)
        return object()

    def stop_after_object_store(_config):
        raise RuntimeError("stop after object store")

    monkeypatch.setattr(wiring, "S3ObjectStore", object_store)
    monkeypatch.setattr(wiring.chain_module, "connect_substrate", stop_after_object_store)

    with pytest.raises(RuntimeError, match="stop after object store"):
        wiring.build_service_from_environment("shadow")
    assert captured["bucket"] == "arena"
    assert captured["prefix"] == expected_prefix
