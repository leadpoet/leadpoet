from __future__ import annotations

from io import BytesIO

import pytest

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

    def put_object(self, **request):
        self.puts.append(request)
        key = request["Key"]
        if key in self.objects:
            raise _S3Error(412)
        self.objects[key] = bytes(request["Body"])

    def get_object(self, **request):
        return {"Body": BytesIO(self.objects[request["Key"]])}


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
