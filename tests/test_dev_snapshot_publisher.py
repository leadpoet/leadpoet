"""S3/KMS publication-order tests for immutable development snapshots."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

from research_lab.canonical import sha256_json
from research_lab.eval.dev_eval import compute_dev_set_hash
from research_lab.eval.snapshot_store import (
    MODE_RECORD,
    POINTER_NAME,
    READY_NAME,
    ProviderSnapshotStore,
    build_snapshot_request,
)
from scripts import publish_research_lab_dev_snapshot as publisher


IMAGE = "123456789.dkr.ecr.test/model@sha256:" + "a" * 64


class NoSuchKey(Exception):
    pass


class _Paginator:
    def __init__(self, s3):
        self.s3 = s3

    def paginate(self, *, Bucket, Prefix):
        contents = [
            {"Key": key}
            for (bucket, key), _body in sorted(self.s3.objects.items())
            if bucket == Bucket and key.startswith(Prefix)
        ]
        return [{"Contents": contents}]


class _S3:
    def __init__(self):
        self.objects = {}
        self.puts = []

    def put_object(self, **kwargs):
        body = kwargs["Body"]
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.objects[(kwargs["Bucket"], kwargs["Key"])] = bytes(body)
        self.puts.append(kwargs["Key"])
        return {}

    def get_object(self, *, Bucket, Key):
        try:
            body = self.objects[(Bucket, Key)]
        except KeyError as exc:
            raise NoSuchKey("NoSuchKey") from exc
        return {"Body": io.BytesIO(body)}

    def list_objects_v2(self, *, Bucket, Prefix, MaxKeys=1000):
        keys = [
            key
            for (bucket, key) in self.objects
            if bucket == Bucket and key.startswith(Prefix)
        ][:MaxKeys]
        return {"Contents": [{"Key": key} for key in keys]}

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _Paginator(self)


class _Kms:
    def sign(self, **kwargs):
        return {
            "KeyId": kwargs["KeyId"],
            "SigningAlgorithm": kwargs["SigningAlgorithm"],
            "Signature": b"valid-test-signature",
        }

    def verify(self, **_kwargs):
        return {"SignatureValid": True}


def _snapshot(tmp_path):
    root = tmp_path / "snapshot"
    store = ProviderSnapshotStore(str(root), mode=MODE_RECORD)
    items = []
    for index in range(8):
        icp = {
            "icp_id": f"dev-{index}",
            "industry": "Software Development",
            "sub_industry": f"segment-{index}",
            "country": "United States",
            "employee_count": "51-200",
            "intent_signals": [f"signal-{index}"],
        }
        item = {
            "icp": icp,
            "icp_ref": f"dev:{index}",
            "icp_hash": sha256_json({"icp": icp}),
        }
        items.append(item)
        store.record_response(
            build_snapshot_request("GET", f"https://api.exa.ai/search?id={index}"),
            status=200,
            body_text='{"results":[]}',
        )
    store.write_dev_icp_items(items)
    manifest = store.build_manifest(
        icp_set_hash=compute_dev_set_hash(items),
        dev_set_manifest={"manifest_type": "research_lab_dev_icp_set"},
        recorded_at="2026-07-13T00:00:00Z",
        provenance={
            "champion_image_digest": IMAGE,
            "source_commit": "b" * 40,
            "model_config_hash": "sha256:" + "c" * 64,
            "provider_model_ids": ["provider/model"],
            "replay_output_hashes": [
                {
                    "icp_hash": item["icp_hash"],
                    "output_hash": "sha256:" + "d" * 64,
                }
                for item in items
            ],
        },
    )
    store.write_manifest(manifest)
    store.write_ready_document(store.build_ready_document(manifest))
    assert store.verify_ready_document(require_signature=False)["passed"]
    return root, manifest


def _install_boto(monkeypatch, s3, kms):
    monkeypatch.setitem(
        __import__("sys").modules,
        "boto3",
        SimpleNamespace(client=lambda name: s3 if name == "s3" else kms),
    )


def _run(monkeypatch, source):
    monkeypatch.setattr(
        __import__("sys"),
        "argv",
        [
            "publish",
            "--source-dir",
            str(source),
            "--s3-base-uri",
            "s3://private-bucket/dev",
            "--kms-key-id",
            "alias/dev-snapshot",
        ],
    )
    return publisher.main()


def test_publication_writes_verified_ready_then_pointer_last(monkeypatch, tmp_path):
    source, manifest = _snapshot(tmp_path)
    s3, kms = _S3(), _Kms()
    _install_boto(monkeypatch, s3, kms)

    assert _run(monkeypatch, source) == 0
    target_prefix = f"dev/{manifest['manifest_hash'].split(':', 1)[1]}/"
    ready_key = target_prefix + READY_NAME
    pointer_key = "dev/" + POINTER_NAME
    assert ready_key in s3.puts
    assert pointer_key == s3.puts[-1]
    assert s3.puts.index(ready_key) > max(
        index
        for index, key in enumerate(s3.puts)
        if key.startswith(target_prefix) and key != ready_key
    )

    target_put_count = sum(key.startswith(target_prefix) for key in s3.puts)
    del s3.objects[("private-bucket", pointer_key)]
    before = len(s3.puts)
    assert _run(monkeypatch, source) == 0
    assert s3.puts[before:] == [pointer_key]
    assert sum(key.startswith(target_prefix) for key in s3.puts) == target_put_count


def test_incomplete_existing_immutable_prefix_is_rejected_without_pointer(
    monkeypatch,
    tmp_path,
):
    source, manifest = _snapshot(tmp_path)
    s3, kms = _S3(), _Kms()
    target_prefix = f"dev/{manifest['manifest_hash'].split(':', 1)[1]}/"
    s3.objects[("private-bucket", target_prefix + "partial.json")] = b"{}"
    _install_boto(monkeypatch, s3, kms)

    assert _run(monkeypatch, source) == 1
    assert ("private-bucket", "dev/" + POINTER_NAME) not in s3.objects
    assert not s3.puts
