from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.tee.acceptance_corpus_v2 import (
    REQUIRED_PROMOTION_BRANCHES,
    build_acceptance_corpus_v2,
)
from leadpoet_canonical.attested_v2 import sha256_bytes
from leadpoet_canonical.production_parity import sha256_json
from scripts import production_parity_acceptance_transfer as transfer


CANDIDATE_SHA = "a" * 40
RELEASE_HASH = "sha256:" + "b" * 64


def test_direct_cli_bootstraps_candidate_repository_imports(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(transfer.__file__).resolve()),
            "--help",
        ],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def _hash(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _fixture(
    root: Path,
    *,
    kind: str,
    index: int,
    metadata: dict | None = None,
) -> dict:
    relative = Path(kind) / f"{index:04d}.json"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    payload = f'{{"index":{index},"kind":"{kind}"}}\n'.encode("ascii")
    path.write_bytes(payload)
    path.chmod(0o600)
    return {
        "kind": kind,
        "fixture_id": f"{kind}:{index:04d}",
        "captured_at": f"2026-06-{1 + (index % 30):02d}T00:00:00Z",
        "artifact_path": relative.as_posix(),
        "artifact_hash": sha256_bytes(payload),
        "expected_output_hash": _hash(f"output:{kind}:{index}"),
        "receipt_root": _hash(f"receipt:{kind}:{index}"),
        "metadata": dict(metadata or {}),
    }


def _signed_source(tmp_path: Path) -> tuple[Path, str]:
    config = tmp_path / "production-v2"
    corpus = config / "acceptance-corpus-v2"
    corpus.mkdir(parents=True, mode=0o700)
    config.chmod(0o700)
    corpus.chmod(0o700)
    fixtures = [
        _fixture(corpus, kind="autoresearch_run", index=0),
        _fixture(corpus, kind="provider_tape", index=0),
        _fixture(corpus, kind="reward_allocation", index=0),
    ]
    fixtures.extend(
        _fixture(corpus, kind="score_bundle", index=index)
        for index in range(100)
    )
    fixtures.extend(
        _fixture(
            corpus,
            kind="daily_benchmark",
            index=index,
            metadata={"benchmark_date": f"2026-06-{index + 1:02d}"},
        )
        for index in range(14)
    )
    fixtures.extend(
        _fixture(
            corpus,
            kind="promotion_branch",
            index=index,
            metadata={"status": status},
        )
        for index, status in enumerate(sorted(REQUIRED_PROMOTION_BRANCHES))
    )
    fixtures.extend(
        _fixture(
            corpus,
            kind="weight_epoch",
            index=index,
            metadata={"epoch_id": 23_000 + index},
        )
        for index in range(50)
    )
    signing_key = Ed25519PrivateKey.generate()
    public_key = signing_key.public_key().public_bytes_raw()
    manifest = build_acceptance_corpus_v2(
        fixtures=fixtures,
        captured_from="2026-06-01T00:00:00Z",
        captured_through="2026-07-01T00:00:00Z",
        signing_pubkey_hex=public_key.hex(),
        sign_digest=signing_key.sign,
    )
    manifest_path = config / "acceptance-corpus-v2.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    manifest_path.chmod(0o600)
    return config, sha256_bytes(public_key)


def _release_identity(monkeypatch, signer_hash: str) -> None:
    monkeypatch.setattr(
        transfer,
        "_release_identity",
        lambda **_kwargs: (
            {
                "commit_sha": CANDIDATE_SHA,
                "release_hash": RELEASE_HASH,
                "acceptance_signer_pubkey_hash": signer_hash,
            },
            signer_hash,
        ),
    )


def _pack(
    monkeypatch,
    tmp_path: Path,
) -> tuple[bytes, bytes, dict]:
    source, signer_hash = _signed_source(tmp_path)
    _release_identity(monkeypatch, signer_hash)
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    output.chmod(0o700)
    archive = output / transfer.ARCHIVE_NAME
    binding = output / transfer.BINDING_NAME
    evidence = transfer.package_transfer(
        source_config_dir=source,
        candidate_sha=CANDIDATE_SHA,
        archive_path=archive,
        binding_path=binding,
    )
    return archive.read_bytes(), binding.read_bytes(), evidence


def _destination(tmp_path: Path, name: str = "destination") -> Path:
    parent = tmp_path / "destination-root"
    parent.mkdir(mode=0o700, exist_ok=True)
    parent.chmod(0o700)
    return parent / name


def test_signed_acceptance_transfer_round_trips_exactly(
    monkeypatch,
    tmp_path: Path,
):
    archive, binding, packaged = _pack(monkeypatch, tmp_path)
    destination = _destination(tmp_path)
    unpacked = transfer.unpack_transfer(
        archive_payload=archive,
        binding_payload=binding,
        candidate_sha=CANDIDATE_SHA,
        destination_config_dir=destination,
        candidate_release_manifest={},
    )

    assert unpacked == {
        **packaged,
        "release_hash": RELEASE_HASH,
        "copied_exact": True,
    }
    assert (destination / "acceptance-corpus-v2.json").is_file()
    assert unpacked["fixture_count"] == 173
    assert destination.stat().st_mode & 0o777 == 0o700
    assert all(
        path.stat().st_mode & 0o777 == (0o700 if path.is_dir() else 0o600)
        for path in destination.rglob("*")
    )


class _S3:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects
        self.bodies: list[io.BytesIO] = []

    def get_object(self, *, Bucket: str, Key: str):
        assert Bucket == "parity-artifacts"
        body = io.BytesIO(self.objects[Key])
        self.bodies.append(body)
        return {"Body": body, "ContentLength": len(self.objects[Key])}


def test_fetch_closes_every_s3_body_and_unpacks_exactly(
    monkeypatch,
    tmp_path: Path,
):
    archive, binding, _packaged = _pack(monkeypatch, tmp_path)
    prefix = "production-parity/runs/pp-test-1"
    s3 = _S3(
        {
            f"{prefix}/{transfer.ARCHIVE_NAME}": archive,
            f"{prefix}/{transfer.BINDING_NAME}": binding,
        }
    )
    result = transfer.fetch_and_unpack_transfer(
        s3_client=s3,
        artifact_bucket="parity-artifacts",
        run_id="pp-test-1",
        candidate_sha=CANDIDATE_SHA,
        destination_config_dir=_destination(tmp_path),
        candidate_release_manifest={},
    )

    assert result["copied_exact"] is True
    assert len(s3.bodies) == 2
    assert all(body.closed for body in s3.bodies)


def test_bounded_reader_closes_body_when_object_size_is_invalid():
    body = io.BytesIO(b"payload")

    with pytest.raises(
        transfer.AcceptanceTransferError,
        match="object size is invalid",
    ):
        transfer._read_bounded_body(
            {"Body": body, "ContentLength": transfer.MAX_ARCHIVE_BYTES + 1},
            transfer.MAX_ARCHIVE_BYTES,
        )

    assert body.closed


def _rebind(binding_payload: bytes, archive_payload: bytes) -> bytes:
    binding = json.loads(binding_payload)
    binding["archive_sha256"] = transfer._sha256_bytes(archive_payload)
    binding["archive_size_bytes"] = len(archive_payload)
    body = {key: value for key, value in binding.items() if key != "binding_hash"}
    binding["binding_hash"] = sha256_json(body)
    return (
        json.dumps(binding, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def _unsafe_archive(kind: str) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        root = transfer._tar_info("acceptance-corpus-v2", directory=True)
        manifest_payload = b"{}\n"
        manifest = transfer._tar_info(
            "acceptance-corpus-v2.json",
            directory=False,
            size=len(manifest_payload),
        )
        if kind == "reordered":
            archive.addfile(root)
            archive.addfile(manifest, io.BytesIO(manifest_payload))
        else:
            archive.addfile(manifest, io.BytesIO(manifest_payload))
            archive.addfile(root)
        if kind == "traversal":
            payload = b"x"
            member = transfer._tar_info(
                "acceptance-corpus-v2/../escape",
                directory=False,
                size=len(payload),
            )
            archive.addfile(member, io.BytesIO(payload))
        elif kind == "symlink":
            member = tarfile.TarInfo("acceptance-corpus-v2/link")
            member.type = tarfile.SYMTYPE
            member.linkname = "/etc/passwd"
            member.mode = 0o600
            member.uid = 0
            member.gid = 0
            member.mtime = 0
            archive.addfile(member)
    return buffer.getvalue()


@pytest.mark.parametrize("kind", ["reordered", "traversal", "symlink"])
def test_unpack_rejects_noncanonical_or_unsafe_members(
    monkeypatch,
    tmp_path: Path,
    kind: str,
):
    _archive, binding, _packaged = _pack(monkeypatch, tmp_path)
    unsafe = _unsafe_archive(kind)
    destination = _destination(tmp_path)

    with pytest.raises(transfer.AcceptanceTransferError):
        transfer.unpack_transfer(
            archive_payload=unsafe,
            binding_payload=_rebind(binding, unsafe),
            candidate_sha=CANDIDATE_SHA,
            destination_config_dir=destination,
            candidate_release_manifest={},
        )

    assert not destination.exists()


def test_unpack_rejects_conflicting_archive_without_creating_destination(
    monkeypatch,
    tmp_path: Path,
):
    archive, binding, _packaged = _pack(monkeypatch, tmp_path)
    changed = archive[:-1] + bytes([archive[-1] ^ 1])
    destination = _destination(tmp_path)

    with pytest.raises(
        transfer.AcceptanceTransferError,
        match="archive differs",
    ):
        transfer.unpack_transfer(
            archive_payload=changed,
            binding_payload=binding,
            candidate_sha=CANDIDATE_SHA,
            destination_config_dir=destination,
            candidate_release_manifest={},
        )

    assert not destination.exists()
