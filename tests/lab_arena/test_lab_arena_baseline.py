"""The public baseline enters every daily round through source admission."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lab_arena import source_bundle
from lab_arena.service import ArenaService, DEFAULT_BASELINE_SOURCE_URL, ServiceError


def _submission(submission_id: str, hotkey: str, *, status: str = "accepted") -> dict:
    return {
        "submission_id": submission_id,
        "round_id": "arena-2026-09-05",
        "miner_hotkey": hotkey,
        "status": status,
        "source_ref": "arena/arena-2026-09-05/sources/%s.tar.gz" % submission_id,
        "source_size_bytes": 123,
        "consent": {"public_rerun": True},
        "is_king": False,
    }


class _Objects:
    def __init__(self) -> None:
        self.values = {}

    def put(self, ref, data):
        self.values[ref] = bytes(data)

    def get_bounded(self, ref, max_bytes):
        if ref not in self.values:
            raise KeyError(ref)
        value = self.values[ref]
        if len(value) > max_bytes:
            raise ValueError("too large")
        return value


class _Store:
    def __init__(self, current: dict, submissions: list, published: list = None) -> None:
        self.rounds = {current["round_id"]: current}
        self.submissions = {row["submission_id"]: row for row in submissions}
        for row in published or []:
            self.rounds[row["round_id"]] = row

    def get_round(self, round_id):
        return self.rounds.get(round_id)

    def pending_promotions(
        self, *, pinned_round_id=None, network_name=None, netuid=None, **_kwargs
    ):
        return [row for row in self.rounds.values() if (
            row.get("promotion_required") and not row.get("baseline_promoted_at")
            and row.get("status") == "published"
            and (row.get("publication_doc") or {}).get("king_decision", {}).get("outcome") == "crowned"
            and (pinned_round_id is None or row.get("round_id") == pinned_round_id)
            and (
                network_name is None
                or (
                    (row.get("configuration_doc") or {}).get("network_name", "finney")
                    == network_name
                    and int((row.get("configuration_doc") or {}).get("netuid", 71))
                    == int(netuid)
                )
            )
        )]

    def list_rounds(self, *, status=None, limit=None, **_kwargs):
        rows = list(reversed(list(self.rounds.values())))
        return [row for row in rows if status is None or row.get("status") == status][
            :limit
        ]

    def list_submissions(self, round_id, *, status=None):
        rows = [
            row
            for row in self.submissions.values()
            if row.get("round_id") == round_id
        ]
        return [dict(row) for row in rows if status is None or row["status"] == status]

    def get_submission(self, submission_id):
        row = self.submissions.get(submission_id)
        return dict(row) if row is not None else None

    def update_submission(
        self, _round_id, submission_id, expected, target, patch=None
    ):
        row = self.submissions[submission_id]
        if row["status"] != expected:
            return {"status": "stale"}
        row.update(patch or {})
        row["status"] = target
        return {"status": "ok"}

    def register_submission(self, round_id, submission_id, hotkey, document):
        self.submissions[submission_id] = {
            "round_id": round_id,
            "submission_id": submission_id,
            "miner_hotkey": hotkey,
            "status": "uploading",
            **document,
        }
        return {
            "status": "registered",
            "submission_id": submission_id,
            "source_ref": document["source_ref"],
        }


def _round(round_id: str = "arena-2026-09-05") -> dict:
    return {
        "round_id": round_id,
        "status": "open",
        "configuration_doc": {
            "mode": "live",
            "max_challengers": 1,
            "baseline_hotkey": "baseline",
            "baseline_source_url": DEFAULT_BASELINE_SOURCE_URL,
        },
    }


def _service(store: _Store, objects: _Objects, payload: bytes) -> ArenaService:
    service = object.__new__(ArenaService)
    service._store = store
    service._objects = objects
    service._config = SimpleNamespace(
        mode="live",
        defaults=SimpleNamespace(
            baseline_source_url=DEFAULT_BASELINE_SOURCE_URL
        ),
        baseline_source_fetcher=lambda _url, _limit: payload,
    )
    return service


def _archive(tmp_path) -> bytes:
    source = tmp_path / "baseline"
    source.mkdir()
    (source / "harness.py").write_text(
        "def run_icp(icp):\n    return []\n", encoding="utf-8"
    )
    target = tmp_path / "baseline.tar.gz"
    source_bundle.write_source_archive(source, target)
    return target.read_bytes()


def test_baseline_download_uses_the_same_source_checks_and_freezes(tmp_path):
    current = _round()
    challenger = _submission("sub-c", "challenger")
    overflow = _submission("sub-d", "overflow")
    store = _Store(current, [challenger, overflow])
    objects = _Objects()
    service = _service(store, objects, _archive(tmp_path))

    participants = service.freeze_participants(current["round_id"])

    assert [(row["submission_id"], row["is_king"]) for row in participants] == [
        ("sub-c", False),
        ("baseline-2026-09-05", True),
    ]
    assert store.submissions["baseline-2026-09-05"]["status"] == "frozen"
    assert "source_sha256" not in store.submissions["baseline-2026-09-05"]
    assert "source_cache_key" not in store.submissions["baseline-2026-09-05"]
    assert store.submissions["sub-d"]["status"] == "rejected"
    assert list(objects.values) == [
        "arena/arena-2026-09-05/sources/baseline-2026-09-05.tar.gz"
    ]


def test_yesterdays_miner_identity_never_replaces_the_registered_baseline_identity(tmp_path):
    current = _round()
    baseline = _submission("baseline-2026-09-05", "baseline")
    baseline["is_king"] = True
    previous = {
        "round_id": "arena-2026-09-04",
        "status": "published",
        "configuration_doc": {"mode": "live"},
        "publication_doc": {"king_decision": {"king_submission_id": "old-winner"}},
        "king_hotkey": "miner-winner",
        "king_outcome": "crowned",
    }
    store = _Store(current, [baseline], [previous])
    service = _service(store, _Objects(), _archive(tmp_path))

    participants = service.freeze_participants(current["round_id"])

    assert [(row["miner_hotkey"], row["is_king"]) for row in participants] == [
        ("baseline", True)
    ]
    assert "king-arena-2026-09-05" not in store.submissions


def test_invalid_public_baseline_source_prevents_the_round_from_starting():
    current = _round()
    service = _service(_Store(current, []), _Objects(), b"not a source archive")

    with pytest.raises(ServiceError, match="baseline_source_invalid"):
        service.freeze_participants(current["round_id"])


def test_open_live_round_with_legacy_main_config_downloads_only_promoted_lab(tmp_path):
    current = _round()
    current["configuration_doc"]["baseline_source_url"] = (
        "https://github.com/leadpoet/pydantic-harness/archive/refs/heads/main.tar.gz"
    )
    calls = []
    payload = _archive(tmp_path)
    service = _service(_Store(current, []), _Objects(), payload)
    service._config.baseline_source_fetcher = lambda url, _limit: (
        calls.append(url) or payload
    )

    service.freeze_participants(current["round_id"])

    assert calls == [DEFAULT_BASELINE_SOURCE_URL]


def test_partially_registered_round_recovers_existing_source_object_without_fetch(tmp_path):
    current = _round()
    current["configuration_doc"]["baseline_source_url"] = (
        "https://github.com/leadpoet/pydantic-harness/archive/refs/heads/main.tar.gz"
    )
    objects = _Objects()
    source_ref = "arena/arena-2026-09-05/sources/baseline-2026-09-05.tar.gz"
    payload = _archive(tmp_path)
    objects.put(source_ref, payload)
    service = _service(_Store(current, []), objects, b"unused")
    service._config.baseline_source_fetcher = lambda *_args: pytest.fail("refetched")

    participants = service.freeze_participants(current["round_id"])

    assert [row["submission_id"] for row in participants] == ["baseline-2026-09-05"]


@pytest.mark.parametrize("has_frozen_object", [False, True])
def test_pending_promotion_blocks_a_new_snapshot_but_preserves_recovery(tmp_path, has_frozen_object):
    current = _round()
    previous = {
        "round_id": "arena-2026-09-04",
        "status": "published",
        "promotion_required": True,
        "configuration_doc": {"mode": "live"},
        "publication_doc": {"king_decision": {"outcome": "crowned"}},
    }
    payload = _archive(tmp_path)
    objects = _Objects()
    if has_frozen_object:
        objects.put("arena/arena-2026-09-05/sources/baseline-2026-09-05.tar.gz", payload)
    service = _service(_Store(current, [], [previous]), objects, b"unused")
    service._config.baseline_source_fetcher = lambda *_args: pytest.fail("unexpected download")
    if has_frozen_object:
        # The first stored object is the daily snapshot. A registration retry
        # cannot change that snapshot after another round publishes a winner.
        participants = service.freeze_participants(current["round_id"])
        assert participants[0]["submission_id"] == "baseline-2026-09-05"
        assert next(iter(objects.values.values())) == payload
    else:
        with pytest.raises(ServiceError, match="baseline_promotion_pending"):
            service.freeze_participants(current["round_id"])


@pytest.mark.parametrize(
    ("foreign_network", "foreign_netuid"),
    [("test", 71), ("finney", 401), ("test", 401)],
)
def test_foreign_pending_promotion_does_not_block_current_chain_baseline(
    tmp_path, foreign_network, foreign_netuid
):
    current = _round()
    foreign = {
        "round_id": "arena-foreign-2026-09-04",
        "status": "published",
        "promotion_required": True,
        "configuration_doc": {
            "mode": "live",
            "network_name": foreign_network,
            "netuid": foreign_netuid,
        },
        "publication_doc": {"king_decision": {"outcome": "crowned"}},
    }
    payload = _archive(tmp_path)
    store = _Store(current, [], [foreign])
    service = _service(store, _Objects(), payload)

    participants = service.freeze_participants(current["round_id"])

    assert [row["submission_id"] for row in participants] == [
        "baseline-2026-09-05"
    ]


def test_same_chain_pending_promotion_still_blocks_baseline(tmp_path):
    current = _round()
    same_chain = {
        "round_id": "arena-same-2026-09-04",
        "status": "published",
        "promotion_required": True,
        "configuration_doc": {
            "mode": "live",
            "network_name": "finney",
            "netuid": 71,
        },
        "publication_doc": {"king_decision": {"outcome": "crowned"}},
    }
    store = _Store(current, [], [same_chain])
    service = _service(store, _Objects(), _archive(tmp_path))
    service._config.pinned_round_id = current["round_id"]

    with pytest.raises(ServiceError, match="baseline_promotion_pending"):
        service.freeze_participants(current["round_id"])


def test_existing_frozen_baseline_recovers_without_refetching_old_main():
    current = _round()
    current["configuration_doc"]["baseline_source_url"] = (
        "https://github.com/leadpoet/pydantic-harness/archive/refs/heads/main.tar.gz"
    )
    baseline = _submission("baseline-2026-09-05", "baseline", status="frozen")
    baseline["is_king"] = True
    service = _service(_Store(current, [baseline]), _Objects(), b"unused")
    service._config.baseline_source_fetcher = lambda *_args: pytest.fail("refetched")

    participants = service.freeze_participants(current["round_id"])

    assert [row["submission_id"] for row in participants] == [baseline["submission_id"]]


def test_lab_promotion_changes_only_the_next_round_snapshot(tmp_path):
    first_payload = _archive(tmp_path)
    second_source = tmp_path / "promoted"
    second_source.mkdir()
    (second_source / "harness.py").write_text(
        "def run_icp(icp):\n    return ['promoted']\n", encoding="utf-8"
    )
    second_target = tmp_path / "promoted.tar.gz"
    source_bundle.write_source_archive(second_source, second_target)
    promoted_payload = second_target.read_bytes()
    selected = {DEFAULT_BASELINE_SOURCE_URL: first_payload}

    def freeze(round_id):
        current = _round(round_id)
        objects = _Objects()
        service = _service(_Store(current, []), objects, b"unused")
        service._config.baseline_source_fetcher = lambda url, _limit: selected[url]
        service.freeze_participants(round_id)
        return next(iter(objects.values.values()))

    first_frozen = freeze("arena-2026-09-05")
    selected[DEFAULT_BASELINE_SOURCE_URL] = promoted_payload
    second_frozen = freeze("arena-2026-09-06")

    assert first_frozen == first_payload
    assert second_frozen == promoted_payload
    assert first_frozen != second_frozen


def test_main_change_does_not_change_a_new_live_round_snapshot(tmp_path):
    lab_payload = _archive(tmp_path)
    sources = {
        DEFAULT_BASELINE_SOURCE_URL: lab_payload,
        "https://github.com/leadpoet/pydantic-harness/archive/refs/heads/main.tar.gz": b"main-v1",
    }

    def freeze(round_id):
        current = _round(round_id)
        objects = _Objects()
        service = _service(_Store(current, []), objects, b"unused")
        service._config.baseline_source_fetcher = lambda url, _limit: sources[url]
        service.freeze_participants(round_id)
        return next(iter(objects.values.values()))

    first_frozen = freeze("arena-2026-09-07")
    sources[
        "https://github.com/leadpoet/pydantic-harness/archive/refs/heads/main.tar.gz"
    ] = b"main-v2"
    second_frozen = freeze("arena-2026-09-08")

    assert first_frozen == lab_payload
    assert second_frozen == lab_payload


def test_shadow_source_query_is_not_logged(tmp_path, caplog):
    current = _round()
    current["configuration_doc"]["mode"] = "shadow"
    private_url = "https://example.test/candidate.tar.gz?signature=private"
    current["configuration_doc"]["baseline_source_url"] = private_url
    service = _service(_Store(current, []), _Objects(), _archive(tmp_path))
    service._config.mode = "shadow"

    with caplog.at_level("INFO", logger="lab_arena.service"):
        service.freeze_participants(current["round_id"])

    assert private_url not in caplog.text
    assert "source=configured_shadow_source" in caplog.text
