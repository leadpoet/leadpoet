import json

from fastapi.testclient import TestClient

from lab_arena.api import create_app
from lab_arena.service import ArenaService

HOTKEY = "5" + "A" * 47


def test_miners_route_scopes_and_validates_cursor():
    service = object.__new__(ArenaService)
    service.public_collection_miners = lambda after: {"miners": [], "next": after}
    with TestClient(create_app(service)) as client:
        response = client.get("/arena/v1/collection/miners")
        assert response.status_code == 200
        assert response.json() == {"miners": [], "next": None}
        assert client.get("/arena/v1/collection/miners", params={"after": HOTKEY}).json()["next"] == HOTKEY
        assert client.get("/arena/v1/collection/miners", params={"after": "../other"}).status_code == 422


def test_confirmed_pending_and_unavailable_are_distinct(tmp_path):
    from lab_arena.collection_contributors import public_page

    def save(path, value):
        path.write_text(json.dumps(value))
        path.chmod(0o600)
    def row(epochs):
        return dict(hotkey=HOTKEY, joined_at="2026-11-26T00:00:00Z", last_funded_at=None,
                    funded_runs=2, epochs=epochs, coldkey="private", funding_key="private")
    watch = tmp_path
    (watch / "settlements").mkdir()
    save(watch / "status.json", {"settlement": {"through": 8}})
    paid = {HOTKEY: {"quoted_microusd": 11000, "actual_rao": 9007199254740993}}
    receipt = {"schema": "leadpoet.collection.revealed_payment.v1", "result": "collection_recipients_observed",
               "quote_epoch": 8, "distribution_epoch": 9, "quotes_microusd": {HOTKEY: 11000},
               "burn_hotkey": "burn", "amounts": paid, "miner_emissions_rao": {HOTKEY: 9007199254740993}}
    save(watch / "settlements/8.json", receipt)
    def summary(epochs):
        return public_page([row(epochs)], watch)["miners"][0]
    assert summary({})["paid_alpha_rao"] == "0"
    confirmed = summary({"8": 11000})
    assert confirmed["payout_status"] == "confirmed"
    assert confirmed["paid_alpha_rao"] == "9007199254740993"
    assert set(confirmed) == {"hotkey","joined_at","last_funded_at","funded_runs","earned_quote_microusd","paid_alpha_rao","payout_status"}
    pending = summary({"8": 11000, "9": 3000})
    assert pending["paid_alpha_rao"] == confirmed["paid_alpha_rao"]
    assert pending["payout_status"] == "pending" and pending["earned_quote_microusd"] == "14000"
    for epochs in ({"7": 500}, {"8": 99}):
        bad = summary(epochs)
        assert bad["payout_status"] == "unavailable" and bad["paid_alpha_rao"] is None
    receipt["result"] = "inspect"
    save(watch / "settlements/6.json", {**receipt, "quote_epoch": 6, "distribution_epoch": 7})
    assert summary({"6": 11000})["paid_alpha_rao"] is None


def test_chart_uses_one_finalized_epoch_and_the_full_miner_pool(tmp_path):
    from hashlib import sha256
    from lab_arena.collection_contributors import public_page

    other = "5" + "B" * 47
    burn = "5" + "C" * 47
    (tmp_path / "settlements").mkdir()
    def write_file(path, obj):
        raw = json.dumps(obj).encode()
        path.write_bytes(raw)
        path.chmod(0o600)
        return sha256(raw).hexdigest()
    def latest():
        # The comparison is independent of the visible roster page.
        return public_page([], tmp_path)["latest_payout"]
    def publish(epoch, amounts, total, *, digest=None, result="collection_recipients_observed"):
        receipt = {
            "schema": "leadpoet.collection.revealed_payment.v1", "result": result,
            "quote_epoch": epoch, "distribution_epoch": epoch + 1, "closed_at": "2026-11-27T03:23:00Z",
            "burn_hotkey": burn, "total_miner_rao": total, "miner_emissions_rao": amounts,
            "quotes_microusd": {HOTKEY: 90, other: 10},
            "amounts": {k: {"quoted_microusd": q, "actual_rao": amounts[k]} for k, q in ((HOTKEY,90),(other,10))},
        }
        saved = write_file(tmp_path / "settlements" / f"{epoch}.json", receipt)
        write_file(tmp_path / "status.json", {"settlement": {"through": epoch + 3, "last_paid": {
            "quote_epoch": epoch, "distribution_epoch": epoch + 1, "result": result, "receipt_sha256": digest or saved}}})
    assert latest() == {"status": "unavailable"}
    write_file(tmp_path / "status.json", {"settlement": {"through": 100}})
    assert latest() == {"status": "none"}
    amounts = {burn: 999950, HOTKEY: 40, other: 10}
    publish(101, amounts, 1000000)
    assert latest() == {
        "status": "confirmed", "epoch": 102, "closed_at": "2026-11-27T03:23:00Z",
        "shares": {HOTKEY: 0.004, other: 0.001},
    }
    publish(102, amounts, 1000000, digest="0" * 64)
    assert latest() == {"status": "unavailable"}
    publish(103, amounts, 50)
    assert latest() == {"status": "unavailable"}
    publish(104, amounts, 1000000, result="inspect")
    assert latest() == {"status": "unavailable"}
