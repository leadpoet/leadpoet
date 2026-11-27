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
