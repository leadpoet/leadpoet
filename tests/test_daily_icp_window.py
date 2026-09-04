from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from gateway.research_lab.icp_window import (
    RollingIcpWindowUnavailable,
    select_daily_icp_window_from_set,
)
from research_lab.canonical import sha256_json


class DailyIcpWindowTests(unittest.TestCase):
    def test_selects_every_icp_from_exact_active_set(self):
        active_at = datetime(2026, 9, 3, 12, tzinfo=timezone.utc)
        row = {
            "set_id": 20260903,
            "icp_set_hash": sha256_json({"set": 20260903}),
            "is_active": True,
            "active_from": (active_at - timedelta(hours=12)).isoformat(),
            "active_until": (active_at + timedelta(hours=12)).isoformat(),
            "icps": [
                {"icp_id": "one", "intent_signals": ["hiring"]},
                {"icp_id": "two", "intent_signals": ["funding"]},
            ],
        }

        window = select_daily_icp_window_from_set(
            row,
            required_set_id=20260903,
            active_at=active_at,
        )

        self.assertEqual([item["icp"]["icp_id"] for item in window.benchmark_items], ["one", "two"])
        self.assertEqual(window.set_ids, (20260903,))
        self.assertEqual(window.public_doc["window_mode"], "daily_set")

    def test_rejects_an_inactive_set(self):
        active_at = datetime(2026, 9, 3, 12, tzinfo=timezone.utc)
        row = {
            "set_id": 20260903,
            "icp_set_hash": sha256_json({"set": 20260903}),
            "is_active": False,
            "icps": [{"icp_id": "one"}],
        }
        with self.assertRaisesRegex(RollingIcpWindowUnavailable, "not_active"):
            select_daily_icp_window_from_set(
                row,
                required_set_id=20260903,
                active_at=active_at,
            )


if __name__ == "__main__":
    unittest.main()
