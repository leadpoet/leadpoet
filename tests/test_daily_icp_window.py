from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from gateway.research_lab.daily_icp_set import (
    DailyIcpSetUnavailable,
    daily_icp_set_from_input_doc,
    select_daily_icp_set,
)


class DailyIcpWindowTests(unittest.TestCase):
    def test_selects_every_icp_from_exact_active_set(self):
        active_at = datetime(2026, 9, 3, 12, tzinfo=timezone.utc)
        row = {
            "set_id": 20260903,
            "is_active": True,
            "active_from": (active_at - timedelta(hours=12)).isoformat(),
            "active_until": (active_at + timedelta(hours=12)).isoformat(),
            "icps": [
                {"icp_id": f"icp-{index}", "intent_signals": ["hiring"]}
                for index in range(20)
            ],
        }

        window = select_daily_icp_set(
            row,
            required_set_id=20260903,
            active_at=active_at,
        )

        self.assertEqual(
            [item["icp"]["icp_id"] for item in window.benchmark_items],
            [f"icp-{index}" for index in range(20)],
        )
        self.assertEqual(window.set_id, 20260903)
        self.assertEqual(window.public_doc["icp_count"], 20)
        self.assertEqual(
            window.public_doc["icp_refs"],
            [item["icp_ref"] for item in window.benchmark_items],
        )
        restored = daily_icp_set_from_input_doc(
            window.input_doc, required_set_id=20260903
        )
        self.assertEqual(restored.input_doc, window.input_doc)

    def test_rejects_an_inactive_set(self):
        active_at = datetime(2026, 9, 3, 12, tzinfo=timezone.utc)
        row = {
            "set_id": 20260903,
            "is_active": False,
            "icps": [{"icp_id": f"icp-{index}"} for index in range(20)],
        }
        with self.assertRaisesRegex(DailyIcpSetUnavailable, "not_active"):
            select_daily_icp_set(
                row,
                required_set_id=20260903,
                active_at=active_at,
            )

    def test_rejects_partial_daily_set(self):
        active_at = datetime(2026, 9, 3, 12, tzinfo=timezone.utc)
        row = {
            "set_id": 20260903,
            "is_active": True,
            "icps": [{"icp_id": f"icp-{index}"} for index in range(19)],
        }
        with self.assertRaisesRegex(DailyIcpSetUnavailable, "requires_20_icps"):
            select_daily_icp_set(
                row,
                required_set_id=20260903,
                active_at=active_at,
            )


if __name__ == "__main__":
    unittest.main()
