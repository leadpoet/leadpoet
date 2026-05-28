"""Per-call cost tracking with per-ICP ceilings."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict


class CostCeilingExceeded(Exception):
    pass


@dataclass
class CostTracker:
    """One per qualify() call. Aggregates spend; abstain on hard ceiling."""
    hard_ceiling_usd: float = 2.00
    soft_ceiling_usd: float = 1.50
    by_provider: dict = field(default_factory=lambda: defaultdict(float))
    by_layer: dict = field(default_factory=lambda: defaultdict(float))

    def add(self, provider: str, amount_usd: float, layer: str = "unknown") -> None:
        if amount_usd is None or amount_usd <= 0:
            return
        self.by_provider[provider] += amount_usd
        self.by_layer[layer] += amount_usd
        if self.total > self.hard_ceiling_usd:
            raise CostCeilingExceeded(
                f"Total ${self.total:.4f} exceeds hard ceiling ${self.hard_ceiling_usd:.2f}"
            )

    @property
    def total(self) -> float:
        return sum(self.by_provider.values())

    @property
    def soft_breached(self) -> bool:
        return self.total >= self.soft_ceiling_usd

    def breakdown(self) -> dict:
        return {
            "total_usd": round(self.total, 4),
            "by_provider": {k: round(v, 4) for k, v in self.by_provider.items()},
            "by_layer": {k: round(v, 4) for k, v in self.by_layer.items()},
        }
