"""In-memory structured tracing for a single qualify() call."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tracer:
    """Collects per-layer events. One per qualify() call."""
    icp_id: str = ""
    events: list[dict] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def emit(self, layer: str, action: str, **fields: Any) -> None:
        self.events.append(
            {
                "t_offset_ms": int((time.time() - self.started_at) * 1000),
                "layer": layer,
                "action": action,
                **fields,
            }
        )

    def latency_ms(self) -> int:
        return int((time.time() - self.started_at) * 1000)

    def summary(self) -> list[dict]:
        """Compact view per layer."""
        return self.events
