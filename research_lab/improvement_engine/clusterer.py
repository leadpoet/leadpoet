"""Cluster Engine trace events by deterministic fingerprint."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .fingerprints import issue_fingerprint
from .models import EngineTraceEvent


def cluster_events(events: Iterable[EngineTraceEvent]) -> dict[str, list[EngineTraceEvent]]:
    clusters: dict[str, list[EngineTraceEvent]] = defaultdict(list)
    for event in events:
        clusters[issue_fingerprint(event)].append(event)
    return dict(clusters)
