"""Sealed benchmark contracts for private Research Lab evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from research_lab.canonical import sha256_json


@dataclass(frozen=True)
class SealedBenchmarkSet:
    benchmark_id: str
    icp_set_hash: str
    split_ref: str
    item_refs: tuple[str, ...]
    scoring_version: str
    hidden_plaintext_available: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SealedBenchmarkSet":
        item_refs = data.get("item_refs")
        if item_refs is None and data.get("items"):
            item_refs = [
                str(item.get("icp_ref") or item.get("icp_hash") or "")
                for item in data.get("items", [])
            ]
        return cls(
            benchmark_id=str(data["benchmark_id"]),
            icp_set_hash=str(data["icp_set_hash"]),
            split_ref=str(data["split_ref"]),
            item_refs=tuple(str(item) for item in (item_refs or [])),
            scoring_version=str(data["scoring_version"]),
            hidden_plaintext_available=bool(data.get("hidden_plaintext_available", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["item_refs"] = list(self.item_refs)
        return payload


def compute_public_benchmark_hash(items: Sequence[Mapping[str, Any]]) -> str:
    """Hash only public benchmark refs/hashes, never hidden ICP plaintext."""
    public_items = [
        {
            "icp_ref": str(item.get("icp_ref") or ""),
            "icp_hash": str(item.get("icp_hash") or ""),
        }
        for item in items
    ]
    return sha256_json(public_items)


def validate_sealed_benchmark_set(benchmark: SealedBenchmarkSet | Mapping[str, Any]) -> list[str]:
    if not isinstance(benchmark, SealedBenchmarkSet):
        benchmark = SealedBenchmarkSet.from_mapping(benchmark)
    errors: list[str] = []
    if not benchmark.benchmark_id:
        errors.append("benchmark_id_required")
    if not benchmark.icp_set_hash.startswith("sha256:"):
        errors.append("icp_set_hash_must_be_sha256")
    if not benchmark.split_ref:
        errors.append("split_ref_required")
    if not benchmark.item_refs:
        errors.append("benchmark_requires_item_refs")
    if not benchmark.scoring_version:
        errors.append("scoring_version_required")
    return errors
