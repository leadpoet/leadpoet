"""Exact old worker validator from b56af2539; offline compatibility test only."""
def validate_request(value: Any) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("invalid signal cache request")
    expected = REQUEST_FIELDS | ({"reference", "judgment"} if value.get("operation") == "complete" else set())
    if "verified_source_pages" in value:
        expected.add("verified_source_pages")
        if not isinstance(value["verified_source_pages"], Mapping):
            raise ValueError("invalid retained source context")
    if set(value) != expected or value.get("operation") not in {"claim", "complete"}:
        raise ValueError("invalid signal cache operation")
    if any(type(value.get(key)) is not int or not 0 <= value[key] < 100 for key in ("company_index", "signal_index")):
        raise ValueError("invalid signal cache position")
    if not isinstance(value.get("verified_company_identity"), Mapping) or not isinstance(value.get("verified_funding_stage_context"), list):
        raise ValueError("invalid signal cache context")
    from lab_arena import contracts
    encoded = contracts.canonical_json(value).encode("utf-8")
    if len(encoded) > MAX_BYTES:
        raise ValueError("signal cache request too large")
    return json.loads(encoded)
