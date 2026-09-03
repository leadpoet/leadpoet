#!/usr/bin/env python3
"""Rebuild a published Lab Arena round from the bucket's public prefix alone.

Anyone can check a round from the public prefix ``arena/<round>/public/``:
``publication.json`` carries the signed publication, the signed reward basis,
and the Arena signing key; ``bundle.json`` is the signed result bundle it
names; ``benchmark.json`` carries the committed ICPs; and ``outputs/<hash>.json``
holds every published output. This command fetches those, rebuilds every
aggregate and decision with ``lab_arena.verify.rebuild_round``, and prints the
report. With ``--api`` it also checks that the Arena serves the same
publication.

    python3 scripts/lab_arena_verify.py --round arena-2026-09-05 \\
        --bucket-url https://bucket.example/ [--api https://arena.example/]

Exit status 0 means the round rebuilt exactly; 1 means it did not; 2 means the
material could not be fetched.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable, Dict, Optional

from lab_arena import contracts, verify

ApiGet = Callable[[str], Dict[str, Any]]
ObjectGet = Callable[[str], bytes]


class VerifierFetchError(RuntimeError):
    """Public material could not be fetched or did not hash as published."""


def _public_ref(round_id: str, name: str) -> str:
    return "arena/%s/public/%s" % (round_id, name)


def _json_object(raw: bytes, what: str) -> Dict[str, Any]:
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise VerifierFetchError("%s is not JSON" % what) from exc
    if not isinstance(document, dict):
        raise VerifierFetchError("%s is not an object" % what)
    return document


def verify_published_round(round_id: str, *, object_get: ObjectGet, api_get: Optional[ApiGet] = None) -> Dict[str, Any]:
    """Fetch a published round's public material from the bucket and rebuild it.

    ``object_get`` takes an object reference such as
    ``arena/<id>/public/bundle.json`` and returns its bytes. ``api_get``, when
    given, takes an API path such as ``/arena/v1/rounds/<id>`` and returns the
    decoded JSON; it is used only to check the Arena publishes the same round.
    """

    publication_doc = _json_object(object_get(_public_ref(round_id, "publication.json")), "publication.json")
    publication = publication_doc.get("publication")
    signing_key = publication_doc.get("signing_key")
    if not isinstance(publication, dict) or not isinstance(signing_key, dict) or publication.get("round_id") != round_id:
        raise VerifierFetchError("publication.json does not describe round %s" % round_id)
    if api_get is not None:
        view = api_get("/arena/v1/rounds/%s" % round_id)
        served = view.get("publication") if isinstance(view.get("publication"), dict) else {}
        if view.get("status") != "published" or served.get("publication_hash") != publication.get("publication_hash"):
            raise VerifierFetchError("the Arena does not serve the publication the bucket holds for round %s" % round_id)
    raw_bundle = object_get(publication["result_bundle_ref"])
    bundle = _json_object(raw_bundle, "bundle.json")
    if contracts.hash_bytes(raw_bundle) != publication["result_bundle_hash"] and contracts.document_hash(bundle) != publication["result_bundle_hash"]:
        raise VerifierFetchError("the fetched bundle does not hash to the published result_bundle_hash")
    benchmark = _json_object(object_get(_public_ref(round_id, "benchmark.json")), "benchmark.json")
    outputs: Dict[str, Any] = {}
    wanted = sorted({str(entry["output_hash"]) for entry in bundle.get("outputs") or []})
    for output_hash in wanted:
        document = _json_object(object_get(_public_ref(round_id, "outputs/%s.json" % output_hash.split(":", 1)[-1])), "output %s" % output_hash)
        outputs[output_hash] = document
    verifier_bundle = {
        "round_configuration": bundle["round_configuration"],
        "benchmark_commitment": bundle["benchmark_commitment"],
        "benchmark": benchmark["icps"],
        "participants": bundle["participants"],
        "scorer_policy": bundle["scorer_policy"],
        "scoring_plan": bundle["scoring_plan"],
        "score_bundle": bundle["score_bundle"],
        "outputs": outputs,
        "final_ranking": bundle["final_ranking"],
        "king_decision": bundle["king_decision"],
        "reward_basis": publication_doc.get("reward_basis"),
    }
    report = verify.rebuild_round(verifier_bundle, signing_key)
    report["round_id"] = round_id
    report["result_bundle_hash"] = publication["result_bundle_hash"]
    report["outputs_checked"] = len(wanted)
    report["api_checked"] = api_get is not None
    return report


def _http_fetchers(api_base: Optional[str], bucket_url: str, timeout_seconds: float):
    import httpx

    client = httpx.Client(http1=True, http2=False, follow_redirects=False, timeout=httpx.Timeout(timeout_seconds))
    bucket_url = bucket_url.rstrip("/")

    def object_get(ref: str) -> bytes:
        response = client.get(bucket_url + "/" + ref.lstrip("/"))
        if response.status_code != 200:
            raise VerifierFetchError("GET %s returned HTTP %d" % (ref, response.status_code))
        return response.content

    if not api_base:
        return None, object_get
    base = api_base.rstrip("/")

    def api_get(path: str) -> Dict[str, Any]:
        response = client.get(base + path)
        if response.status_code != 200:
            raise VerifierFetchError("GET %s returned HTTP %d" % (path, response.status_code))
        document = response.json()
        if not isinstance(document, dict):
            raise VerifierFetchError("GET %s returned a non-object" % path)
        return document

    return api_get, object_get


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild a published Lab Arena round from the bucket's public prefix")
    parser.add_argument("--round", required=True, help="round id, e.g. arena-2026-09-05")
    parser.add_argument("--bucket-url", required=True, help="HTTPS base URL of the bucket's public prefix root")
    parser.add_argument("--api", default=None, help="optional Arena API base URL, checked against the bucket's publication")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        api_get, object_get = _http_fetchers(args.api, args.bucket_url, args.timeout)
        report = verify_published_round(args.round, object_get=object_get, api_get=api_get)
    except VerifierFetchError as exc:
        print("could not fetch the round: %s" % exc, file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
