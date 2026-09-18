#!/usr/bin/env python3
"""Render the sealed one-time Sep18 cancelled rerun302 migration locally."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROUND = "arena-2026-09-18"
BASELINE = "baseline-2026-09-18"
BANK_SHA = "6999fdd7bcc09f95943f29127471659d966a86bdb370863f4d895376863acf91"
SOURCE_REF = f"arena/{ROUND}/sources/{BASELINE}-rerun300.tar.gz"
SOURCE_SIZE = 660_992
SOURCE_SHA = "6e44f6a123ac5b89bb1c074d01d83fa05d38fe01178036a06077242cf576f677"
SOURCE_COMMIT = "f8e54592a8ad339c2798935bb909244fd586a8db"
SCORER_DIGEST = "sha256:6af423ff5de5b4acfa83ef4d0a2e6842d142453ceaac2c17d1f40ecca7cdf67f"
SCORER_REFERENCE = (
    "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@" + SCORER_DIGEST
)
SCORING_PREIMAGE_SHA = "2b5adfe57fd9094bb0b8498cb57d3395ace7e1f911ca88c8187f46ee4a332191"
SCORE_RUNS_STRIP_HASH = "sha256:b77e27e2647b6826a4e9aa0489e4e68dd5657920d4b1c4e0479b2ced823250f5"
SCORE_LEDGER_STRIP_HASH = "sha256:e1b67994c341386e8e47a4b6962a604960c5571d914a2cba01014410bc67297b"
SCHEDULE = {
    "submission_open": "2026-09-17T00:00:00Z",
    "submission_cutoff": "2026-09-18T00:00:00Z",
    "benchmark_deadline": "2026-09-18T19:15:00Z",
    "stage_1_start": "2026-09-18T19:15:01Z",
    "stage_1_close": "2026-09-18T22:30:01Z",
    "stage_1_scoring_close": "2026-09-19T03:45:01Z",
    "stage_2_start": "2026-09-19T03:45:02Z",
    "stage_2_close": "2026-09-19T03:45:03Z",
    "final_scoring_close": "2026-09-19T09:00:03Z",
    "publication_deadline": "2026-09-19T09:00:04Z",
}


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def replace_once(value: str, old: str, new: str) -> str:
    if value.count(old) != 1:
        raise ValueError(f"pinned template seam differs: {old[:80]!r}")
    return value.replace(old, new)


def patch_scoring(definition: str, configuration: dict) -> str:
    if hashlib.sha256(definition.encode()).hexdigest() != SCORING_PREIMAGE_SHA:
        raise ValueError("exact post300 scoring preimage required")
    generation = "  v_generation := v_round.stage_generation + 1;"
    validation = f"""{generation}
  IF p_round_id = '{ROUND}'
     AND EXISTS (SELECT 1 FROM public.lab_arena_rounds
                 WHERE round_id = '{ROUND}-rerun300archive') THEN
    IF v_round.configuration_doc IS DISTINCT FROM
         $configuration${canonical(configuration)}$configuration$::JSONB
       OR public.lab_arena_sep18_rerun300_cancelled_archive_valid302_v1() IS NOT TRUE
       OR public.lab_arena_sep18_cancelled_rerun302_active_frozen_valid_v1() IS NOT TRUE THEN
      RAISE EXCEPTION 'lab_arena_rerun302_frozen_state_invalid'
        USING ERRCODE = '22023';
    END IF;
  END IF;"""
    changed = replace_once(definition, generation, validation)
    case_start = "      CASE WHEN p_round_id = 'arena-2026-09-17'"
    changed = replace_once(
        changed,
        case_start,
        f"""      CASE WHEN p_round_id = '{ROUND}'
                  AND EXISTS (SELECT 1 FROM public.lab_arena_rounds
                    WHERE round_id = '{ROUND}-rerun300archive')
           THEN ':rerun302'
           WHEN p_round_id = 'arena-2026-09-17'""",
    )
    old_start = f"    IF p_round_id = '{ROUND}'\n       AND v_scored.submission_id <> '{BASELINE}'"
    if changed.count(old_start) != 5:
        raise ValueError("exact five retained miner branches required")
    first = changed.index(old_start)
    marker = "    v_status := CASE WHEN v_cache.cache_key IS NULL THEN 'pending' ELSE 'accepted' END;"
    end = changed.index(marker, first)
    changed = (
        changed[:first]
        + f"""    IF NOT (p_round_id = '{ROUND}'
+       AND EXISTS (SELECT 1 FROM public.lab_arena_rounds
+                   WHERE round_id = '{ROUND}-rerun300archive')) THEN
+""".replace("+", "")
        + changed[first:end]
        + "    END IF;\n"
        + changed[end:]
    )
    if changed.count(":score:rerun302"):
        raise ValueError("unexpected literal full rerun302 namespace")
    return changed


def render(capture: Path, template: Path, post_sha: str) -> str:
    raw = capture.read_bytes()
    receipt = json.loads(raw)
    payload = receipt["payload"]
    if not receipt.get("read_only") or receipt.get("production_writes") != 0 or receipt.get("provider_calls") != 0:
        raise ValueError("terminal capture is not read-only")
    if not payload.get("transaction_read_only") or not payload.get("repeatable_read"):
        raise ValueError("terminal capture transaction guarantees differ")
    terminal = payload["round"]
    baseline = payload["baseline"]
    if terminal.get("status") != "cancelled" or terminal.get("round_id") != ROUND:
        raise ValueError("exact cancelled terminal300 required")
    if baseline.get("source_ref") != SOURCE_REF or baseline.get("source_size_bytes") != SOURCE_SIZE:
        raise ValueError("terminal source300 identity differs")
    if baseline.get("submission_doc", {}).get("source_sha256") != SOURCE_SHA or baseline.get("submission_doc", {}).get("source_commit") != SOURCE_COMMIT:
        raise ValueError("terminal source300 digest differs")
    if payload.get("active_run_count") != 0 or payload.get("inflight_costs") != 0 or payload.get("success_unresolved_costs") != 0:
        raise ValueError("terminal300 is not quiescent")
    if [payload.get(k) for k in ("archive291_valid", "archive295_valid", "archive297_valid", "archive298_valid", "archive299_valid", "nonbaseline300_valid")] != [True] * 6:
        raise ValueError("terminal archive chain differs")
    configuration = dict(terminal["configuration_doc"])
    configuration["schedule"] = SCHEDULE
    configuration["scorer_image_digest"] = SCORER_DIGEST
    configuration["scorer_image_reference"] = SCORER_REFERENCE
    changed = patch_scoring(payload["scoring_definition"], configuration)
    actual_post = hashlib.sha256(changed.encode()).hexdigest()
    if post_sha != "0" * 64 and post_sha != actual_post:
        raise ValueError("post302 scoring hash differs")
    values = {
        "__SCORING_DEFINITION_SHA256__": SCORING_PREIMAGE_SHA,
        "__POST302_SCORING_DEFINITION_SHA256__": actual_post,
        "__SCORING_DEFINITION_302__": changed + ";",
        "__TERMINAL_ROUND_JSON__": canonical(terminal),
        "__TERMINAL_BASELINE_JSON__": canonical(baseline),
        "__FORWARD_SCHEDULE_JSON__": canonical(SCHEDULE),
        "__NEW_SOURCE_SIZE_BYTES__": str(SOURCE_SIZE),
        "__NEW_SOURCE_SHA256__": SOURCE_SHA,
        "__NEW_SOURCE_COMMIT__": SOURCE_COMMIT,
        "__NEW_SCORER_IMAGE_DIGEST__": SCORER_DIGEST,
        "__NEW_SCORER_IMAGE_REFERENCE__": SCORER_REFERENCE,
        "__TERMINAL_SOURCE_SIZE_BYTES__": str(SOURCE_SIZE),
        "__TERMINAL_SOURCE_SHA256__": SOURCE_SHA,
        "__TERMINAL_SOURCE_COMMIT__": SOURCE_COMMIT,
        "__TERMINAL_BASELINE_RUN_COUNT__": str(payload["BASELINE_RUN_COUNT"]),
        "__TERMINAL_BASELINE_LEDGER_COUNT__": str(payload["BASELINE_LEDGER_COUNT"]),
        "__TERMINAL_BASELINE_RUNS_HASH__": payload["BASELINE_RUN_HASH"],
        "__TERMINAL_BASELINE_LEDGER_HASH__": payload["BASELINE_LEDGER_HASH"],
        "__TERMINAL_NONBASELINE_SUBMISSION_COUNT__": str(payload["NONBASELINE_SUBMISSION_COUNT"]),
        "__TERMINAL_NONBASELINE_SUBMISSIONS_HASH__": payload["NONBASELINE_SUBMISSION_HASH"],
        "__TERMINAL_NONBASELINE_EXECUTE_COUNT__": str(payload["NONBASELINE_EXECUTE_COUNT"]),
        "__TERMINAL_NONBASELINE_EXECUTE_HASH__": payload["NONBASELINE_EXECUTE_HASH"],
        "__TERMINAL_NONBASELINE_EXECUTE_LEDGER_HASH__": payload["NONBASELINE_EXECUTE_LEDGER_HASH"],
        "__TERMINAL_NONBASELINE_EXECUTE_STRIPPED_HASH__": "sha256:dd4f2fd3003fa310a626896eafc962d11556aa554bff5cdc3a61870beb090704",
        "__TERMINAL_NONBASELINE_EXECUTE_LINKED_LEDGER_COUNT__": "2568",
        "__TERMINAL_NONBASELINE_EXECUTE_LINKED_LEDGER_HASH__": "sha256:eb141277f1ad8fc2bab99c8a2b9fa7aa3c627217ed9c22ea773c53b4a1eac0ad",
        "__TERMINAL_NONBASELINE_ORPHAN_LEDGER_HASH__": "sha256:24edb97d881c1446ef87eb3ae4b7e93d660af72ddb40da4459a215a3535b7718",
        "__TERMINAL_NONBASELINE_DERIVED_HASH__": "sha256:5e136618c577ccded99c10f0cb691242cdb3b397959a98b6a06b5a43e188cb72",
        "__TERMINAL_NONBASELINE_SCORE_COUNT__": str(payload["NONBASELINE_SCORE_COUNT"]),
        "__TERMINAL_NONBASELINE_SCORE_LEDGER_COUNT__": str(payload["NONBASELINE_SCORE_LEDGER_COUNT"]),
        "__TERMINAL_NONBASELINE_SCORE_RUNS_STRIP_HASH__": SCORE_RUNS_STRIP_HASH,
        "__TERMINAL_NONBASELINE_SCORE_LEDGER_STRIP_HASH__": SCORE_LEDGER_STRIP_HASH,
        "__ARCHIVE_ROUND_COLUMNS__": ",".join(payload["round_insert_columns"]),
    }
    result = template.read_text()
    for token, value in values.items():
        result = result.replace(token, value)
    atomic = (
        "SELECT public.lab_arena_prepare_sep18_cancelled_rerun302_v1("
        f"{SOURCE_SIZE},'{SOURCE_SHA}','{SOURCE_COMMIT}','{BANK_SHA}',"
        f"'{canonical(SCHEDULE)}'::jsonb);\n"
    )
    result = replace_once(result, "NOTIFY pgrst, 'reload schema';\nCOMMIT;", "NOTIFY pgrst, 'reload schema';\n" + atomic + "COMMIT;")
    leftovers = sorted(set(re.findall(r"__[A-Z0-9_]+__", result)))
    if leftovers:
        raise ValueError(f"unsealed template fields remain: {leftovers}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--post-scoring-sha", default="0" * 64)
    args = parser.parse_args()
    result = render(args.capture, args.template, args.post_scoring_sha)
    args.output.write_text(result)
    print(canonical({
        "output": str(args.output),
        "sha256": hashlib.sha256(result.encode()).hexdigest(),
        "bytes": len(result.encode()),
        "configuration_sha256": hashlib.sha256(canonical((lambda config: (config.update({"schedule": SCHEDULE, "scorer_image_digest": SCORER_DIGEST, "scorer_image_reference": SCORER_REFERENCE}) or config))(dict(json.loads(args.capture.read_text())["payload"]["round"]["configuration_doc"]))).encode()).hexdigest(),
        "schedule": SCHEDULE,
        "post_scoring_sha256": hashlib.sha256(patch_scoring(json.loads(args.capture.read_text())["payload"]["scoring_definition"], (lambda c: (c.update({"schedule": SCHEDULE, "scorer_image_digest": SCORER_DIGEST, "scorer_image_reference": SCORER_REFERENCE}) or c))(dict(json.loads(args.capture.read_text())["payload"]["round"]["configuration_doc"]))).encode()).hexdigest(),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
