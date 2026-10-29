"""Pinned Oct10 fixture algorithms; current production guards remain unchanged.

445, 448 and 452 shipped byte-identical plan/commit/publication algorithms.
These tests keep that historical algorithm on actual historical SQL, while
separately proving that the unmodified current publisher holds the same data.
No production service or database function imports or installs this boundary.
"""
import ast
import hashlib
import inspect
import textwrap
from datetime import datetime, timedelta
from types import MethodType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from lab_arena import contracts, integrity, verify
from lab_arena.contracts import ArenaContractError
from lab_arena.service import ArenaService, _PublicationReads, _iso
from tests.lab_arena.historical_service import pin_pre470_baseline_service

OCT10_COMMITS = ('d26dce06d817de42c70fbc1e1b54e352980c266c',
                 '68b422aef1b2c96fa030e35126a18627efe47901',
                 '5889418bfd943f4c1f69cffe95475f57b3d7e9a7')
OCT10_AST_SHA256 = {'build_scoring_plan': '488f4243fe0c4a3d11301cf5fbc4f988d6674ddfc3bff7d7a137685cedf238ac', 'commit_scoring_plan': '5ca3732b4dd6958e9ba0e368f7e524ac66cef471f29f1bb7c61a8ee47fa461b4', 'publish': '28a1848ddb588ad425f6e919ca7ab0609037142d0bdd03ec9e08f7e2add5de19'}

def _oct10_build_scoring_plan(
    *,
    round_id: str,
    stage: int,
    runs: Sequence[Mapping[str, Any]],
    execution_sequence_policy: Optional[str] = None,
    configuration: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Plan from the frozen stage result set.

    ``runs`` are every attempt row of the stage. An assignment with an
    accepted attempt contributes exactly one work item named by its run and
    output reference. A model-caused failure, an exhausted provider error, or
    a provider error before a dispatched final retry reaches the stage deadline
    contributes a zero row. Deadline-isolated infrastructure assignments stay
    explicitly incomplete; they never erase completed work or receive a score.
    """

    if stage not in (1, 2):
        raise ArenaContractError("stage must be 1 or 2")
    positions = contracts.execution_positions(stage, execution_sequence_policy, configuration)
    latest: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    accepted: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for run in runs:
        if int(run["stage"]) != stage:
            continue
        key = (str(run["submission_id"]), int(run["icp_position"]))
        if run.get("status") == "accepted":
            accepted[key] = run
        current = latest.get(key)
        if current is None or int(run.get("attempt") or 0) >= int(current.get("attempt") or 0):
            latest[key] = run
    items: Dict[str, Dict[str, Any]] = {}
    zero_rows: List[Dict[str, Any]] = []
    incomplete_rows: List[Dict[str, Any]] = []
    for key in sorted(latest):
        submission_id, position = key
        if position not in positions:
            raise ArenaContractError("run position %d is outside stage %d" % (position, stage))
        run = accepted.get(key)
        if run is not None:
            scored_run_id = str(run.get("run_id") or "")
            output_ref = str(run.get("output_ref") or "")
            if not scored_run_id or not output_ref:
                raise ArenaContractError("accepted run has no output reference")
            # One work item per miner and ICP: identical outputs from two miners
            # are still judged as distinct competition results.
            items[scored_run_id] = {"scored_run_id": scored_run_id, "icp_position": position, "output_ref": output_ref, "submission_id": submission_id}
            continue
        latest_run = latest[key]
        cause = str(latest_run.get("terminal_cause") or "")
        if (
            latest_run.get("status") == "failed"
            and isinstance(latest_run.get("terminal_doc"), Mapping)
            and latest_run["terminal_doc"].get("infrastructure_incomplete") is True
            and cause in (contracts.INFRASTRUCTURE_TERMINAL_CAUSES | {"stage_closed"})
        ):
            incomplete_rows.append({"submission_id": submission_id, "icp_position": position, "cause": cause})
            continue
        provider_error_exhausted = (
            cause == "provider_error"
            and latest_run.get("status") == "failed"
            and int(latest_run.get("attempt") or 0)
            >= contracts.MAX_ATTEMPTS_PER_ASSIGNMENT
        )
        terminal_doc = latest_run.get("terminal_doc")
        deadline_provider_retry_exhausted = (
            cause == "stage_closed"
            and latest_run.get("status") == "failed"
            and int(latest_run.get("attempt") or 0)
            >= contracts.MAX_ATTEMPTS_PER_ASSIGNMENT
            and isinstance(terminal_doc, Mapping)
            and terminal_doc.get("previous_status") == "leased"
            and terminal_doc.get("deadline_provider_retry_exhausted") is True
        )
        if (
            cause not in contracts.MODEL_CAUSED_TERMINAL_CAUSES
            and not provider_error_exhausted
        ):
            # A retry that the window closed before it ran leaves the
            # earlier model-caused failure standing.
            confirmed = [run for run in runs if (str(run.get("submission_id")), int(run.get("icp_position") or 0)) == key and str(run.get("terminal_cause") or "") in contracts.MODEL_CAUSED_TERMINAL_CAUSES]
            if confirmed:
                cause = str(max(confirmed, key=lambda run: int(run.get("attempt") or 0))["terminal_cause"])
            elif deadline_provider_retry_exhausted:
                confirmed_provider_failures = [
                    run
                    for run in runs
                    if (
                        (
                            str(run.get("submission_id")),
                            int(run.get("icp_position") or 0),
                        )
                        == key
                        and run.get("status") == "failed"
                        and str(run.get("terminal_cause") or "")
                        == "provider_error"
                        and int(run.get("attempt") or 0)
                        < int(latest_run.get("attempt") or 0)
                    )
                ]
                if not confirmed_provider_failures:
                    raise ArenaContractError(
                        "assignment %s/%d has no prior provider failure; the stage must cancel"
                        % (submission_id, position)
                    )
                cause = "provider_error"
            else:
                raise ArenaContractError("assignment %s/%d ended for an infrastructure reason (%s); the stage must cancel" % (submission_id, position, cause or "none"))
        zero_rows.append({"submission_id": submission_id, "icp_position": position, "cause": cause})
    plan = {
        "schema_version": contracts.SCORING_PLAN_SCHEMA_VERSION,
        "round_id": round_id,
        "stage": stage,
        "work_items": [dict(item) for _, item in sorted(items.items())],
        "zero_rows": sorted(zero_rows, key=lambda row: (row["submission_id"], row["icp_position"])),
        **({"incomplete_rows": incomplete_rows} if incomplete_rows else {}),
    }
    if execution_sequence_policy is not None:
        plan["execution_sequence_policy"] = execution_sequence_policy
    return contracts.validate_scoring_plan(plan, configuration)


class _Oct10HistoricalMethods:
    def commit_scoring_plan(self, round_id: str, stage: int) -> Dict[str, Any]:
        round_row = self._round(round_id)
        self.benchmark_icps(round_id)
        plan = _oct10_build_scoring_plan(
            round_id=round_id,
            stage=stage,
            runs=self._store.list_runs(round_id, stage=stage, kind="execute"),
            execution_sequence_policy=(round_row.get("configuration_doc") or {}).get(
                "execution_sequence_policy"
            ),
            configuration=round_row.get("configuration_doc"),
        )
        status = "stage%d_closed" % stage
        result = self._store.transition_round(round_id, status, status, {"stage%d_scoring_plan_doc" % stage: plan})
        return {"status": result.get("status"), "work_items": len(plan["work_items"])}

    def publish(self, round_id: str) -> Dict[str, Any]:
        round_row = self._round(round_id)
        if round_row["status"] != "scored":
            return {"status": "stale", "round_status": round_row["status"]}
        configuration = round_row.get("configuration_doc") or {}
        baseline_first = (
            configuration.get("execution_sequence_policy")
            == contracts.BASELINE_SCORED_FIRST_POLICY
        )
        publication_reads = _PublicationReads(
            self, round_row, self._store.list_runs(round_id, kind="execute")
        )
        stage1_ranking = verify.stage1_ranking(
            self._score_entries_from_runs(
                round_row,
                (
                    range(contracts.benchmark_icp_count(round_row.get("configuration_doc")))
                    if baseline_first
                    else contracts.stage_positions(1, configuration)
                ),
                "stage1_score",
                publication_reads=publication_reads,
            )
        )
        finalists = list(round_row.get("finalists") or [])
        final_entries = self._score_entries_from_runs(
            round_row, range(contracts.benchmark_icp_count(round_row.get("configuration_doc"))), "final_score",
            publication_reads=publication_reads,
        )
        eligibility = {
            str(entry["submission_id"]): ({
                "cost_summary": None,
                "eligible": False,
                "eligibility_reason": "execution_incomplete",
            } if entry.get("execution_incomplete") else self._submission_cost_eligibility(
                round_row,
                str(entry["submission_id"]),
                publication_reads.runs,
                positions=range(contracts.benchmark_icp_count(round_row.get("configuration_doc"))),
                publication_reads=publication_reads,
            ))
            for entry in final_entries
        }
        effective_final_entries = [dict(entry) for entry in final_entries]
        king_entry = next(entry for entry in effective_final_entries if entry["is_king"])
        if (
            king_entry["final_score"] is not None
            and integrity.enabled(round_row.get("configuration_doc") or {})
            and eligibility[str(king_entry["submission_id"])]["eligibility_reason"]
            in ("cost_per_company_exceeded", "execution_cap_exceeded")
        ):
            # Retain the accepted per-ICP scores in the run ledger, but apply
            # the same final cost consequence to the baseline that challengers
            # face. Other fail-closed cost states do not invent a zero score.
            king_entry["final_score"] = 0.0
        decision = verify.king_decision(
            [
                entry
                for entry in effective_final_entries
                if not entry["is_king"]
                and eligibility[str(entry["submission_id"])]["eligible"]
            ],
            king_entry,
            configuration=configuration,
        )
        published_at = _iso(self.now())
        final_ranking = verify.final_ranking(effective_final_entries)
        for row in final_ranking:
            row.update(eligibility[str(row["submission_id"])])
        publication = {
            "schema_version": contracts.PUBLICATION_SCHEMA_VERSION,
            "round_id": round_id,
            "participants": [{"submission_id": p["submission_id"], "miner_hotkey": p["miner_hotkey"], "is_baseline": bool(p.get("is_king"))} for p in round_row.get("participants") or []],
            "stage1_ranking": stage1_ranking,
            "finalists": finalists,
            "final_ranking": final_ranking,
            "king_decision": decision,
            "published_at": published_at,
        }
        contracts.check_strict_document(publication, contracts.PUBLICATION_LIMITS)
        transition = self._store.transition_round(round_id, "scored", "published", {
            "publication_doc": publication,
            "published_at": published_at,
        })
        return {
            "status": transition.get("status"),
            "king_outcome": decision["outcome"],
            "king_hotkey": str(decision.get("king_hotkey") or ""),
        }



def validate_oct10_algorithms():
    """Accept only the source ASTs shared by the three introduction commits."""
    for name, function in (
        ('build_scoring_plan', _oct10_build_scoring_plan),
        ('commit_scoring_plan', _Oct10HistoricalMethods.commit_scoring_plan),
        ('publish', _Oct10HistoricalMethods.publish),
    ):
        tree = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
        tree.name = name
        # This is the only source adaptation: a local name keeps the pinned
        # plan builder separate from the current scoring module.
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == '_oct10_build_scoring_plan':
                node.func = ast.Attribute(value=ast.Name(id='scoring', ctx=ast.Load()),
                                          attr='build_scoring_plan', ctx=ast.Load())
        digest = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
        assert digest == OCT10_AST_SHA256[name], 'Oct10 historical algorithm drift: ' + name


def pin_oct10_service(service, connect, *, round_id):
    """Bind only the exact dynamic-ten-ICP historical SQL/profile."""
    validate_oct10_algorithms()
    assert round_id == 'arena-2026-10-10', 'foreign Oct10 round'
    signatures = {
        'public.lab_arena_record_run_scores(text,smallint,jsonb)': 'd62f624b1da7a1a6bcac69052c3feb562f9141d683abc14ef874a6b67ac280d9',
        'public.lab_arena_close_scoring(text,smallint)': '3bc8179fb87fdeadf200d4f3f53a9550fc2e56ee781584a327b1e6ab00b7bd50',
        'public.lab_arena_transition_round(text,text,text,jsonb)': 'f1826c1ca0789fb00ebef9f3ec6a4bea9de0e02ae4caedd3291d56d901576b90',
        'public.lab_arena_complete_attempt(text,text,jsonb,text,text)': '6e71deaba402ae37b0676ab0c0347d23679c621ac21efd7a3d93835cfcf72043',
    }
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT to_regprocedure('public.lab_arena_recover_baseline_v1(text,jsonb)'), "
                       "to_regprocedure('public.lab_arena_reward_slot_snapshot(text,jsonb)') IS NOT NULL, "
                       "EXISTS(SELECT 1 FROM pg_attribute WHERE attrelid='public.lab_arena_runs'::regclass "
                       "AND attname='scoring_generation_floor' AND NOT attisdropped)")
        assert cursor.fetchone() == (None, True, False), 'Oct10 boundary cannot replace current authority'
        cursor.execute("SELECT p.oid::regprocedure::text,encode(extensions.digest("
                       "pg_get_functiondef(p.oid),'sha256'),'hex') FROM pg_proc p "
                       "WHERE p.oid=ANY(%s::regprocedure[])", (list(signatures),))
        actual = {name.removeprefix('public.'): digest for name, digest in cursor.fetchall()}
        assert actual == {name.removeprefix('public.'): digest for name, digest in signatures.items()}, 'Oct10 SQL protocol drift'
        cursor.execute("SELECT pg_get_userbyid(proowner),prosecdef,proacl::text "
                       "FROM pg_proc WHERE oid=ANY(%s::regprocedure[])", (list(signatures),))
        assert all(row == ('lab_arena_owner', True,
            '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}')
            for row in cursor.fetchall()), 'Oct10 SQL permissions drift'
    row = service._store.get_round(round_id)
    configuration = row['configuration_doc']
    assert str(row['evaluation_date']) == '2026-10-10', 'foreign Oct10 evaluation date'
    assert configuration['mode'] == 'live'
    assert configuration['integrity_policy'] == 'arena_integrity_v1'
    assert configuration['scorer_policy']['scoring_adapter_version'] == 'qualification_integrity_v2'
    assert contracts.benchmark_icp_count(configuration) == 10, 'foreign Oct10 benchmark'
    assert configuration.get('execution_sequence_policy') == contracts.BASELINE_SCORED_FIRST_POLICY
    assert configuration['sourcing_cost_eligibility_policy'] == contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY
    configuration_hash = contracts.document_hash(configuration)
    pin_pre470_baseline_service(service, connect, round_id=round_id)

    for name in ('commit_scoring_plan', 'publish'):
        original = getattr(_Oct10HistoricalMethods, name)
        def bounded(self, *args, _original=original, **kwargs):
            assert args[0] == round_id, 'foreign Oct10 round'
            current = self._store.get_round(round_id)
            assert str(current['evaluation_date']) == '2026-10-10'
            assert contracts.document_hash(current['configuration_doc']) == configuration_hash, 'Oct10 frozen configuration drift'
            return _original(self, *args, **kwargs)
        setattr(service, name, MethodType(bounded, service))
    return service


def assert_current_publisher_holds(service, connect, snapshot):
    """The actual current publisher still rejects this old partial baseline."""
    round_id = 'arena-2026-10-10'
    row = service._store.get_round(round_id)
    assert row['status'] == 'scored'
    deadline = datetime.fromisoformat(row['configuration_doc']['schedule']['publication_deadline'].replace('Z', '+00:00'))
    with connect() as connection, connection.cursor() as cursor:
        before = snapshot(cursor)
    saved_now = service.now
    service.now = lambda: deadline - timedelta(seconds=1)
    try:
        assert ArenaService.publish(service, round_id) == {
            'status': 'retry', 'round_status': 'scored', 'reason': 'baseline_incomplete',
        }
    finally:
        service.now = saved_now
    with connect() as connection, connection.cursor() as cursor:
        assert snapshot(cursor) == before
