"""Explicit service boundaries for exact historical SQL migration exercises."""

from datetime import date


def pin_pre484_credit_terminal(service, connect, *, round_id):
    """Deny terminal-credit authority in an explicitly pinned pre-484 SQL test.

    This does not disable baseline recovery or replace an existing RPC. Current
    schemas must use their real terminal-credit predicate.
    """
    from lab_arena import contracts

    connection = connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT to_regprocedure('public.lab_arena__miner_credit_judgment_terminal_v1(text)')")
            assert cursor.fetchone()[0] is None, 'pre484 service cannot replace terminal-credit authority'
    finally:
        connection.close()
    row = service._store.get_round(round_id)
    assert row is not None
    configuration_hash = contracts.document_hash(row['configuration_doc'])

    def historical_terminal_credit_boundary(source_id):
        source = service._store.get_run(source_id)
        assert source is not None and source['round_id'] == round_id
        current = service._store.get_round(round_id)
        assert current is not None and contracts.document_hash(current['configuration_doc']) == configuration_hash
        return False

    service._terminal_miner_credit_judgment = historical_terminal_credit_boundary
    return service


def start_pre470_round(harness, *, round_id, epoch):
    """Admit a historical SQL exercise using a live disposable wall-clock window."""
    from datetime import datetime, timedelta, timezone

    assert date.fromisoformat(round_id[6:16]) < date(2026, 10, 14)
    harness.chain.epoch = epoch
    harness.clock.now = datetime.now(timezone.utc)
    harness.service.create_round(harness.clock.now + timedelta(minutes=30), round_id=round_id)
    harness.round_id = round_id
    for flavor in harness.challengers:
        harness.submit(flavor, round_id)
    harness.clock.advance_to(harness.schedule()['submission_cutoff'])
    assert harness.service.advance_round(round_id)['status'] == 'ok'
    pin_pre470_baseline_service(harness.service, harness.connect, round_id=round_id)
    participants = harness.service.store.get_round(round_id)['participants']
    for participant in participants:
        harness.flavors.setdefault(participant['submission_id'], 'PublicBaseline')
    return len(participants)


def pin_pre470_baseline_service(service, connect, *, round_id):
    """Keep pre-470 migration tests out of the later baseline-recovery branch.

    Migration 470 introduced the recovery RPC and its service hook. A test of
    an earlier frozen migration still uses actual SQL transitions, scoring and
    archive guards; it cannot exercise that later recovery protocol. This
    explicit boundary is forbidden on current schemas and modern rounds.
    """
    connection = connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT to_regprocedure('public.lab_arena_recover_baseline_v1(text,jsonb)')")
            assert cursor.fetchone()[0] is None, 'pre470 service cannot replace current recovery authority'
    finally:
        connection.close()
    row = service._store.get_round(round_id)
    assert row is not None
    # Fresh rehearsals use a current wall-clock schedule on the old protocol.
    # The immutable identifier selects that protocol; bind the actual committed
    # evaluation date separately so it cannot drift after this fixture boundary.
    assert date.fromisoformat(round_id[6:16]) < date(2026, 10, 14)
    frozen_date = str(row.get('evaluation_date') or round_id[6:16])[:10]
    date.fromisoformat(frozen_date)
    assert (row.get('configuration_doc') or {}).get('mode') == 'live'

    def historical_baseline_boundary(current):
        assert current['round_id'] == round_id
        assert str(current.get('evaluation_date') or current['round_id'][6:16])[:10] == frozen_date
        # No baseline-recovery RPC existed in this frozen schema. The ordinary
        # historical transition and all of its database guards still execute.

    pin_pre484_credit_terminal(service, connect, round_id=round_id)
    terminal_credit = service._terminal_miner_credit_judgment

    def historical_terminal_credit_boundary(source_id):
        # Terminal miner-credit authority arrived in 484. These old SQL proofs
        # cannot grant it; False preserves the ordinary NULL/incomplete path.
        historical_baseline_boundary(service._store.get_round(round_id))
        return terminal_credit(source_id)

    service._recover_baseline = historical_baseline_boundary
    service._terminal_miner_credit_judgment = historical_terminal_credit_boundary
    return service


def seed_pre428_initial_reward(service, connect, *, round_id, expected_evaluation_date=None):
    """Seed the genuine V1 reward used by a pre-428 migration proof.

    This is fixture setup, not a claim that current slot-based activation runs
    on old SQL. The canonical document, signature and historical activation RPC
    stay real. The construction follows service.py before 1b5d196af (428).
    """
    from lab_arena import rewards

    connection = connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT to_regprocedure('public.lab_arena_reward_slot_snapshot(text,jsonb)')")
            assert cursor.fetchone()[0] is None, 'pre428 seed cannot replace current reward authority'
    finally:
        connection.close()
    row = service._store.get_round(round_id)
    assert date.fromisoformat(round_id[6:16]) < date(2026, 10, 8)
    if expected_evaluation_date is not None:
        assert str(row['evaluation_date']) == str(expected_evaluation_date)
    else:
        assert date.fromisoformat(str(row['evaluation_date'])[:10]) < date(2026, 10, 8)
    assert row['status'] == 'published'
    config = row['configuration_doc']
    assert config['mode'] == 'live' and config['rewards_enabled'] is True
    assert row.get('reward_activated_at') is None
    decision = row['publication_doc']['king_decision']
    assert decision['outcome'] in ('crowned', 'defended', 'retained_ineligible', 'no_king')
    daily_hotkey = decision.get('king_hotkey') if decision['outcome'] == 'crowned' else ''
    if daily_hotkey:
        assert daily_hotkey != config['baseline_hotkey']
        assert not row.get('promotion_required') or row['baseline_promoted_at'] is not None
    assert not service._superseded_by_published_day(row)
    assert not service._pending_promotion_blocks()
    network, netuid = service._chain_scope()
    prior = service._store.published_reward_bases(mode='live', network_name=network, netuid=netuid, limit=200)
    assert not service._usable_reward_bases(prior), 'initial reward seed cannot replace an existing authority'
    maximum = max((int(r['effective_reward_epoch']) for r in prior if r.get('effective_reward_epoch') is not None), default=-1)
    effective = max(int(service.config.chain.current_settlement_epoch()) + 1, maximum + 1)
    basis = service._sign(rewards.reward_basis_document(
        round_id=round_id, published_at=row['publication_doc']['published_at'],
        finalized_epoch=effective - 1, king_outcome='crowned' if daily_hotkey else 'no_king',
        king_hotkey=daily_hotkey or '', reward_constants=config['reward_constants'],
    ), 'reward_basis_hash')
    return service._store.activate_reward(round_id, basis, service.signing_key_document())


# The following four algorithms are pinned to f5c841857 (the original 336
# publication release). Unlike current policy, that contract scored an exhausted
# provider failure as zero. Only exact historical SQL exercises may use it.
# Accepted artifact verification, score writes, and transitions remain real.
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from types import MethodType
from lab_arena import contracts, integrity, verify
from lab_arena.contracts import ArenaContractError
from lab_arena.service import CANCEL_REASONS, _iso

SEP20_SOURCE_COMMIT = 'f5c841857'
SEP20_SOURCE_SHA256 = {'build_scoring_plan': 'b238655ca7468c60bc135cfbb463f3907d594bac8650cfece988db16b1941a54', 'commit_scoring_plan': '29fa096aba2805187f122de30dfc2040ebac6aea34dded990845b4d74efdf17e', '_score_entries_from_runs': '90dc4f973dc7c9deb313feba52a4180645858612a0fa7b16112ad4f2e0a804fb', 'publish': '5f1a71997413566f3ddbceb55c3958eda571a0beac4e8acc8e834032a59800b1'}

def _sep20_build_scoring_plan(
    *,
    round_id: str,
    stage: int,
    runs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Plan from the frozen stage result set.

    ``runs`` are every attempt row of the stage. An assignment with an
    accepted attempt contributes exactly one work item named by its run and
    output reference. A model-caused failure, or a provider error that remains
    after both normal attempts, contributes a zero row. Other infrastructure
    causes mean the stage should have cancelled and are refused here.
    """

    if stage not in (1, 2):
        raise ArenaContractError("stage must be 1 or 2")
    positions = contracts.stage_positions(stage)
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
        provider_error_exhausted = (
            cause == "provider_error"
            and latest_run.get("status") == "failed"
            and int(latest_run.get("attempt") or 0)
            >= contracts.MAX_ATTEMPTS_PER_ASSIGNMENT
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
            else:
                raise ArenaContractError("assignment %s/%d ended for an infrastructure reason (%s); the stage must cancel" % (submission_id, position, cause or "none"))
        zero_rows.append({"submission_id": submission_id, "icp_position": position, "cause": cause})
    plan = {
        "schema_version": contracts.SCORING_PLAN_SCHEMA_VERSION,
        "round_id": round_id,
        "stage": stage,
        "work_items": [dict(item) for _, item in sorted(items.items())],
        "zero_rows": sorted(zero_rows, key=lambda row: (row["submission_id"], row["icp_position"])),
    }
    return contracts.validate_scoring_plan(plan)

class _Sep20HistoricalMethods:
    def commit_scoring_plan(self, round_id: str, stage: int) -> Dict[str, Any]:
            round_row = self._round(round_id)
            self.benchmark_icps(round_id)
            plan = _sep20_build_scoring_plan(
                round_id=round_id, stage=stage, runs=self._store.list_runs(round_id, stage=stage, kind="execute"),
            )
            status = "stage%d_closed" % stage
            result = self._store.transition_round(round_id, status, status, {"stage%d_scoring_plan_doc" % stage: plan})
            return {"status": result.get("status"), "work_items": len(plan["work_items"])}

    def _score_entries_from_runs(
            self,
            round_row: Mapping[str, Any],
            positions: Sequence[int],
            score_key: str,
        ) -> List[Dict[str, Any]]:
            """Derive one score per participant from write-once run scores."""

            wanted = set(int(position) for position in positions)
            all_runs = self._store.list_runs(str(round_row["round_id"]), kind="execute")
            selected: Dict[Tuple[str, int], Mapping[str, Any]] = {}
            for run in all_runs:
                position = int(run["icp_position"])
                if position not in wanted or run.get("per_icp_score") is None:
                    continue
                key = (str(run["submission_id"]), position)
                current = selected.get(key)
                if current is None or int(run.get("attempt") or 0) > int(current.get("attempt") or 0):
                    selected[key] = run
            entries = []
            for participant in round_row.get("participants") or []:
                submission_id = str(participant["submission_id"])
                rows = [selected.get((submission_id, position)) for position in sorted(wanted)]
                if any(row is None for row in rows):
                    continue
                values = [float(row["per_icp_score"]) for row in rows if row is not None]
                if (
                    round_row.get("configuration_doc", {}).get(
                        "sourcing_cost_eligibility_policy"
                    )
                    == contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY
                ):
                    qualified_by_position = self._qualified_company_counts(
                        round_row, submission_id, all_runs, positions=sorted(wanted)
                    )
                    adjusted = []
                    for position, value in zip(sorted(wanted), values):
                        cost = self._per_icp_cost_eligibility(
                            round_row, submission_id, position, all_runs,
                            qualified=qualified_by_position.get(position, 0),
                        )
                        adjusted.append(value if cost["eligible"] else 0.0)
                    values = adjusted
                entry = {
                    "submission_id": submission_id,
                    score_key: verify.stage_score(values, len(wanted)),
                    "is_king": bool(participant.get("is_king")),
                }
                if score_key == "final_score":
                    entry["hotkey"] = str(participant["miner_hotkey"])
                    if not any(row.get("terminal_cause") == "accepted" for row in rows if row is not None):
                        entry[score_key] = None
                entries.append(entry)
            return entries

    def publish(self, round_id: str) -> Dict[str, Any]:
            round_row = self._round(round_id)
            if round_row["status"] != "scored":
                return {"status": "stale", "round_status": round_row["status"]}
            stage1_ranking = verify.stage1_ranking(
                self._score_entries_from_runs(round_row, contracts.stage_positions(1), "stage1_score")
            )
            finalists = list(round_row.get("finalists") or [])
            final_entries = self._score_entries_from_runs(
                round_row, range(contracts.BENCHMARK_ICP_COUNT), "final_score"
            )
            execution_runs = self._store.list_runs(round_id, kind="execute")
            eligibility = {
                str(entry["submission_id"]): self._submission_cost_eligibility(
                    round_row,
                    str(entry["submission_id"]),
                    execution_runs,
                    positions=range(contracts.BENCHMARK_ICP_COUNT),
                )
                for entry in final_entries
            }
            stored_king_entry = next((e for e in final_entries if e["is_king"]), None)
            if stored_king_entry is None or stored_king_entry["final_score"] is None:
                return self._store.cancel_round(
                    round_id, CANCEL_REASONS["scoring_incomplete"]
                )
            effective_final_entries = [dict(entry) for entry in final_entries]
            king_entry = next(entry for entry in effective_final_entries if entry["is_king"])
            if (
                integrity.enabled(round_row.get("configuration_doc") or {})
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


def pin_sep20_publication_service(service, connect, *, round_id):
    """Bind the original Sep20 algorithms to the exact pre333 SQL protocol.

    This is an old-service migration proof, not a current publication path.
    No future SQL, missing-function shim, or current readiness override is used.
    """
    assert round_id == 'arena-2026-09-20'
    signatures = {
        'lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)': '4c5eb83c3daddfee5bfa2c33eaf33002be07bcd2f7a6fc23b5ff700d5980100a',
        'lab_arena_mark_uncertain(text,text,text,jsonb,integer)': 'da6b73e012983bb59ed83be2976e435af8ea5c0c9fea1b6ceca177cd892b8d99',
        'lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)': 'eb08656f73d4421be399a7bb24599a2b3d63912511372d6738e7d989dd134263',
        'lab_arena_settle_call(text,text,text,bigint,jsonb,integer)': '7f8537fa88aadd0cfe796c1ce326a6fc0b16a30f76d1f51fc800bd5e18052356',
    }
    connection = connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT to_regprocedure('public.lab_arena_recover_baseline_v1(text,jsonb)'), "
                           "to_regprocedure('public.lab_arena_reward_slot_snapshot(text,jsonb)')")
            assert cursor.fetchone() == (None, None), 'Sep20 algorithms cannot replace current authority'
            cursor.execute("SELECT p.oid::regprocedure::text,encode(extensions.digest("
                           "pg_get_functiondef(p.oid),'sha256'),'hex') FROM pg_proc p "
                           "WHERE p.oid=ANY(%s::regprocedure[])", (list(signatures),))
            assert dict(cursor.fetchall()) == signatures, 'Sep20 SQL protocol drift'
    finally:
        connection.close()
    row = service._store.get_round(round_id)
    configuration = row['configuration_doc']
    assert configuration['mode'] == 'live'
    assert contracts.benchmark_icp_count(configuration) == 20
    assert contracts.stage_positions(1, configuration) == tuple(range(10))
    assert contracts.stage_positions(2, configuration) == tuple(range(10, 20))
    assert configuration['sourcing_cost_eligibility_policy'] == contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY
    configuration_hash = contracts.document_hash(configuration)

    def checked(current):
        assert current['round_id'] == round_id
        assert contracts.document_hash(current['configuration_doc']) == configuration_hash

    for name in ('commit_scoring_plan', '_score_entries_from_runs', 'publish'):
        original = getattr(_Sep20HistoricalMethods, name)
        def bounded(self, *args, _original=original, _name=name, **kwargs):
            current = args[0] if _name == '_score_entries_from_runs' else self._store.get_round(args[0])
            checked(current)
            return _original(self, *args, **kwargs)
        setattr(service, name, MethodType(bounded, service))
    return service


def pin_sep20_round_reads(service, connect):
    """Use actual old round columns, without installing future read functions."""
    from lab_arena.store import PUBLIC_CONFIGURATION_COLUMN
    from tests.lab_arena.historical_round_reads import pre517_round_select
    connection = connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT to_regprocedure('public.lab_arena_public_configuration_v1(public.lab_arena_rounds)'), "
                           "to_regprocedure('public.lab_arena_recover_baseline_v1(text,jsonb)'), "
                           "to_regprocedure('public.lab_arena_competition_publication_v1(public.lab_arena_rounds)')")
            assert cursor.fetchone() == (None, None, None), 'Sep20 reads require the actual historical schema'
    finally:
        connection.close()
    transport = service._store._transport
    select = pre517_round_select(transport.select)
    def historical_select(table, *, columns='*', **kwargs):
        if table == 'lab_arena_rounds':
            replacements = {PUBLIC_CONFIGURATION_COLUMN: 'configuration_doc',
                            'publication_doc:lab_arena_competition_publication_v1': 'publication_doc'}
            columns = ','.join(replacements.get(field, field) for field in columns.split(','))
        return select(table, columns=columns, **kwargs)
    transport.select = historical_select
