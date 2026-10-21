-- Exact, one-time recovery of the sole unfinished October 21 judgment.
-- No execution or accepted judgment is changed. Ordinary continuation creates
-- the next score attempt and ordinary publication retains all finality gates.
BEGIN;
SET LOCAL TIME ZONE 'UTC';
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='120s';

DO $audit_table$
DECLARE owner_create BOOLEAN;
BEGIN
  IF to_regclass('public.lab_arena_oct21_judge_recovery535') IS NULL THEN
    owner_create:=has_schema_privilege('lab_arena_owner','public','CREATE');
    IF NOT owner_create THEN GRANT CREATE ON SCHEMA public TO lab_arena_owner; END IF;
    CREATE TABLE public.lab_arena_oct21_judge_recovery535 (
      singleton BOOLEAN PRIMARY KEY CHECK(singleton),
      before_round JSONB NOT NULL,
      after_round JSONB NOT NULL,
      preserved_hashes JSONB NOT NULL,
      trigger_snapshot JSONB NOT NULL,
      recorded_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
    );
    ALTER TABLE public.lab_arena_oct21_judge_recovery535 OWNER TO lab_arena_owner;
    ALTER TABLE public.lab_arena_oct21_judge_recovery535 ENABLE ROW LEVEL SECURITY;
    REVOKE ALL ON public.lab_arena_oct21_judge_recovery535
      FROM PUBLIC,anon,authenticated,service_role,lab_arena_service,leadpoet_parity_reader;
    CREATE TRIGGER lab_arena_oct21_judge_recovery535_append_only
      BEFORE UPDATE OR DELETE ON public.lab_arena_oct21_judge_recovery535
      FOR EACH ROW EXECUTE FUNCTION public.lab_arena_append_only_v1();
    IF NOT owner_create THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
  END IF;
  IF NOT EXISTS(SELECT 1 FROM pg_class c WHERE c.oid='public.lab_arena_oct21_judge_recovery535'::regclass
      AND c.relkind='r' AND pg_get_userbyid(c.relowner)='lab_arena_owner'
      AND c.relrowsecurity AND NOT c.relforcerowsecurity AND c.reloptions IS NULL
      AND NOT EXISTS(SELECT 1 FROM aclexplode(coalesce(c.relacl,acldefault('r',c.relowner))) a
        WHERE a.grantee<>c.relowner))
    OR EXISTS(SELECT 1 FROM pg_policy WHERE polrelid='public.lab_arena_oct21_judge_recovery535'::regclass)
    OR (SELECT count(*) FROM pg_trigger WHERE tgrelid='public.lab_arena_oct21_judge_recovery535'::regclass AND NOT tgisinternal)<>1
    OR NOT EXISTS(SELECT 1 FROM pg_trigger WHERE
      tgrelid='public.lab_arena_oct21_judge_recovery535'::regclass
      AND tgname='lab_arena_oct21_judge_recovery535_append_only'
      AND tgfoid='public.lab_arena_append_only_v1()'::regprocedure
      AND tgenabled='O' AND tgtype=27) THEN
    RAISE EXCEPTION 'Oct21 recovery audit security differs' USING ERRCODE='42501';
  END IF;
END;
$audit_table$;

DO $recover_oct21_judge$
DECLARE
  rid CONSTANT TEXT:='arena-2026-10-21';
  target CONSTANT TEXT:='arena-2026-10-21:sub-8394c3b3fceb6bf6527e4711b8ea4de3:2:9:score';
  -- PostgreSQL JSONB::TEXT SHA-256 values captured before this migration.
  expected_round CONSTANT TEXT:='c38b66e74101dadacb583ba793dd82b83bb40752707448971bbec7996f02e149';
  expected_config CONSTANT TEXT:='4c30f5244dc8152b6f554ecab29aaa69bc543e4e94db83152d4c30e50655edcc';
  expected_participants CONSTANT TEXT:='a3eb9eadae0bf55e54589d416e9fe7210f0b2fab88168a88846be2c546d2df02';
  expected_state CONSTANT JSONB:='{"runs":"744c28cfcd1c8dd6ecca6c312b21000ac8a0dddcd05bffc093ed5e76fc120836","submissions":"8f748c8436e8cf323ff1e9eead87ffc2ac66ec5ce762c7182acfebd79ffb473b","ledger_count":154650,"ledger_max":5565345,"trajectory_count":108613,"trajectory_max":3205873}'::JSONB;
  r public.lab_arena_rounds; after_row public.lab_arena_rounds;
  audit public.lab_arena_oct21_judge_recovery535;
  original_triggers JSONB; current_triggers JSONB;
  before_hashes JSONB; after_hashes JSONB; config JSONB; n BIGINT;
BEGIN
  -- The claim-control lock precedes the round lock, as in all current RPCs.
  PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
  PERFORM 1 FROM public.lab_arena_restart_claim_control WHERE singleton FOR UPDATE;
  LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_submissions,public.lab_arena_runs,
    public.lab_arena_ledger,public.lab_arena_trajectory_events,
    public.lab_arena_judgment_cache,public.lab_arena_company_judgments
    IN SHARE ROW EXCLUSIVE MODE;
  SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=rid FOR UPDATE;
  SELECT * INTO audit FROM public.lab_arena_oct21_judge_recovery535 WHERE singleton;
  IF audit.singleton IS NOT NULL THEN
    IF encode(extensions.digest(audit.before_round::TEXT,'sha256'),'hex')<>expected_round
      OR NOT audit.preserved_hashes @> expected_state
      OR r.configuration_doc IS DISTINCT FROM audit.after_round->'configuration_doc'
      OR r.scoring_generation_floor<>0 THEN
      RAISE EXCEPTION 'Oct21 recovery replay differs' USING ERRCODE='55000';
    END IF;
    RETURN;
  END IF;
  IF expected_round !~ '^[0-9a-f]{64}$' OR expected_config !~ '^[0-9a-f]{64}$'
    OR expected_participants !~ '^[0-9a-f]{64}$' THEN
    RAISE EXCEPTION 'Oct21 recovery evidence is not pinned' USING ERRCODE='55000';
  END IF;
  IF r.status<>'cancelled' OR r.cancel_reason IS DISTINCT FROM 'scoring_incomplete'
    OR r.stage_generation<>9 OR r.status_generation<>11 OR r.scoring_generation_floor<>0
    OR r.evaluation_date IS DISTINCT FROM '2026-10-21'
    OR r.configuration_doc->>'mode' IS DISTINCT FROM 'live'
    OR r.configuration_doc->>'execution_sequence_policy' IS DISTINCT FROM 'baseline_scored_first_v1'
    OR r.configuration_doc#>>'{schedule,final_scoring_close}' IS DISTINCT FROM '2026-10-21T20:30:02Z'
    OR r.configuration_doc#>>'{schedule,publication_deadline}' IS DISTINCT FROM '2026-10-21T20:30:03Z'
    OR r.publication_doc IS NOT NULL OR r.published_at IS NOT NULL
    OR r.reward_activated_at IS NOT NULL OR r.reward_basis_doc IS NOT NULL
    OR r.reward_basis_hash IS NOT NULL OR r.effective_reward_epoch IS NOT NULL
    OR r.king_outcome IS NOT NULL OR r.king_hotkey IS NOT NULL
    OR r.promotion_doc IS NOT NULL OR r.baseline_promoted_at IS NOT NULL
    OR encode(extensions.digest(to_jsonb(r)::TEXT,'sha256'),'hex')<>expected_round
    OR encode(extensions.digest(r.configuration_doc::TEXT,'sha256'),'hex')<>expected_config
    OR encode(extensions.digest(r.participants::TEXT,'sha256'),'hex')<>expected_participants
    OR clock_timestamp()>='2026-10-21T23:59:58Z'::TIMESTAMPTZ THEN
    RAISE EXCEPTION 'Oct21 recovery exact state or window differs' USING ERRCODE='55000';
  END IF;
  IF EXISTS(SELECT 1 FROM public.lab_arena_rounds newer
    WHERE newer.round_id<>rid AND newer.status='published'
      AND newer.arena_network_name=r.arena_network_name AND newer.arena_netuid=r.arena_netuid
      AND newer.evaluation_date>=r.evaluation_date)
    OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE round_id=rid
      AND status IN ('pending','leased','submitted'))
    OR EXISTS(SELECT 1 FROM (SELECT DISTINCT ON(call_identity)
      entry_kind,entry_doc,terminal_response FROM public.lab_arena_ledger
      WHERE round_id=rid AND call_identity IS NOT NULL ORDER BY call_identity,entry_id DESC) head
      WHERE entry_kind IN ('reservation','dispatch')
        OR (entry_kind='uncertain' AND head.entry_doc#>'{call,call_succeeded}' IS DISTINCT FROM 'false'::JSONB)
        OR (entry_kind='settlement' AND jsonb_typeof(head.terminal_response->'call_succeeded') IS DISTINCT FROM 'boolean'))
    THEN RAISE EXCEPTION 'Oct21 recovery is not drained or is superseded' USING ERRCODE='55000'; END IF;
  IF (SELECT count(*) FROM public.lab_arena_runs WHERE round_id=rid AND kind='execute' AND status='accepted')<>168
    OR (SELECT count(*) FROM public.lab_arena_runs WHERE round_id=rid AND kind='execute' AND status='failed')<>69
    OR (SELECT count(*) FROM public.lab_arena_runs WHERE round_id=rid AND kind='score' AND status='accepted')<>167
    OR (SELECT count(*) FROM public.lab_arena_runs WHERE round_id=rid AND kind='score' AND status='failed')<>94
    OR (SELECT count(*) FROM public.lab_arena_runs source WHERE source.round_id=rid
      AND source.kind='execute' AND source.status='accepted' AND NOT EXISTS(
        SELECT 1 FROM public.lab_arena_runs judge WHERE judge.scored_run_id=source.run_id
          AND judge.kind='score' AND judge.status='accepted' AND judge.scoring_generation_floor=0))<>1
    OR NOT EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE run_id=target||':13'
      AND assignment_id=target AND round_id=rid AND stage=2 AND icp_position=9
      AND attempt=13 AND kind='score' AND status='failed' AND terminal_cause='stage_closed'
      AND scored_run_id=replace(target,':score','')||':1' AND output_ref IS NULL
      AND scoring_generation_floor=0)
    OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE assignment_id=target AND attempt>13)
    THEN RAISE EXCEPTION 'Oct21 sole unfinished judgment differs' USING ERRCODE='55000'; END IF;
  -- Preservation of large append-only payload tables follows from the held
  -- table locks, the enabled ledger append-only guard, and reviewed absence of DML
  -- against them. Indexed provider-call receipt count/max and trajectory
  -- count/max detect appends in those scopes; they are not payload hashes.
  -- Rows without a call identity stay protected by the same locks and no-DML rule.
  IF NOT EXISTS(SELECT 1 FROM pg_trigger
    WHERE tgrelid='public.lab_arena_ledger'::regclass
      AND tgname='lab_arena_ledger_append_only' AND tgenabled='O' AND tgtype=27
      AND tgfoid='public.lab_arena_append_only_v1()'::regprocedure) THEN
    RAISE EXCEPTION 'Oct21 recovery ledger append-only guard differs' USING ERRCODE='55000';
  END IF;
  SELECT jsonb_build_object(
    'runs',(SELECT encode(extensions.digest(coalesce(jsonb_agg(to_jsonb(x) ORDER BY run_id),'[]')::TEXT,'sha256'),'hex') FROM public.lab_arena_runs x WHERE round_id=rid),
    'submissions',(SELECT encode(extensions.digest(coalesce(jsonb_agg(to_jsonb(x) ORDER BY submission_id),'[]')::TEXT,'sha256'),'hex') FROM public.lab_arena_submissions x WHERE round_id=rid),
    'ledger_count',(SELECT count(*) FROM public.lab_arena_ledger WHERE round_id=rid AND call_identity IS NOT NULL),
    'ledger_max',(SELECT max(entry_id) FROM public.lab_arena_ledger WHERE round_id=rid AND call_identity IS NOT NULL),
    'trajectory_count',(SELECT count(*) FROM public.lab_arena_trajectory_events WHERE round_id=rid),
    'trajectory_max',(SELECT max(trajectory_id) FROM public.lab_arena_trajectory_events WHERE round_id=rid),
    'judgment_cache',(SELECT encode(extensions.digest(coalesce(jsonb_agg(to_jsonb(x) ORDER BY cache_key),'[]')::TEXT,'sha256'),'hex') FROM public.lab_arena_judgment_cache x WHERE scope_doc->>'round_id'=rid),
    'company_judgments',(SELECT encode(extensions.digest(coalesce(jsonb_agg(to_jsonb(x) ORDER BY cache_key,authority_slot),'[]')::TEXT,'sha256'),'hex') FROM public.lab_arena_company_judgments x WHERE scope_doc->>'round_id'=rid)
  ) INTO before_hashes;
  IF NOT before_hashes @> expected_state THEN
    RAISE EXCEPTION 'Oct21 recovery protected rows differ' USING ERRCODE='55000'; END IF;
  SELECT jsonb_agg(jsonb_build_array(t.tgname,t.tgenabled,pg_get_triggerdef(t.oid),
    pg_get_functiondef(t.tgfoid),to_jsonb(p)-'prosrc') ORDER BY t.tgname)
    INTO original_triggers FROM pg_trigger t JOIN pg_proc p ON p.oid=t.tgfoid
    WHERE t.tgrelid='public.lab_arena_rounds'::regclass AND NOT t.tgisinternal;
  IF EXISTS(SELECT 1 FROM pg_trigger WHERE tgrelid='public.lab_arena_rounds'::regclass
      AND NOT tgisinternal AND tgenabled<>'O')
    OR NOT EXISTS(SELECT 1 FROM pg_trigger WHERE tgrelid='public.lab_arena_rounds'::regclass
      AND tgname='lab_arena_rounds_write_once' AND tgfoid='public.lab_arena_rounds_write_once_v1()'::regprocedure)
    OR (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
      FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
      WHERE p.oid='public.lab_arena_rounds_write_once_v1()'::regprocedure) IS DISTINCT FROM
      '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",false,"v",["search_path=pg_catalog"],"plpgsql"]'::JSONB
    OR encode(extensions.digest(pg_get_functiondef('public.lab_arena_rounds_write_once_v1()'::regprocedure),'sha256'),'hex')
      <>'f6e03c8ad4080e3e713e282719fae154ba5d3d98b9151fab282ac098e44a7c77' THEN
    RAISE EXCEPTION 'Oct21 recovery round guard differs' USING ERRCODE='55000'; END IF;
  config:=jsonb_set(jsonb_set(r.configuration_doc,'{schedule,final_scoring_close}',
    '"2026-10-21T23:59:58Z"'),'{schedule,publication_deadline}','"2026-10-21T23:59:59Z"');
  -- Exactly one guard must permit the audited cancelled-round transition.
  -- All execution, score, ledger, cache and publication guards stay enabled.
  ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER lab_arena_rounds_write_once;
  UPDATE public.lab_arena_rounds SET status='stage2_judged',cancel_reason=NULL,
    status_generation=status_generation+1,stage_generation=stage_generation+1,
    configuration_doc=config,updated_at=clock_timestamp() WHERE round_id=rid;
  GET DIAGNOSTICS n=ROW_COUNT;
  IF n<>1 THEN RAISE EXCEPTION 'Oct21 recovery round update differs'; END IF;
  ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER lab_arena_rounds_write_once;
  SELECT * INTO STRICT after_row FROM public.lab_arena_rounds WHERE round_id=rid;
  IF (to_jsonb(after_row)-ARRAY['status','cancel_reason','status_generation','stage_generation','configuration_doc','updated_at'])
      IS DISTINCT FROM (to_jsonb(r)-ARRAY['status','cancel_reason','status_generation','stage_generation','configuration_doc','updated_at'])
    OR after_row.configuration_doc IS DISTINCT FROM config
    OR after_row.status IS DISTINCT FROM 'stage2_judged'
    OR after_row.status_generation<>r.status_generation+1
    OR after_row.stage_generation<>r.stage_generation+1
    OR after_row.cancel_reason IS NOT NULL THEN
    RAISE EXCEPTION 'Oct21 recovery changed frozen round authority'; END IF;
  SELECT jsonb_agg(jsonb_build_array(t.tgname,t.tgenabled,pg_get_triggerdef(t.oid),
    pg_get_functiondef(t.tgfoid),to_jsonb(p)-'prosrc') ORDER BY t.tgname)
    INTO current_triggers FROM pg_trigger t JOIN pg_proc p ON p.oid=t.tgfoid
    WHERE t.tgrelid='public.lab_arena_rounds'::regclass AND NOT t.tgisinternal;
  IF current_triggers IS DISTINCT FROM original_triggers THEN
    RAISE EXCEPTION 'Oct21 recovery round guard readback differs'; END IF;
  SELECT jsonb_build_object(
    'runs',(SELECT encode(extensions.digest(coalesce(jsonb_agg(to_jsonb(x) ORDER BY run_id),'[]')::TEXT,'sha256'),'hex') FROM public.lab_arena_runs x WHERE round_id=rid),
    'submissions',(SELECT encode(extensions.digest(coalesce(jsonb_agg(to_jsonb(x) ORDER BY submission_id),'[]')::TEXT,'sha256'),'hex') FROM public.lab_arena_submissions x WHERE round_id=rid),
    'ledger_count',(SELECT count(*) FROM public.lab_arena_ledger WHERE round_id=rid AND call_identity IS NOT NULL),
    'ledger_max',(SELECT max(entry_id) FROM public.lab_arena_ledger WHERE round_id=rid AND call_identity IS NOT NULL),
    'trajectory_count',(SELECT count(*) FROM public.lab_arena_trajectory_events WHERE round_id=rid),
    'trajectory_max',(SELECT max(trajectory_id) FROM public.lab_arena_trajectory_events WHERE round_id=rid),
    'judgment_cache',(SELECT encode(extensions.digest(coalesce(jsonb_agg(to_jsonb(x) ORDER BY cache_key),'[]')::TEXT,'sha256'),'hex') FROM public.lab_arena_judgment_cache x WHERE scope_doc->>'round_id'=rid),
    'company_judgments',(SELECT encode(extensions.digest(coalesce(jsonb_agg(to_jsonb(x) ORDER BY cache_key,authority_slot),'[]')::TEXT,'sha256'),'hex') FROM public.lab_arena_company_judgments x WHERE scope_doc->>'round_id'=rid)
  ) INTO after_hashes;
  IF after_hashes IS DISTINCT FROM before_hashes THEN
    RAISE EXCEPTION 'Oct21 recovery changed protected rows'; END IF;
  -- The only DML so far was the exact round UPDATE. The statement-level table
  -- locks exclude all concurrent writes to the protected row collections.
  INSERT INTO public.lab_arena_oct21_judge_recovery535
    (singleton,before_round,after_round,preserved_hashes,trigger_snapshot)
  VALUES(TRUE,to_jsonb(r),to_jsonb(after_row),before_hashes,original_triggers);
END;
$recover_oct21_judge$;
COMMIT;
