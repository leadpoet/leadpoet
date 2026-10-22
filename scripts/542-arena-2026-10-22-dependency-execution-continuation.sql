-- DRAFT: exact owner-controlled October22 data repair; no production application.
-- No function, trigger, permission, scorer, bank, historical run or ledger changes.
BEGIN;
SET LOCAL TIME ZONE 'UTC';
SET LOCAL lock_timeout='3s';
SET LOCAL statement_timeout='60s';

DO $recovery542$
DECLARE
 rid CONSTANT TEXT:='arena-2026-10-22'; sid CONSTANT TEXT:='sub-5f48ab5a81c56a43b1ce1b66048920a2';
 actor CONSTANT TEXT:='oct22-dependency-execution-recovery542';
 r public.lab_arena_rounds; n public.lab_arena_rounds;
 c public.lab_arena_restart_claim_control; h public.lab_arena_scoring_supersessions;
 revision BIGINT; before_hash TEXT; after_hash TEXT; prior_projection JSONB;
 position INTEGER; participant JSONB;
BEGIN
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 SELECT * INTO STRICT c FROM public.lab_arena_restart_claim_control WHERE singleton FOR UPDATE;
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=rid FOR UPDATE;
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions
  WHERE round_id=rid AND request_id='oct22-dependency-542-phase1-v1';
 IF FOUND THEN
  IF h.actor_ref<>actor OR h.recovery_doc->>'operation'<>'oct22_dependency_execution_replacement_v1'
   OR h.recovery_doc->>'source_archive_sha256'<>'a7c546a447d46f0fc03af17878ca3d9ea6dccc49bbb00547b3b8fe97e0b51f00'
   OR (SELECT count(*)FROM public.lab_arena_runs a WHERE a.round_id=rid AND a.submission_id=sid
      AND a.kind='execute'AND a.attempt=3 AND a.icp_position BETWEEN 0 AND 9)<>10 THEN
   RAISE EXCEPTION 'Oct22 phase1 replay differs'; END IF;
  RETURN; -- Preserve subsequent legitimate progress; never enqueue another attempt.
 END IF;
 IF c.operator_paused OR c.guard_commitment<>'' OR c.owner_commitment<>''
  OR c.restart_phase<>'' OR c.guard_expires_at IS NOT NULL OR c.captured_leases<>'[]'::JSONB
  OR c.guard_generation<>627 THEN RAISE EXCEPTION 'Oct22 control preimage differs'; END IF;
 IF r.status<>'scored'OR r.status_generation<>11 OR r.stage_generation<>8 OR r.scoring_generation_floor<>0
  OR r.benchmark_ref<>'arena/arena-2026-10-22/benchmark.json'
  OR r.evaluation_date::TEXT<>'2026-10-22'OR r.icp_set_date::TEXT<>'2026-10-21'
  OR encode(extensions.digest(r.configuration_doc::TEXT,'sha256'),'hex')<>'3cc3155db480dee9b412414fed4dd97a785025b761ea7d027f562c09a53a7409'
  OR encode(extensions.digest(r.participants::TEXT,'sha256'),'hex')<>'68b71bf87589a2f61fddad3033bae322d3c8ec2e7dc4e2ccbdf88aef3236e52d'
  OR r.publication_doc IS NOT NULL OR r.published_at IS NOT NULL OR r.promotion_doc IS NOT NULL
  OR r.baseline_promoted_at IS NOT NULL OR r.reward_basis_doc IS NOT NULL OR r.reward_basis_hash IS NOT NULL
  OR r.reward_activated_at IS NOT NULL OR r.effective_reward_epoch IS NOT NULL
  OR r.king_outcome IS NOT NULL OR r.king_hotkey IS NOT NULL
  OR clock_timestamp()>='2026-10-22T18:42:02Z'::TIMESTAMPTZ THEN
  RAISE EXCEPTION 'Oct22 round preimage or remaining single-execution budget differs'; END IF;
 -- Fresh external operational preflight verifies20 repaired slots, so ten jobs
 -- can use one execution wave. This is capacity evidence, not an assurance that
 -- model calls succeed or that an unpatched validator may claim replacement work.
 IF EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE status IN('pending','leased','submitted'))
  OR EXISTS(SELECT 1 FROM public.lab_arena_rounds other WHERE other.round_id<>rid
   AND other.configuration_doc->>'mode'='live'AND other.status NOT IN('published','cancelled')
   AND (other.status<>'open'OR (other.configuration_doc#>>'{schedule,submission_cutoff}')::TIMESTAMPTZ<=clock_timestamp()))
  THEN RAISE EXCEPTION 'Oct22 recovery would affect unrelated due or active work'; END IF;
 IF NOT EXISTS(SELECT 1 FROM public.lab_arena_submissions s WHERE s.round_id=rid AND s.submission_id=sid
   AND s.status='frozen'AND NOT s.is_king
   AND s.source_archive_sha256='sha256:a7c546a447d46f0fc03af17878ca3d9ea6dccc49bbb00547b3b8fe97e0b51f00') THEN
  RAISE EXCEPTION 'Oct22 exact submitted source differs'; END IF;
 IF (SELECT encode(extensions.digest(jsonb_agg(to_jsonb(a)ORDER BY run_id)::TEXT,'sha256'),'hex')
   FROM public.lab_arena_runs a WHERE a.round_id=rid AND a.submission_id=sid)<>'2e7d5f6f88800d7af2c37e6d75abd3244f8a6605ceed38a068082266eb8c13d1'
  OR EXISTS(SELECT 1 FROM public.lab_arena_ledger l WHERE l.round_id=rid AND l.submission_id=sid)
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.round_id=rid AND a.submission_id=sid
   AND (a.kind<>'execute'OR a.status<>'failed'OR a.terminal_cause<>'model_error'OR a.attempt NOT IN(1,2)
    OR a.output_ref IS NOT NULL OR a.per_icp_score<>0)) THEN
  RAISE EXCEPTION 'Oct22 twenty historical zero-call setup records differ'; END IF;
 SELECT encode(extensions.digest(coalesce(jsonb_agg(jsonb_build_array(run_id,assignment_id,submission_id,
  stage,icp_position,attempt,kind,status,terminal_cause,output_ref,per_icp_score,stage_generation,
  scoring_generation_floor)ORDER BY run_id),'[]')::TEXT,'sha256'),'hex') INTO before_hash
  FROM public.lab_arena_runs WHERE round_id=rid;
 IF before_hash<>'df3e9c5090ea45e42bcdd1f6b9d919fcf004eff35cbfdaef9b822e2d0ff4f0aa' OR (SELECT count(*)FROM public.lab_arena_runs WHERE round_id=rid)<>665
  THEN RAISE EXCEPTION 'Oct22 preserved cohort differs'; END IF;
 n:=r; n.configuration_doc:=jsonb_set(r.configuration_doc,'{schedule,stage_2_close}',
  '"2026-10-22T19:48:02Z"'::JSONB);
 n.status:='stage2'; n.status_generation:=r.status_generation+1; n.stage_generation:=r.stage_generation+1;
 SELECT greatest(n.stage_generation,coalesce(max(generation_floor),0)+1)INTO revision
  FROM public.lab_arena_scoring_supersessions WHERE round_id=rid;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,
  actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(rid,revision,'oct22-dependency-542-phase1-v1','sha256:'||encode(extensions.digest(
  jsonb_build_object('old',to_jsonb(r)-'updated_at','new',to_jsonb(n)-'updated_at')::TEXT,'sha256'),'hex'),
  actor,txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(n)-'updated_at','{}'::JSONB,
  jsonb_build_object('operation','oct22_dependency_execution_replacement_v1','source_archive_sha256',
   'a7c546a447d46f0fc03af17878ca3d9ea6dccc49bbb00547b3b8fe97e0b51f00',
   'native_sourcecache_receipt_sha256','7e91e98b4694d3c23d94f0cc47444caf244a6bdc6aa869d6004e94b144a8b794',
   'historical_proved_enospc_attempts',9,'historical_unknown_causes',11,
   'reason','Unchanged submitted bundle fails512MiB installer workspace and passes unchanged512MiB final policy under repaired1GiB workspace; no historical cause is rewritten.',
   'preserved_run_metadata_sha256',before_hash,'old_target_rows_sha256','2e7d5f6f88800d7af2c37e6d75abd3244f8a6605ceed38a068082266eb8c13d1',
   'control_before',to_jsonb(c),'worker_access_unchanged',TRUE,'execution_close','2026-10-22T19:48:02Z'));
 UPDATE public.lab_arena_rounds SET configuration_doc=n.configuration_doc,status=n.status,
  status_generation=n.status_generation,stage_generation=n.stage_generation WHERE round_id=rid;
 SELECT p INTO STRICT participant FROM jsonb_array_elements(r.participants)p WHERE p->>'submission_id'=sid;
 FOR position IN 0..9 LOOP
  INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,
   stage,icp_position,attempt,status,stage_generation,kind)
  VALUES(rid||':'||sid||':2:'||position||':3',rid||':'||sid||':2:'||position,rid,sid,
   participant->>'miner_hotkey',2,position,3,'pending',n.stage_generation,'execute');
 END LOOP;
 SELECT encode(extensions.digest(coalesce(jsonb_agg(jsonb_build_array(run_id,assignment_id,submission_id,
  stage,icp_position,attempt,kind,status,terminal_cause,output_ref,per_icp_score,stage_generation,
  scoring_generation_floor)ORDER BY run_id),'[]')::TEXT,'sha256'),'hex') INTO after_hash
  FROM public.lab_arena_runs WHERE round_id=rid AND NOT(submission_id=sid AND kind='execute'AND attempt=3);
 IF after_hash<>before_hash THEN RAISE EXCEPTION 'Oct22 existing cohort changed'; END IF;
END;
$recovery542$;
COMMIT;
