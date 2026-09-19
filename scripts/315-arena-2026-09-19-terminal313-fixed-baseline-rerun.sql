-- Render only after the Sep19 rerun313 round is terminal and every seal below
-- has been captured from the protected database. This file is not an active
-- migration while it retains the .template suffix.
BEGIN;
SET LOCAL TIME ZONE 'UTC';
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='120s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_sep19_rerun315_archive_valid_v1()
RETURNS BOOLEAN LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public,extensions AS $archive315$
DECLARE archived public.lab_arena_rounds; derived JSONB;
BEGIN
 SELECT * INTO archived FROM public.lab_arena_rounds
  WHERE round_id='arena-2026-09-19-r315archive';
 IF NOT FOUND THEN RETURN FALSE; END IF;
 derived:=archived.configuration_doc->'archived_execution_judgments';
 RETURN archived.status='cancelled'
  AND archived.rewards_enabled IS FALSE
  AND archived.cancel_reason='authorized_sep19_terminal313_fixed_baseline_archive'
  AND archived.configuration_doc->>'archived_terminal_status'='cancelled'
  AND archived.configuration_doc->>'archived_terminal_cancel_reason'=
      'execution_incomplete:stage1:5'
  AND archived.configuration_doc->>'archived_terminal_round_sha256'='19187b93b5d46f9603afe3f4fff281f62ef57258cdbc4729392b5d789c7a84f1'
  AND archived.configuration_doc->>'archived_terminal_baseline_sha256'='a3703c5fb1d8ce5e4884e063cfe3c313439ffe20e674faeee75e340ad1226842'
  AND archived.configuration_doc->>'archived_terminal_miner_submissions_sha256'='781027021d3cadc5f2c0540d9f24a67f65cb25126912ef3544867d5a70153297'
  AND archived.configuration_doc->>'archived_terminal_runs_sha256'='f32cca7cfda7ad689d0bbf34e33f5c071ce77ed03b2bea3bb2e80619626701c1'
  AND archived.configuration_doc->>'archived_terminal_ledger_sha256'='0d75adca0f362d863da3550493b1dde358580cac2cd26879ca8864cc44e72220'
  AND pg_catalog.encode(extensions.digest(pg_catalog.jsonb_build_object(
    'confirmation_bank_ref',archived.confirmation_bank_ref,
    'confirmation_bank_hash',archived.confirmation_bank_hash,
    'confirmation_cohort',archived.confirmation_cohort)::TEXT,'sha256'),'hex')
      ='1af19ff80c88e0fa121d57fa11ea6af14b35220017b0d11a099c70f6b60b314f'
  AND pg_catalog.encode(extensions.digest(pg_catalog.jsonb_build_object(
    'rewards_enabled',archived.configuration_doc->'archived_rewards_enabled',
    'effective_reward_epoch',archived.configuration_doc->'archived_effective_reward_epoch',
    'reward_basis_hash',archived.configuration_doc->'archived_reward_basis_hash',
    'reward_basis_doc',archived.configuration_doc->'archived_reward_basis_doc',
    'signing_key_doc',archived.configuration_doc->'archived_signing_key_doc',
    'reward_activated_at',archived.configuration_doc->'archived_reward_activated_at',
    'king_outcome',archived.configuration_doc->'archived_king_outcome',
    'king_hotkey',archived.configuration_doc->'archived_king_hotkey',
    'king_start_epoch',archived.configuration_doc->'archived_king_start_epoch',
    'promotion_required',archived.configuration_doc->'archived_promotion_required',
    'promotion_doc',archived.configuration_doc->'archived_promotion_doc',
    'baseline_promoted_at',archived.configuration_doc->'archived_baseline_promoted_at')::TEXT,
    'sha256'),'hex')='22a411e2edcd506ab991f1e65db14f92dd83bfbfeedf7c95b4d478b2d880de2b'
  AND pg_catalog.jsonb_typeof(derived)='array'
  AND pg_catalog.jsonb_array_length(derived)=120
  AND (SELECT pg_catalog.count(DISTINCT value->>'run_id')
       FROM pg_catalog.jsonb_array_elements(derived))=120
  AND (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
       WHERE round_id='arena-2026-09-19-r315archive')=5
  AND (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
       WHERE round_id='arena-2026-09-19-r315archive' AND kind='execute'
        AND submission_id='baseline-2026-09-19:r315archive')=40
  AND (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
       WHERE round_id='arena-2026-09-19-r315archive' AND kind='execute'
        AND submission_id='baseline-2026-09-19:r315archive'
        AND status='failed')=40
  AND (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
       WHERE round_id='arena-2026-09-19-r315archive' AND kind='execute'
        AND submission_id='baseline-2026-09-19:r315archive'
        AND terminal_cause='model_error')=25
  AND (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
       WHERE round_id='arena-2026-09-19-r315archive' AND kind='execute'
        AND submission_id='baseline-2026-09-19:r315archive'
        AND terminal_cause='provider_error')=15
  AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs
       WHERE round_id='arena-2026-09-19-r315archive' AND kind='execute'
        AND submission_id='baseline-2026-09-19:r315archive'
        AND assignment_id NOT LIKE '%:rerun313')
  AND (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
       WHERE round_id='arena-2026-09-19-r315archive' AND kind='score')=0
  AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs
       WHERE round_id='arena-2026-09-19-r315archive' AND kind='score'
        AND submission_id NOT LIKE '%:r315archive')
  AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs
       WHERE round_id='arena-2026-09-19-r315archive' AND kind='score'
        AND assignment_id NOT LIKE '%:score:rerun313')
  AND NOT EXISTS(SELECT 1 FROM public.lab_arena_ledger
       WHERE round_id='arena-2026-09-19-r315archive'
        AND submission_id='baseline-2026-09-19:r315archive')
  AND NOT EXISTS(SELECT 1 FROM public.lab_arena_ledger ledger
       WHERE ledger.round_id='arena-2026-09-19-r315archive'
        AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs run
          WHERE run.round_id='arena-2026-09-19-r315archive'
           AND run.run_id=ledger.run_id));
END $archive315$;
ALTER FUNCTION public.lab_arena_sep19_rerun315_archive_valid_v1()
 OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep19_rerun315_archive_valid_v1()
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_sep19_rerun315_active_valid_v1()
RETURNS BOOLEAN LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $active315$
DECLARE active public.lab_arena_rounds; archived public.lab_arena_rounds;
BEGIN
 SELECT * INTO active FROM public.lab_arena_rounds
  WHERE round_id='arena-2026-09-19';
 SELECT * INTO archived FROM public.lab_arena_rounds
  WHERE round_id='arena-2026-09-19-r315archive';
 RETURN active.round_id IS NOT NULL AND archived.round_id IS NOT NULL
  AND public.lab_arena_sep19_rerun315_archive_valid_v1() IS TRUE
  AND pg_catalog.encode(extensions.digest(pg_catalog.jsonb_build_object(
    'confirmation_bank_ref',active.confirmation_bank_ref,
    'confirmation_bank_hash',active.confirmation_bank_hash,
    'confirmation_cohort',active.confirmation_cohort)::TEXT,'sha256'),'hex')
      ='1af19ff80c88e0fa121d57fa11ea6af14b35220017b0d11a099c70f6b60b314f'
  AND pg_catalog.encode(extensions.digest(pg_catalog.jsonb_build_object(
    'rewards_enabled',active.rewards_enabled,
    'effective_reward_epoch',active.effective_reward_epoch,
    'reward_basis_hash',active.reward_basis_hash,
    'reward_basis_doc',active.reward_basis_doc,
    'signing_key_doc',active.signing_key_doc,
    'reward_activated_at',active.reward_activated_at,
    'king_outcome',active.king_outcome,
    'king_hotkey',active.king_hotkey,
    'king_start_epoch',active.king_start_epoch,
    'promotion_required',active.promotion_required,
    'promotion_doc',active.promotion_doc,
    'baseline_promoted_at',active.baseline_promoted_at)::TEXT,'sha256'),'hex')
      ='22a411e2edcd506ab991f1e65db14f92dd83bfbfeedf7c95b4d478b2d880de2b'
  AND EXISTS(SELECT 1 FROM public.lab_arena_rounds round
    WHERE round.round_id='arena-2026-09-19'
     AND round.configuration_doc->>'scorer_image_digest'='sha256:2df1280d2b5110cad323a8cdf32b5be96b5a7a1e1ac9edfc57dd48fc385b968e'
     AND round.configuration_doc->>'scorer_image_reference'='493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:2df1280d2b5110cad323a8cdf32b5be96b5a7a1e1ac9edfc57dd48fc385b968e')
  AND EXISTS(SELECT 1 FROM public.lab_arena_submissions baseline
    WHERE baseline.round_id='arena-2026-09-19'
     AND baseline.submission_id='baseline-2026-09-19'
     AND baseline.source_ref='arena/arena-2026-09-19/sources/baseline-2026-09-19-rerun313-415f52f3.tar.gz'
     AND baseline.source_size_bytes=788550
     AND baseline.submission_doc->>'source_sha256'='b7f73cf49b73a055c73b470acff224333ab5ca3e7247c3502f9873cd3f3e4a56'
     AND baseline.submission_doc->>'source_commit'='415f52f39aa0d0cef1cc4e57ea52d46d2c4df69d')
  AND (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
    WHERE round_id='arena-2026-09-19' AND kind='execute'
     AND submission_id<>'baseline-2026-09-19' AND status='accepted'
     AND terminal_cause='accepted' AND output_ref IS NOT NULL)=80
  AND (SELECT pg_catalog.count(DISTINCT assignment_id) FROM public.lab_arena_runs
    WHERE round_id='arena-2026-09-19' AND kind='execute'
     AND submission_id='baseline-2026-09-19'
     AND assignment_id LIKE '%:rerun315')=20
  AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs
    WHERE round_id='arena-2026-09-19' AND kind='execute'
     AND submission_id='baseline-2026-09-19'
     AND assignment_id NOT LIKE '%:rerun315')
  AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs
    WHERE round_id='arena-2026-09-19' AND kind='score'
     AND assignment_id NOT LIKE '%:score:rerun315');
END $active315$;
ALTER FUNCTION public.lab_arena_sep19_rerun315_active_valid_v1()
 OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep19_rerun315_active_valid_v1()
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

-- Replace the exact rerun313 guard and namespace in the current scorer. A
-- second stacked guard would reject rerun315 after the archive is created.
DO $patch_sep19_rerun315_scorer$
DECLARE definition TEXT; current_hash TEXT;
 old_assignment TEXT:=$assignment313$    IF p_round_id='arena-2026-09-19'
       AND EXISTS(SELECT 1 FROM public.lab_arena_rounds
         WHERE round_id='arena-2026-09-19-r313archive') THEN
      v_assignment:=p_round_id||':'||v_scored.submission_id||':'||p_stage::TEXT||':'||
        v_scored.icp_position::TEXT||':score:rerun313';
    END IF;
    v_status := CASE WHEN v_cache.cache_key IS NULL THEN 'pending' ELSE 'accepted' END;$assignment313$;
 old_guard TEXT:=$guard313$  IF p_round_id='arena-2026-09-19'
     AND EXISTS(SELECT 1 FROM public.lab_arena_rounds
       WHERE round_id='arena-2026-09-19-r313archive')
     AND public.lab_arena_sep19_rerun313_active_valid_v1() IS NOT TRUE THEN
    RAISE EXCEPTION 'lab_arena_sep19_rerun313_frozen_state_invalid'
      USING ERRCODE='22023';
  END IF;
  FOR v_item IN$guard313$;
BEGIN
 definition:=pg_catalog.pg_get_functiondef(
  'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::pg_catalog.regprocedure);
 current_hash:=pg_catalog.encode(extensions.digest(definition,'sha256'),'hex');
 IF current_hash='5fb72273dbd4aef5417bf89bbabb57cd55df7213feb5abd7ebbe4fc3482c573a' THEN
  IF (pg_catalog.length(definition)-pg_catalog.length(pg_catalog.replace(
      definition,old_assignment,'')))<>pg_catalog.length(old_assignment)
    OR (pg_catalog.length(definition)-pg_catalog.length(pg_catalog.replace(
      definition,old_guard,'')))<>pg_catalog.length(old_guard) THEN
   RAISE EXCEPTION 'Sep19 rerun315 scorer seam differs';
  END IF;
  definition:=pg_catalog.replace(definition,old_guard,$new_guard$  IF p_round_id='arena-2026-09-19'
     AND EXISTS(SELECT 1 FROM public.lab_arena_rounds
       WHERE round_id='arena-2026-09-19-r315archive')
     AND public.lab_arena_sep19_rerun315_active_valid_v1() IS NOT TRUE THEN
    RAISE EXCEPTION 'lab_arena_sep19_rerun315_frozen_state_invalid'
      USING ERRCODE='22023';
  END IF;
  FOR v_item IN$new_guard$);
  definition:=pg_catalog.replace(definition,old_assignment,$new_assignment$    IF p_round_id='arena-2026-09-19'
       AND EXISTS(SELECT 1 FROM public.lab_arena_rounds
         WHERE round_id='arena-2026-09-19-r315archive') THEN
      v_assignment:=p_round_id||':'||v_scored.submission_id||':'||p_stage::TEXT||':'||
        v_scored.icp_position::TEXT||':score:rerun315';
    END IF;
    v_status := CASE WHEN v_cache.cache_key IS NULL THEN 'pending' ELSE 'accepted' END;$new_assignment$);
  EXECUTE definition;
 ELSIF current_hash<>'0cf75a5ac52a23b51d761d03471deaa77e770175525b79befb6b75aa3d2ecb13' THEN
  RAISE EXCEPTION 'Sep19 rerun315 scorer definition differs';
 END IF;
 IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
   'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::pg_catalog.regprocedure),
   'sha256'),'hex')<>'0cf75a5ac52a23b51d761d03471deaa77e770175525b79befb6b75aa3d2ecb13' THEN
  RAISE EXCEPTION 'Sep19 rerun315 scorer patch differs';
 END IF;
END $patch_sep19_rerun315_scorer$;

CREATE OR REPLACE FUNCTION public.lab_arena_prepare_sep19_rerun315_v1(
 p_source_size_bytes BIGINT,p_source_sha256 TEXT,p_source_commit TEXT,
 p_schedule JSONB,p_scorer_digest TEXT,p_scorer_reference TEXT)
RETURNS JSONB LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,public,extensions AS $prepare315$
DECLARE
 active_round public.lab_arena_rounds; active_baseline public.lab_arena_submissions;
 expected_submission JSONB; archive_config JSONB; archive_participants JSONB;
 new_participants JSONB; derived JSONB; other_before JSONB; other_after JSONB;
 miner_runs_before JSONB; miner_runs_after JSONB;
 miner_ledger_before JSONB; miner_ledger_after JSONB;
 active_count BIGINT; unsettled BIGINT; moved_scores BIGINT;
 moved_score_ledger BIGINT; moved_baseline_runs BIGINT;
 moved_baseline_ledger BIGINT; inserted_runs BIGINT; position INTEGER;
 stage SMALLINT; assignment TEXT;
BEGIN
 PERFORM pg_catalog.set_config('lock_timeout','5s',TRUE);
 PERFORM pg_catalog.set_config('statement_timeout','120s',TRUE);
 IF p_source_size_bytes<>788550
  OR p_source_sha256<>'b7f73cf49b73a055c73b470acff224333ab5ca3e7247c3502f9873cd3f3e4a56'
  OR p_source_commit<>'415f52f39aa0d0cef1cc4e57ea52d46d2c4df69d'
  OR p_schedule IS DISTINCT FROM '{"benchmark_deadline":"2026-09-19T12:11:42Z","final_scoring_close":"2026-09-20T00:11:42Z","publication_deadline":"2026-09-20T00:11:43Z","stage_1_close":"2026-09-19T15:11:43Z","stage_1_scoring_close":"2026-09-19T18:11:42Z","stage_1_start":"2026-09-19T12:11:43Z","stage_2_close":"2026-09-19T21:11:43Z","stage_2_start":"2026-09-19T18:11:43Z","submission_cutoff":"2026-09-19T00:00:00Z","submission_open":"2026-09-18T00:00:00Z"}'::JSONB
  OR p_scorer_digest<>'sha256:2df1280d2b5110cad323a8cdf32b5be96b5a7a1e1ac9edfc57dd48fc385b968e'
  OR p_scorer_reference<>'493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:2df1280d2b5110cad323a8cdf32b5be96b5a7a1e1ac9edfc57dd48fc385b968e' THEN
  RAISE EXCEPTION 'Sep19 rerun315 source, judge, bank, or schedule differs'
   USING ERRCODE='22023';
 END IF;
 PERFORM pg_catalog.pg_advisory_xact_lock(
  pg_catalog.hashtextextended('arena-2026-09-19-rerun315',0));
 LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
 LOCK TABLE public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;
 LOCK TABLE public.lab_arena_runs IN ACCESS EXCLUSIVE MODE;
 LOCK TABLE public.lab_arena_ledger IN ACCESS EXCLUSIVE MODE;
 SELECT * INTO STRICT active_round FROM public.lab_arena_rounds
  WHERE round_id='arena-2026-09-19' FOR UPDATE;
 SELECT * INTO STRICT active_baseline FROM public.lab_arena_submissions
  WHERE round_id='arena-2026-09-19'
   AND submission_id='baseline-2026-09-19' FOR UPDATE;
 IF EXISTS(SELECT 1 FROM public.lab_arena_rounds
   WHERE round_id='arena-2026-09-19-r315archive') THEN
  IF public.lab_arena_sep19_rerun315_archive_valid_v1() IS NOT TRUE
   OR public.lab_arena_sep19_rerun315_active_valid_v1() IS NOT TRUE THEN
   RAISE EXCEPTION 'existing Sep19 rerun315 differs' USING ERRCODE='55000';
  END IF;
  RETURN pg_catalog.jsonb_build_object('status','existing',
   'round_id','arena-2026-09-19','fresh_baseline_execute_count',20,
   'retained_miner_execute_count',80,'score_namespace','score:rerun315');
 END IF;
 IF (p_schedule->>'benchmark_deadline')::TIMESTAMPTZ<=pg_catalog.clock_timestamp() THEN
  RAISE EXCEPTION 'Sep19 rerun315 admission window closed' USING ERRCODE='22023';
 END IF;
 SELECT pg_catalog.count(*) INTO active_count FROM public.lab_arena_runs
  WHERE round_id='arena-2026-09-19' AND status IN('pending','leased','submitted');
 SELECT pg_catalog.count(*) INTO unsettled FROM public.lab_arena_submissions s
  CROSS JOIN(VALUES('execute'::TEXT),('score'::TEXT)) k(kind)
  CROSS JOIN LATERAL(SELECT public.lab_arena__successful_call_cost_state(
   s.submission_id,k.kind,NULL) state)c
  WHERE s.round_id='arena-2026-09-19'
   AND (COALESCE((c.state->>'inflight_calls')::BIGINT,-1)<>0
    OR COALESCE((c.state->>'success_unresolved_calls')::BIGINT,-1)<>0);
 IF active_round.status<>'cancelled'
  OR active_round.cancel_reason IS DISTINCT FROM 'execution_incomplete:stage1:5'
  OR public.lab_arena_sep19_rerun313_active_valid_v1() IS NOT TRUE
  OR active_count<>0 OR unsettled<>0
  OR active_round.configuration_doc ? 'company_quality_policy'
  OR pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(active_round)::TEXT,
       'sha256'),'hex')<>'19187b93b5d46f9603afe3f4fff281f62ef57258cdbc4729392b5d789c7a84f1'
  OR pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(active_baseline)::TEXT,
       'sha256'),'hex')<>'a3703c5fb1d8ce5e4884e063cfe3c313439ffe20e674faeee75e340ad1226842'
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-19' AND kind='execute'
       AND status='accepted' AND terminal_cause='accepted' AND output_ref IS NOT NULL)
      <>80
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-19' AND kind='execute')<>120
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-19' AND kind='score')<>0
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-19' AND kind='score'
       AND status NOT IN('accepted','failed'))
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-19' AND kind='score'
       AND assignment_id NOT LIKE '%:score:rerun313')
  OR (SELECT pg_catalog.count(DISTINCT assignment_id) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-19' AND kind='execute'
       AND submission_id='baseline-2026-09-19'
       AND assignment_id LIKE '%:rerun313')<>20
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-19' AND kind='execute'
       AND submission_id='baseline-2026-09-19'
       AND assignment_id NOT LIKE '%:rerun313')
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-19' AND kind='execute'
       AND submission_id='baseline-2026-09-19')<>40
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-19' AND kind='execute'
       AND submission_id='baseline-2026-09-19' AND status='failed')
      <>40
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-19' AND kind='execute'
       AND submission_id='baseline-2026-09-19'
       AND terminal_cause='model_error')<>25
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-19' AND kind='execute'
       AND submission_id='baseline-2026-09-19'
       AND terminal_cause='provider_error')<>15
  OR EXISTS(SELECT 1 FROM public.lab_arena_ledger l
      WHERE l.round_id='arena-2026-09-19'
       AND (l.submission_id='baseline-2026-09-19'
        OR EXISTS(SELECT 1 FROM public.lab_arena_runs r
          WHERE r.round_id='arena-2026-09-19' AND r.kind='execute'
           AND r.submission_id='baseline-2026-09-19' AND r.run_id=l.run_id)))
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-19' AND kind='execute'
       AND submission_id<>'baseline-2026-09-19')<>80 THEN
  RAISE EXCEPTION 'Sep19 rerun315 terminal preimage differs' USING ERRCODE='55000';
 END IF;
 IF pg_catalog.encode(extensions.digest(COALESCE((SELECT pg_catalog.string_agg(
      pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(s)::TEXT,'sha256'),'hex'),''
      ORDER BY submission_id) FROM public.lab_arena_submissions s
      WHERE round_id='arena-2026-09-19' AND submission_id<>'baseline-2026-09-19'),''),
      'sha256'),'hex')<>'781027021d3cadc5f2c0540d9f24a67f65cb25126912ef3544867d5a70153297'
  OR pg_catalog.encode(extensions.digest(COALESCE((SELECT pg_catalog.string_agg(
      pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(r)::TEXT,'sha256'),'hex'),''
      ORDER BY run_id) FROM public.lab_arena_runs r
      WHERE round_id='arena-2026-09-19'),''),'sha256'),'hex')<>'f32cca7cfda7ad689d0bbf34e33f5c071ce77ed03b2bea3bb2e80619626701c1'
  OR pg_catalog.encode(extensions.digest(COALESCE((SELECT pg_catalog.string_agg(
      pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(l)::TEXT,'sha256'),'hex'),''
      ORDER BY entry_id) FROM public.lab_arena_ledger l
      WHERE round_id='arena-2026-09-19'),''),'sha256'),'hex')<>'0d75adca0f362d863da3550493b1dde358580cac2cd26879ca8864cc44e72220' THEN
  RAISE EXCEPTION 'Sep19 rerun315 terminal history differs' USING ERRCODE='55000';
 END IF;
 SELECT pg_catalog.jsonb_agg(pg_catalog.jsonb_build_object(
   'run_id',run_id,'per_icp_score',per_icp_score,
   'qualification_doc',qualification_doc,'updated_at',updated_at) ORDER BY run_id)
  INTO derived FROM public.lab_arena_runs
  WHERE round_id='arena-2026-09-19' AND kind='execute';
 IF pg_catalog.jsonb_array_length(derived)<>120 THEN
  RAISE EXCEPTION 'Sep19 rerun315 derived judgment count differs'; END IF;
 expected_submission:=active_baseline.submission_doc||pg_catalog.jsonb_build_object(
  'source_ref','arena/arena-2026-09-19/sources/baseline-2026-09-19-rerun313-415f52f3.tar.gz','source_size_bytes',788550,
  'source_sha256','b7f73cf49b73a055c73b470acff224333ab5ca3e7247c3502f9873cd3f3e4a56','source_commit','415f52f39aa0d0cef1cc4e57ea52d46d2c4df69d');
 SELECT pg_catalog.jsonb_agg(CASE WHEN item->>'submission_id'='baseline-2026-09-19'
   THEN item||pg_catalog.jsonb_build_object('source_ref','arena/arena-2026-09-19/sources/baseline-2026-09-19-rerun313-415f52f3.tar.gz',
    'source_size_bytes',788550) ELSE item END ORDER BY ordinal)
  INTO new_participants FROM pg_catalog.jsonb_array_elements(active_round.participants)
   WITH ORDINALITY entries(item,ordinal);
 archive_participants:=(SELECT pg_catalog.jsonb_agg(
   item||pg_catalog.jsonb_build_object('submission_id',(item->>'submission_id')||':r315archive')
   ORDER BY ordinal) FROM pg_catalog.jsonb_array_elements(active_round.participants)
   WITH ORDINALITY entries(item,ordinal));
 archive_config:=active_round.configuration_doc||pg_catalog.jsonb_build_object(
  'round_id','arena-2026-09-19-r315archive','mode','shadow','rewards_enabled',FALSE,
  'archived_terminal_status',active_round.status,
  'archived_terminal_cancel_reason',active_round.cancel_reason,
  'archived_rewards_enabled',active_round.rewards_enabled,
  'archived_reward_basis_hash',active_round.reward_basis_hash,
  'archived_reward_basis_doc',active_round.reward_basis_doc,
  'archived_signing_key_doc',active_round.signing_key_doc,
  'archived_effective_reward_epoch',active_round.effective_reward_epoch,
  'archived_reward_activated_at',active_round.reward_activated_at,
  'archived_king_outcome',active_round.king_outcome,
  'archived_king_hotkey',active_round.king_hotkey,
  'archived_king_start_epoch',active_round.king_start_epoch,
  'archived_promotion_required',active_round.promotion_required,
  'archived_promotion_doc',active_round.promotion_doc,
  'archived_baseline_promoted_at',active_round.baseline_promoted_at,
  'archived_terminal_round_sha256','19187b93b5d46f9603afe3f4fff281f62ef57258cdbc4729392b5d789c7a84f1',
  'archived_terminal_baseline_sha256','a3703c5fb1d8ce5e4884e063cfe3c313439ffe20e674faeee75e340ad1226842',
  'archived_terminal_miner_submissions_sha256','781027021d3cadc5f2c0540d9f24a67f65cb25126912ef3544867d5a70153297',
  'archived_terminal_runs_sha256','f32cca7cfda7ad689d0bbf34e33f5c071ce77ed03b2bea3bb2e80619626701c1',
  'archived_terminal_ledger_sha256','0d75adca0f362d863da3550493b1dde358580cac2cd26879ca8864cc44e72220',
  'archived_execution_judgments',derived);
 SELECT pg_catalog.jsonb_build_object(
  'rounds',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r) ORDER BY round_id)
    FROM public.lab_arena_rounds r WHERE round_id NOT IN(
     'arena-2026-09-19','arena-2026-09-19-r315archive')),
  'submissions',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(s) ORDER BY round_id,submission_id)
    FROM public.lab_arena_submissions s WHERE round_id NOT IN(
     'arena-2026-09-19','arena-2026-09-19-r315archive')),
  'runs',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r) ORDER BY run_id)
    FROM public.lab_arena_runs r WHERE round_id NOT IN(
     'arena-2026-09-19','arena-2026-09-19-r315archive')),
  -- The ACCESS EXCLUSIVE lock makes this a transaction-local physical tuple
  -- identity check: any concurrent or same-transaction UPDATE changes xmin/ctid.
  -- It is not a durable content authority outside this transaction.
  'ledger',(SELECT pg_catalog.jsonb_build_object(
    'count',pg_catalog.count(*),
    'transaction_tuple_sha256',pg_catalog.encode(extensions.digest(
     COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.jsonb_build_array(entry_id,xmin::TEXT,ctid::TEXT)::TEXT,
      'sha256'),'hex'),'' ORDER BY entry_id),''),'sha256'),'hex'))
    FROM public.lab_arena_ledger l WHERE round_id NOT IN(
     'arena-2026-09-19','arena-2026-09-19-r315archive')),
  'weights',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(w)
    ORDER BY network,netuid,epoch) FROM public.lab_arena_accepted_weight_states w)
 ) INTO other_before;
 SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r)-'per_icp_score'-'qualification_doc'-'updated_at'
   ORDER BY run_id) INTO miner_runs_before FROM public.lab_arena_runs r
   WHERE round_id='arena-2026-09-19' AND kind='execute'
    AND submission_id<>'baseline-2026-09-19';
 SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(l) ORDER BY entry_id)
  INTO miner_ledger_before FROM public.lab_arena_ledger l
  WHERE round_id='arena-2026-09-19' AND submission_id<>'baseline-2026-09-19'
   AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs s
    WHERE s.round_id='arena-2026-09-19' AND s.kind='score' AND s.run_id=l.run_id);
 ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER;
 ALTER TABLE public.lab_arena_submissions DISABLE TRIGGER USER;
 ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER;
 ALTER TABLE public.lab_arena_ledger DISABLE TRIGGER USER;
 INSERT INTO public.lab_arena_rounds(
  round_id,status,status_generation,stage_generation,configuration_doc,
  rewards_enabled,participants,benchmark_ref,evaluation_date,
  stage1_scoring_plan_doc,stage2_scoring_plan_doc,finalists,publication_doc,
  king_outcome,king_hotkey,king_start_epoch,effective_reward_epoch,
  reward_basis_hash,reward_basis_doc,signing_key_doc,reward_activated_at,
  cancel_reason,published_at,created_at,updated_at,promotion_required,
  promotion_doc,baseline_promoted_at,icp_set_date,confirmation_bank_ref,
  confirmation_bank_hash,confirmation_cohort,stage3_scoring_plan_doc,
  champion_funding_frozen,champion_submission_id,champion_hotkey,
  champion_fallback_providers
 ) SELECT round_id,status,status_generation,stage_generation,configuration_doc,
  rewards_enabled,participants,benchmark_ref,evaluation_date,
  stage1_scoring_plan_doc,stage2_scoring_plan_doc,finalists,publication_doc,
  king_outcome,king_hotkey,king_start_epoch,effective_reward_epoch,
  reward_basis_hash,reward_basis_doc,signing_key_doc,reward_activated_at,
  cancel_reason,published_at,created_at,updated_at,promotion_required,
  promotion_doc,baseline_promoted_at,icp_set_date,confirmation_bank_ref,
  confirmation_bank_hash,confirmation_cohort,stage3_scoring_plan_doc,
  champion_funding_frozen,champion_submission_id,champion_hotkey,
  champion_fallback_providers FROM pg_catalog.jsonb_populate_record(
  NULL::public.lab_arena_rounds,pg_catalog.to_jsonb(active_round)||pg_catalog.jsonb_build_object(
   'round_id','arena-2026-09-19-r315archive','status','cancelled',
   'configuration_doc',archive_config,'participants',archive_participants,
   'rewards_enabled',FALSE,'reward_basis_hash',NULL,'reward_basis_doc',NULL,
   'signing_key_doc',NULL,'effective_reward_epoch',NULL,'reward_activated_at',NULL,
   'cancel_reason','authorized_sep19_terminal313_fixed_baseline_archive'));
 INSERT INTO public.lab_arena_submissions SELECT (pg_catalog.jsonb_populate_record(
  NULL::public.lab_arena_submissions,pg_catalog.to_jsonb(s)||pg_catalog.jsonb_build_object(
   'round_id','arena-2026-09-19-r315archive',
   'submission_id',s.submission_id||':r315archive'))).*
  FROM public.lab_arena_submissions s WHERE s.round_id='arena-2026-09-19';
 UPDATE public.lab_arena_runs SET round_id='arena-2026-09-19-r315archive',
  submission_id=submission_id||':r315archive'
  WHERE round_id='arena-2026-09-19' AND kind='score';
 GET DIAGNOSTICS moved_scores=ROW_COUNT;
 UPDATE public.lab_arena_ledger l SET round_id='arena-2026-09-19-r315archive',
  submission_id=submission_id||':r315archive'
  WHERE round_id='arena-2026-09-19' AND EXISTS(SELECT 1 FROM public.lab_arena_runs s
   WHERE s.round_id='arena-2026-09-19-r315archive' AND s.kind='score' AND s.run_id=l.run_id);
 GET DIAGNOSTICS moved_score_ledger=ROW_COUNT;
 UPDATE public.lab_arena_runs SET round_id='arena-2026-09-19-r315archive',
  submission_id='baseline-2026-09-19:r315archive'
  WHERE round_id='arena-2026-09-19' AND kind='execute'
   AND submission_id='baseline-2026-09-19';
 GET DIAGNOSTICS moved_baseline_runs=ROW_COUNT;
 UPDATE public.lab_arena_ledger SET round_id='arena-2026-09-19-r315archive',
  submission_id='baseline-2026-09-19:r315archive'
  WHERE round_id='arena-2026-09-19' AND submission_id='baseline-2026-09-19';
 GET DIAGNOSTICS moved_baseline_ledger=ROW_COUNT;
 UPDATE public.lab_arena_runs SET per_icp_score=NULL,qualification_doc=NULL
  WHERE round_id='arena-2026-09-19' AND kind='execute'
   AND submission_id<>'baseline-2026-09-19';
 UPDATE public.lab_arena_submissions SET source_ref='arena/arena-2026-09-19/sources/baseline-2026-09-19-rerun313-415f52f3.tar.gz',
  source_size_bytes=788550,submission_doc=expected_submission,
  updated_at=pg_catalog.clock_timestamp()
  WHERE round_id='arena-2026-09-19' AND submission_id='baseline-2026-09-19';
 IF NOT FOUND THEN RAISE EXCEPTION 'Sep19 rerun315 baseline update missing'; END IF;
 UPDATE public.lab_arena_rounds SET status='stage1',
  status_generation=status_generation+1,stage_generation=stage_generation+1,
  configuration_doc=pg_catalog.jsonb_set(pg_catalog.jsonb_set(pg_catalog.jsonb_set(
   configuration_doc,'{schedule}',p_schedule,TRUE),'{scorer_image_digest}',
   pg_catalog.to_jsonb(p_scorer_digest),TRUE),'{scorer_image_reference}',
   pg_catalog.to_jsonb(p_scorer_reference),TRUE),participants=new_participants,
  stage1_scoring_plan_doc=NULL,stage2_scoring_plan_doc=NULL,stage3_scoring_plan_doc=NULL,
  finalists=NULL,publication_doc=NULL,published_at=NULL,cancel_reason=NULL,
  updated_at=pg_catalog.clock_timestamp()
  WHERE round_id='arena-2026-09-19';
 IF NOT FOUND THEN RAISE EXCEPTION 'Sep19 rerun315 round update missing'; END IF;
 inserted_runs:=0;
 FOR position IN 0..19 LOOP
  stage:=CASE WHEN position<10 THEN 1 ELSE 2 END;
  assignment:='arena-2026-09-19:baseline-2026-09-19:'||stage::TEXT||':'||
   position::TEXT||':rerun315';
  INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,
   miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation)
  VALUES(assignment||':1',assignment,'arena-2026-09-19','baseline-2026-09-19',
   active_baseline.miner_hotkey,stage,position,1,'execute','pending',
   active_round.stage_generation+1);
  inserted_runs:=inserted_runs+1;
 END LOOP;
 ALTER TABLE public.lab_arena_ledger ENABLE TRIGGER USER;
 ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER;
 ALTER TABLE public.lab_arena_submissions ENABLE TRIGGER USER;
 ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER;
 SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r)-'per_icp_score'-'qualification_doc'-'updated_at'
   ORDER BY run_id) INTO miner_runs_after FROM public.lab_arena_runs r
   WHERE round_id='arena-2026-09-19' AND kind='execute'
    AND submission_id<>'baseline-2026-09-19';
 SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(l) ORDER BY entry_id)
  INTO miner_ledger_after FROM public.lab_arena_ledger l
  WHERE round_id='arena-2026-09-19' AND submission_id<>'baseline-2026-09-19';
 SELECT pg_catalog.jsonb_build_object(
  'rounds',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r) ORDER BY round_id)
    FROM public.lab_arena_rounds r WHERE round_id NOT IN(
     'arena-2026-09-19','arena-2026-09-19-r315archive')),
  'submissions',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(s) ORDER BY round_id,submission_id)
    FROM public.lab_arena_submissions s WHERE round_id NOT IN(
     'arena-2026-09-19','arena-2026-09-19-r315archive')),
  'runs',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r) ORDER BY run_id)
    FROM public.lab_arena_runs r WHERE round_id NOT IN(
     'arena-2026-09-19','arena-2026-09-19-r315archive')),
  'ledger',(SELECT pg_catalog.jsonb_build_object(
    'count',pg_catalog.count(*),
    'transaction_tuple_sha256',pg_catalog.encode(extensions.digest(
     COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
      pg_catalog.jsonb_build_array(entry_id,xmin::TEXT,ctid::TEXT)::TEXT,
      'sha256'),'hex'),'' ORDER BY entry_id),''),'sha256'),'hex'))
    FROM public.lab_arena_ledger l WHERE round_id NOT IN(
     'arena-2026-09-19','arena-2026-09-19-r315archive')),
  'weights',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(w)
    ORDER BY network,netuid,epoch) FROM public.lab_arena_accepted_weight_states w)
 ) INTO other_after;
 IF moved_scores<>0
  OR moved_baseline_runs<>40
  OR inserted_runs<>20 OR miner_runs_after IS DISTINCT FROM miner_runs_before
  OR miner_ledger_after IS DISTINCT FROM miner_ledger_before
  OR other_after IS DISTINCT FROM other_before
  OR public.lab_arena_sep19_rerun315_archive_valid_v1() IS NOT TRUE
  OR public.lab_arena_sep19_rerun315_active_valid_v1() IS NOT TRUE
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs
    WHERE round_id='arena-2026-09-19' AND kind='score') THEN
  RAISE EXCEPTION 'Sep19 rerun315 atomic preservation differs' USING ERRCODE='55000';
 END IF;
 RETURN pg_catalog.jsonb_build_object('status','prepared',
  'round_id','arena-2026-09-19','fresh_baseline_execute_count',20,
  'retained_miner_execute_count',80,'archived_score_count',moved_scores,
  'archived_score_ledger_count',moved_score_ledger,
  'archived_baseline_ledger_count',moved_baseline_ledger,
  'score_namespace','score:rerun315');
END $prepare315$;
ALTER FUNCTION public.lab_arena_prepare_sep19_rerun315_v1(
 BIGINT,TEXT,TEXT,JSONB,TEXT,TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_prepare_sep19_rerun315_v1(
 BIGINT,TEXT,TEXT,JSONB,TEXT,TEXT)
 FROM PUBLIC,anon,authenticated,service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_prepare_sep19_rerun315_v1(
 BIGINT,TEXT,TEXT,JSONB,TEXT,TEXT) TO lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_sep19_rerun315_score_namespace_guard_v1()
RETURNS TRIGGER LANGUAGE plpgsql SECURITY DEFINER
SET search_path=pg_catalog,public AS $score_guard315$
BEGIN
 IF NEW.round_id='arena-2026-09-19' AND NEW.kind='score'
  AND EXISTS(SELECT 1 FROM public.lab_arena_rounds
   WHERE round_id='arena-2026-09-19-r315archive')
  AND NEW.assignment_id IS DISTINCT FROM NEW.round_id||':'||NEW.submission_id||':'||
   NEW.stage::TEXT||':'||NEW.icp_position::TEXT||':score:rerun315' THEN
  RAISE EXCEPTION 'Sep19 rerun315 scoring requires exact namespace'
   USING ERRCODE='55000';
 END IF;
 RETURN NEW;
END $score_guard315$;
ALTER FUNCTION public.lab_arena_sep19_rerun315_score_namespace_guard_v1()
 OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep19_rerun315_score_namespace_guard_v1()
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
DROP TRIGGER IF EXISTS lab_arena_sep19_rerun313_score_namespace_guard
 ON public.lab_arena_runs;
DROP TRIGGER IF EXISTS lab_arena_sep19_rerun315_score_namespace_guard
 ON public.lab_arena_runs;
CREATE TRIGGER lab_arena_sep19_rerun315_score_namespace_guard
 BEFORE INSERT ON public.lab_arena_runs FOR EACH ROW
 EXECUTE FUNCTION public.lab_arena_sep19_rerun315_score_namespace_guard_v1();

SELECT public.lab_arena_prepare_sep19_rerun315_v1(
 788550,'b7f73cf49b73a055c73b470acff224333ab5ca3e7247c3502f9873cd3f3e4a56','415f52f39aa0d0cef1cc4e57ea52d46d2c4df69d',
 '{"benchmark_deadline":"2026-09-19T12:11:42Z","final_scoring_close":"2026-09-20T00:11:42Z","publication_deadline":"2026-09-20T00:11:43Z","stage_1_close":"2026-09-19T15:11:43Z","stage_1_scoring_close":"2026-09-19T18:11:42Z","stage_1_start":"2026-09-19T12:11:43Z","stage_2_close":"2026-09-19T21:11:43Z","stage_2_start":"2026-09-19T18:11:43Z","submission_cutoff":"2026-09-19T00:00:00Z","submission_open":"2026-09-18T00:00:00Z"}'::JSONB,
 'sha256:2df1280d2b5110cad323a8cdf32b5be96b5a7a1e1ac9edfc57dd48fc385b968e','493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:2df1280d2b5110cad323a8cdf32b5be96b5a7a1e1ac9edfc57dd48fc385b968e');

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst,'reload schema';
COMMIT;
