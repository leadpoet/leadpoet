BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '120s';

-- Rerun304 reuses 80 accepted miner executions and creates 20 baseline
-- assignments. Its publication boundary must follow the production scoring
-- plan: an accepted execution needs one fresh judgment, while an authorized
-- model-caused zero has no judgment and is persisted on the selected failed
-- execution attempt. Infrastructure failures never authorize a zero.
CREATE OR REPLACE FUNCTION public.lab_arena_sep18_newjudge_rerun304_publication_guard_v1()
RETURNS TRIGGER LANGUAGE plpgsql SECURITY DEFINER
SET search_path=pg_catalog,public AS $publication304$
DECLARE
 archived public.lab_arena_rounds;
 terminal JSONB;
 terminal_config JSONB;
 unsettled BIGINT;
 completion_invalid BOOLEAN;
 plan_invalid BOOLEAN;
 work_count BIGINT;
 zero_count BIGINT;
BEGIN
 IF NEW.round_id='arena-2026-09-18' AND OLD.status='scored' AND NEW.status='published'
  AND EXISTS(SELECT 1 FROM public.lab_arena_rounds WHERE round_id='arena-2026-09-18-rerun303archive') THEN
  SELECT * INTO STRICT archived FROM public.lab_arena_rounds WHERE round_id='arena-2026-09-18-rerun303archive';
  terminal_config:=(archived.configuration_doc-'archived_execution_judgments'
   -'archived_reward_basis_hash'-'archived_effective_reward_epoch'
   -'archived_reward_activated_at')||pg_catalog.jsonb_build_object(
    'round_id','arena-2026-09-18','mode','live','rewards_enabled',TRUE);
  terminal:=(pg_catalog.to_jsonb(archived)-'configuration_doc')||pg_catalog.jsonb_build_object(
   'round_id','arena-2026-09-18','status','published','configuration_doc',terminal_config,
   'rewards_enabled',TRUE,
   'reward_basis_hash',archived.configuration_doc->'archived_reward_basis_hash',
   'effective_reward_epoch',archived.configuration_doc->'archived_effective_reward_epoch',
   'reward_activated_at',archived.configuration_doc->'archived_reward_activated_at',
   'cancel_reason',NULL);

  plan_invalid:=NEW.stage1_scoring_plan_doc IS NULL
   OR NEW.stage2_scoring_plan_doc IS NULL
   OR pg_catalog.jsonb_typeof(NEW.stage1_scoring_plan_doc) IS DISTINCT FROM 'object'
   OR pg_catalog.jsonb_typeof(NEW.stage2_scoring_plan_doc) IS DISTINCT FROM 'object'
   OR NEW.stage1_scoring_plan_doc->>'schema_version' IS DISTINCT FROM 'leadpoet.lab_arena.scoring_plan.v1'
   OR NEW.stage2_scoring_plan_doc->>'schema_version' IS DISTINCT FROM 'leadpoet.lab_arena.scoring_plan.v1'
   OR NEW.stage1_scoring_plan_doc->>'round_id' IS DISTINCT FROM NEW.round_id
   OR NEW.stage2_scoring_plan_doc->>'round_id' IS DISTINCT FROM NEW.round_id
   OR NEW.stage1_scoring_plan_doc->>'stage' IS DISTINCT FROM '1'
   OR NEW.stage2_scoring_plan_doc->>'stage' IS DISTINCT FROM '2'
   OR pg_catalog.jsonb_typeof(NEW.stage1_scoring_plan_doc->'work_items') IS DISTINCT FROM 'array'
   OR pg_catalog.jsonb_typeof(NEW.stage2_scoring_plan_doc->'work_items') IS DISTINCT FROM 'array'
   OR pg_catalog.jsonb_typeof(NEW.stage1_scoring_plan_doc->'zero_rows') IS DISTINCT FROM 'array'
   OR pg_catalog.jsonb_typeof(NEW.stage2_scoring_plan_doc->'zero_rows') IS DISTINCT FROM 'array';

  WITH plans(stage,doc) AS (
    VALUES (1,NEW.stage1_scoring_plan_doc),(2,NEW.stage2_scoring_plan_doc)
  ), work AS (
    SELECT p.stage,item->>'submission_id' submission_id,
      (item->>'icp_position')::INTEGER icp_position,
      item->>'scored_run_id' scored_run_id,item->>'output_ref' output_ref
    FROM plans p CROSS JOIN LATERAL pg_catalog.jsonb_array_elements(
      CASE WHEN pg_catalog.jsonb_typeof(p.doc->'work_items')='array'
       THEN p.doc->'work_items' ELSE '[]'::JSONB END) item
  ), zeros AS (
    SELECT p.stage,item->>'submission_id' submission_id,
      (item->>'icp_position')::INTEGER icp_position,item->>'cause' cause
    FROM plans p CROSS JOIN LATERAL pg_catalog.jsonb_array_elements(
      CASE WHEN pg_catalog.jsonb_typeof(p.doc->'zero_rows')='array'
       THEN p.doc->'zero_rows' ELSE '[]'::JSONB END) item
  ), effective AS (
    SELECT stage,submission_id,icp_position,'work'::TEXT terminal_kind FROM work
    UNION ALL
    SELECT stage,submission_id,icp_position,'zero'::TEXT terminal_kind FROM zeros
  )
  SELECT (SELECT pg_catalog.count(*) FROM work),
    (SELECT pg_catalog.count(*) FROM zeros),
    plan_invalid
    OR (SELECT pg_catalog.count(*) FROM effective)<>100
    OR (SELECT pg_catalog.count(DISTINCT submission_id||':'||icp_position::TEXT) FROM effective)<>100
    OR EXISTS(
      SELECT 1 FROM effective e
      WHERE e.icp_position NOT BETWEEN (e.stage-1)*10 AND e.stage*10-1
       OR NOT EXISTS(SELECT 1 FROM public.lab_arena_submissions s
         WHERE s.round_id=NEW.round_id AND s.submission_id=e.submission_id))
    OR EXISTS(
      SELECT 1 FROM work w LEFT JOIN public.lab_arena_runs e
       ON e.round_id=NEW.round_id AND e.run_id=w.scored_run_id
       AND e.kind='execute' AND e.submission_id=w.submission_id
       AND e.stage=w.stage AND e.icp_position=w.icp_position
       AND e.status='accepted' AND e.terminal_cause='accepted'
       AND e.output_ref=w.output_ref
      WHERE e.run_id IS NULL)
    OR EXISTS(
      SELECT 1 FROM public.lab_arena_runs e
      WHERE e.round_id=NEW.round_id AND e.kind='execute'
       AND e.status='accepted' AND e.terminal_cause='accepted'
       AND NOT EXISTS(SELECT 1 FROM work w WHERE w.scored_run_id=e.run_id))
    OR (SELECT pg_catalog.count(*) FROM work w
      WHERE w.submission_id<>'baseline-2026-09-18')<>80
    OR EXISTS(
      SELECT 1 FROM zeros z
      LEFT JOIN LATERAL (
        SELECT e.* FROM public.lab_arena_runs e
        WHERE e.round_id=NEW.round_id AND e.kind='execute'
         AND e.submission_id=z.submission_id AND e.stage=z.stage
         AND e.icp_position=z.icp_position
        ORDER BY e.attempt DESC,e.run_id DESC LIMIT 1
      ) latest ON TRUE
      LEFT JOIN LATERAL (
        SELECT e.terminal_cause FROM public.lab_arena_runs e
        WHERE e.round_id=NEW.round_id AND e.kind='execute'
         AND e.submission_id=z.submission_id AND e.stage=z.stage
         AND e.icp_position=z.icp_position
         AND e.terminal_cause IN('model_timeout','invalid_output','budget_exhausted','credential_error','model_error')
        ORDER BY e.attempt DESC,e.run_id DESC LIMIT 1
      ) model_failure ON TRUE
      WHERE z.cause NOT IN('model_timeout','invalid_output','budget_exhausted','credential_error','model_error')
       OR latest.run_id IS NULL OR latest.status<>'failed'
       OR latest.output_ref IS NOT NULL
       OR latest.per_icp_score IS DISTINCT FROM 0::DOUBLE PRECISION
       OR EXISTS(SELECT 1 FROM public.lab_arena_runs accepted
          WHERE accepted.round_id=NEW.round_id AND accepted.kind='execute'
           AND accepted.submission_id=z.submission_id AND accepted.stage=z.stage
           AND accepted.icp_position=z.icp_position AND accepted.status='accepted')
       OR z.cause IS DISTINCT FROM CASE
          WHEN latest.terminal_cause IN('model_timeout','invalid_output','budget_exhausted','credential_error','model_error')
          THEN latest.terminal_cause ELSE model_failure.terminal_cause END)
    OR EXISTS(
      SELECT 1 FROM public.lab_arena_runs e
      WHERE e.round_id=NEW.round_id AND e.kind='execute'
       AND e.status NOT IN('accepted','failed','cancelled'))
    OR EXISTS(
      SELECT 1 FROM work w
      WHERE NOT EXISTS(
        SELECT 1 FROM public.lab_arena_runs s
        WHERE s.round_id=NEW.round_id AND s.kind='score'
         AND s.scored_run_id=w.scored_run_id AND s.submission_id=w.submission_id
         AND s.stage=w.stage AND s.icp_position=w.icp_position
         AND s.status='accepted' AND s.terminal_cause='accepted'
         AND s.assignment_id=NEW.round_id||':'||w.submission_id||':'||w.stage::TEXT||':'||w.icp_position::TEXT||':score:rerun304'
         AND s.judgment_scope_doc->>'scorer_image_digest'='sha256:085e6b586ccac28ca7ec2c8996dd871f1a53050e843398b514f67d8c798aae5c'
         AND s.judgment_scope_doc->>'scorer_image_reference'='493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:085e6b586ccac28ca7ec2c8996dd871f1a53050e843398b514f67d8c798aae5c'))
    OR EXISTS(
      SELECT 1 FROM public.lab_arena_runs s
      WHERE s.round_id=NEW.round_id AND s.kind='score'
       AND (s.status NOT IN('accepted','failed')
        OR s.assignment_id IS DISTINCT FROM NEW.round_id||':'||s.submission_id||':'||s.stage::TEXT||':'||s.icp_position::TEXT||':score:rerun304'
        OR s.judgment_scope_doc->>'scorer_image_digest' IS DISTINCT FROM 'sha256:085e6b586ccac28ca7ec2c8996dd871f1a53050e843398b514f67d8c798aae5c'
        OR s.judgment_scope_doc->>'scorer_image_reference' IS DISTINCT FROM '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:085e6b586ccac28ca7ec2c8996dd871f1a53050e843398b514f67d8c798aae5c'
        OR NOT EXISTS(SELECT 1 FROM work w WHERE w.scored_run_id=s.scored_run_id)))
    OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs s
      WHERE s.round_id=NEW.round_id AND s.kind='score'
       AND s.status='accepted' AND s.terminal_cause='accepted')<>(SELECT pg_catalog.count(*) FROM work)
    OR (SELECT pg_catalog.count(DISTINCT s.scored_run_id) FROM public.lab_arena_runs s
      WHERE s.round_id=NEW.round_id AND s.kind='score'
       AND s.status='accepted' AND s.terminal_cause='accepted')<>(SELECT pg_catalog.count(*) FROM work)
  INTO work_count,zero_count,completion_invalid;

  SELECT pg_catalog.count(*) INTO unsettled FROM public.lab_arena_submissions s
   CROSS JOIN(VALUES('execute'::TEXT),('score'::TEXT)) k(kind)
   CROSS JOIN LATERAL(SELECT public.lab_arena__successful_call_cost_state(s.submission_id,k.kind,NULL) state)c
   WHERE s.round_id=NEW.round_id AND (COALESCE((c.state->>'inflight_calls')::BIGINT,-1)<>0
    OR COALESCE((c.state->>'success_unresolved_calls')::BIGINT,-1)<>0);
  IF pg_catalog.encode(extensions.digest(terminal::TEXT,'sha256'),'hex')<>'c51c2f4a259ab68ea8615e644deae95e9fc4fd775bee951a2f30004401f8ac5b'
   OR public.lab_arena_sep18_rerun303_archive_valid304_v1() IS NOT TRUE
   OR public.lab_arena_sep18_newjudge_rerun304_active_valid_v1() IS NOT TRUE
   OR completion_invalid OR work_count+zero_count<>100
   OR unsettled<>0
   OR NEW.reward_basis_hash IS DISTINCT FROM terminal->>'reward_basis_hash'
   OR NEW.reward_basis_doc IS DISTINCT FROM terminal->'reward_basis_doc'
   OR NEW.signing_key_doc IS DISTINCT FROM terminal->'signing_key_doc'
   OR NEW.effective_reward_epoch IS DISTINCT FROM (terminal->>'effective_reward_epoch')::BIGINT
   OR NEW.reward_activated_at IS DISTINCT FROM (terminal->>'reward_activated_at')::TIMESTAMPTZ
   OR NEW.king_outcome IS DISTINCT FROM terminal->>'king_outcome'
   OR NEW.king_hotkey IS DISTINCT FROM terminal->>'king_hotkey'
   OR NEW.king_start_epoch IS DISTINCT FROM (terminal->>'king_start_epoch')::BIGINT
   OR NEW.promotion_required IS DISTINCT FROM (terminal->>'promotion_required')::BOOLEAN
   OR NEW.promotion_doc IS DISTINCT FROM NULLIF(terminal->'promotion_doc','null'::JSONB)
   OR NEW.baseline_promoted_at IS DISTINCT FROM (terminal->>'baseline_promoted_at')::TIMESTAMPTZ
   OR NEW.champion_funding_frozen IS DISTINCT FROM (terminal->>'champion_funding_frozen')::BOOLEAN
   OR NEW.champion_submission_id IS DISTINCT FROM terminal->>'champion_submission_id'
   OR NEW.champion_hotkey IS DISTINCT FROM terminal->>'champion_hotkey'
   OR NEW.champion_fallback_providers IS DISTINCT FROM ARRAY(SELECT pg_catalog.jsonb_array_elements_text(COALESCE(terminal->'champion_fallback_providers','[]'::JSONB)))
   OR NEW.publication_doc#>>'{king_decision,outcome}' IS DISTINCT FROM terminal#>>'{publication_doc,king_decision,outcome}'
   OR NEW.publication_doc#>>'{king_decision,king_submission_id}' IS DISTINCT FROM terminal#>>'{publication_doc,king_decision,king_submission_id}'
   OR NEW.publication_doc#>>'{king_decision,king_hotkey}' IS DISTINCT FROM terminal#>>'{publication_doc,king_decision,king_hotkey}'
   OR NEW.publication_doc#>>'{king_decision,winner_submission_id}' IS DISTINCT FROM terminal#>>'{publication_doc,king_decision,winner_submission_id}' THEN
   RAISE EXCEPTION 'Sep18 rerun304 publication conflicts with sealed effective completion or reward authority' USING ERRCODE='55000';
  END IF;
 END IF;
 RETURN NEW;
END $publication304$;
ALTER FUNCTION public.lab_arena_sep18_newjudge_rerun304_publication_guard_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep18_newjudge_rerun304_publication_guard_v1() FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

COMMIT;
