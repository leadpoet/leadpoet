-- Render only after the Sep20 rerun328 round publishes exactly one numeric zero
-- baseline, is drained, every accepted output artifact is hash-readable, and every
-- protected database seal below matches. Rejudge the sealed rerun328 executions
-- under the fixed scorer without repeating paid sourcing or changing source/cost policy.
-- The complete prior publication remains in r332archive. Activated reward and
-- completed promotion authority remain byte-identical on the canonical row.
BEGIN;
SET LOCAL TIME ZONE 'UTC';
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='120s';

-- Keep the immutable score run IDs in the archive. Patch only the exact
-- scoring initializer preimage so fresh score rows use a new namespace.
DO $patch_sep20_rerun332_scoring$
DECLARE
 definition TEXT;
 definition_hash TEXT;
 anchor TEXT:=$anchor$    v_status := CASE WHEN v_cache.cache_key IS NULL THEN 'pending' ELSE 'accepted' END;$anchor$;
 replacement TEXT:=$replacement$    IF p_round_id='arena-2026-09-20'
       AND EXISTS(SELECT 1 FROM public.lab_arena_rounds
         WHERE round_id='arena-2026-09-20-r332archive') THEN
      v_assignment:=p_round_id||':'||v_scored.submission_id||':'||p_stage::TEXT||':'||
        v_scored.icp_position::TEXT||':score:rerun332';
    END IF;
    v_status := CASE WHEN v_cache.cache_key IS NULL THEN 'pending' ELSE 'accepted' END;$replacement$;
BEGIN
 definition:=pg_catalog.pg_get_functiondef(
  'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::pg_catalog.regprocedure);
 definition_hash:=pg_catalog.encode(extensions.digest(definition,'sha256'),'hex');
 IF definition_hash='0d2a69c1a4dda9a23ab36ba60219fe36da2e729dde971699eddf829b7034e3f9' THEN
  IF (pg_catalog.length(definition)-pg_catalog.length(pg_catalog.replace(
      definition,anchor,'')))<>pg_catalog.length(anchor) THEN
   RAISE EXCEPTION 'Sep20 rerun332 scoring seam differs';
  END IF;
  EXECUTE pg_catalog.replace(definition,anchor,replacement);
 ELSIF definition_hash<>'3b551cc00f123897e6672df49f7aa7eb6092dc51d62a871725c5f8a4627b4474' THEN
  RAISE EXCEPTION 'Sep20 rerun332 scoring definition differs';
 END IF;
 IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
   'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::pg_catalog.regprocedure),
   'sha256'),'hex')<>'3b551cc00f123897e6672df49f7aa7eb6092dc51d62a871725c5f8a4627b4474' THEN
  RAISE EXCEPTION 'Sep20 rerun332 scoring patch differs';
 END IF;
END $patch_sep20_rerun332_scoring$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;
CREATE OR REPLACE FUNCTION public.lab_arena_sep20_rejudge332_publication_stop_v1()
RETURNS TRIGGER LANGUAGE plpgsql SECURITY DEFINER
SET search_path=pg_catalog,public AS $publication_stop332$
BEGIN
 IF NEW.round_id='arena-2026-09-20' AND OLD.status='scored'
  AND NEW.status='published' AND EXISTS(SELECT 1 FROM public.lab_arena_rounds
   WHERE round_id='arena-2026-09-20-r332archive') THEN
  RAISE EXCEPTION 'Sep20 rejudge332 publication requires sealed review release'
   USING ERRCODE='55000';
 END IF;
 RETURN NEW;
END $publication_stop332$;
ALTER FUNCTION public.lab_arena_sep20_rejudge332_publication_stop_v1()
 OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep20_rejudge332_publication_stop_v1()
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
DROP TRIGGER IF EXISTS lab_arena_000_sep20_rejudge332_publication_stop
 ON public.lab_arena_rounds;
CREATE TRIGGER lab_arena_000_sep20_rejudge332_publication_stop
 BEFORE UPDATE ON public.lab_arena_rounds FOR EACH ROW
 EXECUTE FUNCTION public.lab_arena_sep20_rejudge332_publication_stop_v1();

DO $sep20_rerun332$
DECLARE
 active public.lab_arena_rounds;
 baseline public.lab_arena_submissions;
 archive_config JSONB;
 archive_participants JSONB;
 submissions_before JSONB;
 submissions_after JSONB;
 execution_runs_before JSONB;
 execution_runs_after JSONB;
 execution_ledger_before JSONB;
 execution_ledger_after JSONB;
 archived_execution_judgments JSONB;
 prior_archive_before JSONB;
 prior_archive_after JSONB;
 judgments_before JSONB;
 judgments_after JSONB;
 cache_before JSONB;
 cache_after JSONB;
 authority_before JSONB;
 authority_after JSONB;
 active_count BIGINT;
 unsettled_count BIGINT;
 reservation_count BIGINT;
 baseline_publication_count BIGINT;
 zero_baseline_publication_count BIGINT;
 moved_score_runs BIGINT;
 moved_score_ledger BIGINT;
BEGIN
 PERFORM pg_catalog.pg_advisory_xact_lock(
  pg_catalog.hashtextextended('arena-2026-09-20-rerun332',0));
 LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
 LOCK TABLE public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;
 LOCK TABLE public.lab_arena_runs IN ACCESS EXCLUSIVE MODE;
 LOCK TABLE public.lab_arena_ledger IN ACCESS EXCLUSIVE MODE;
 LOCK TABLE public.lab_arena_company_judgments IN SHARE MODE;
 LOCK TABLE public.lab_arena_judgment_cache IN SHARE MODE;
 LOCK TABLE public.lab_arena_company_judgment_reservations IN SHARE MODE;

 SELECT * INTO STRICT active FROM public.lab_arena_rounds
  WHERE round_id='arena-2026-09-20' FOR UPDATE;
 SELECT * INTO STRICT baseline FROM public.lab_arena_submissions
  WHERE round_id='arena-2026-09-20'
   AND submission_id='baseline-2026-09-20' FOR UPDATE;

 IF EXISTS(SELECT 1 FROM public.lab_arena_rounds
   WHERE round_id='arena-2026-09-20-r332archive') THEN
  IF NOT EXISTS(SELECT 1 FROM public.lab_arena_rounds archived
      WHERE archived.round_id='arena-2026-09-20-r332archive'
       AND archived.status='cancelled' AND archived.rewards_enabled IS FALSE
       AND archived.cancel_reason=
           'authorized_sep20_authority_preserving_rejudge_archive'
       AND archived.configuration_doc->>'archived_terminal_round_sha256'=
           '418609475683ca81fb0f96cbb0c20d5243b805e5ecfffa875106f4dcab654072'
       AND archived.configuration_doc->>'archived_terminal_submissions_sha256'=
           '80147fc44b90f3e139820c27007a577c9d63f76eff5d4fcb02f1abc194e411a9'
       AND archived.configuration_doc->>'archived_terminal_runs_sha256'=
           '482c43a1e1b85febec6d148667f1df65bdded118c6303443ccbda48f65e3fe1e'
       AND archived.configuration_doc->>'archived_terminal_ledger_sha256'=
           'c5b4d28eaf2d11910de3ab316f38273caa004114696d40c9cd312e6efffe6f4e'
       AND archived.configuration_doc->>'archived_terminal_company_judgments_sha256'=
           '0ef317eb17e2318bae2ef9d4bc8de770cc8cbbae941ab9456941b2c44b9c5c41'
       AND archived.configuration_doc->>'archived_terminal_judgment_cache_sha256'=
           '09bf9030e831972b02474dc75606b39516733417db98d9989a0e2cbf9391fde3'
       AND archived.configuration_doc->>'archived_bank_object_sha256'=
           '7aaa7fbb4b4ce078dd62217c4d594dafdc82161d30c7718bd4500fa659fe5ad9'
       AND archived.configuration_doc->>'archived_execution_artifacts_sha256'=
           '0892ee718698880c31476cb2bd87f052c62e92cd20af359a36b3145f3e4669ae'
       AND archived.configuration_doc->>'archived_authority_sha256'=
           '38ca5046aa2200f555130dec321282ecac92fdf5e0a1a95936222ee210022e4f')
    OR baseline.source_ref IS DISTINCT FROM
       'arena/arena-2026-09-20/sources/baseline-2026-09-20-rerun328-5d492f17.tar.gz'
    OR baseline.source_size_bytes IS DISTINCT FROM 854708
    OR baseline.submission_doc->>'source_sha256' IS DISTINCT FROM
       'f1f24e24e0cd736d8c9695640093f4d5f28c360e2773b9824dd5010fe5100f3e'
    OR baseline.submission_doc->>'source_commit' IS DISTINCT FROM
       '5d492f1715e27c8d9b919bcc3ad69d1be5160524'
    OR active.configuration_doc->>'checkpoint_deadline_policy' IS DISTINCT FROM
       'atomic_checkpoint_90m_v1'
    OR (active.configuration_doc->>'icp_wall_clock_seconds')::INTEGER<>5400
    OR (active.configuration_doc->>'lease_ttl_seconds')::INTEGER<>6300
    OR active.configuration_doc->>'scorer_image_digest' IS DISTINCT FROM
       'sha256:cc2dd55dec0b94db8cd0e184d9bf67014e2e83def22d5c5b4e28213c46fe0cd8'
    OR active.configuration_doc->>'scorer_image_reference' IS DISTINCT FROM
       '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:cc2dd55dec0b94db8cd0e184d9bf67014e2e83def22d5c5b4e28213c46fe0cd8'
    OR active.configuration_doc->'call_quotas' IS DISTINCT FROM
       '{"deepline":200,"openrouter":500,"scrapingdog":200}'::JSONB
    OR active.configuration_doc->'schedule' IS DISTINCT FROM
       '{"benchmark_deadline":"2026-09-20T10:30:00Z","final_scoring_close":"2026-09-21T00:30:00Z","publication_deadline":"2026-09-21T00:30:01Z","stage_1_close":"2026-09-20T17:30:00Z","stage_1_scoring_close":"2026-09-20T20:30:00Z","stage_1_start":"2026-09-20T10:30:01Z","stage_2_close":"2026-09-20T21:30:00Z","stage_2_start":"2026-09-20T20:30:01Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB
    OR active.rewards_enabled IS DISTINCT FROM TRUE
    OR active.promotion_required IS DISTINCT FROM TRUE
    OR active.configuration_doc->'rewards_enabled' IS DISTINCT FROM 'true'::JSONB
    OR active.configuration_doc ? 'sep20_rerun328_promotion_reward_hold'
    OR pg_catalog.encode(extensions.digest(pg_catalog.jsonb_build_object(
       'rewards_enabled',active.rewards_enabled,
       'effective_reward_epoch',active.effective_reward_epoch,
       'reward_basis_hash',active.reward_basis_hash,
       'reward_basis_doc',active.reward_basis_doc,
       'signing_key_doc',active.signing_key_doc,
       'reward_activated_at',active.reward_activated_at,
       'king_outcome',active.king_outcome,'king_hotkey',active.king_hotkey,
       'king_start_epoch',active.king_start_epoch,
       'promotion_required',active.promotion_required,
       'promotion_doc',active.promotion_doc,
       'baseline_promoted_at',active.baseline_promoted_at)::TEXT,'sha256'),'hex')
       <>'38ca5046aa2200f555130dec321282ecac92fdf5e0a1a95936222ee210022e4f'
    OR (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
        WHERE round_id='arena-2026-09-20-r332archive')<>
       8
    OR EXISTS(SELECT 1 FROM public.lab_arena_runs
        WHERE round_id='arena-2026-09-20-r332archive' AND kind='execute')
    OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
        WHERE round_id='arena-2026-09-20-r332archive' AND kind='score')<>
       98
    OR (SELECT pg_catalog.count(DISTINCT assignment_id)
        FROM public.lab_arena_runs WHERE round_id='arena-2026-09-20'
         AND kind='execute' AND submission_id='baseline-2026-09-20'
         AND assignment_id LIKE '%:rerun328')<>20
    OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
        WHERE round_id='arena-2026-09-20' AND kind='execute'
         AND submission_id='baseline-2026-09-20')<>
       24
    OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
        WHERE round_id='arena-2026-09-20' AND kind='execute'
         AND submission_id='baseline-2026-09-20'
         AND status='accepted' AND terminal_cause='accepted')<>
       18
    OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
        WHERE round_id='arena-2026-09-20' AND kind='execute'
         AND submission_id='baseline-2026-09-20' AND status='failed')<>
       6
    OR EXISTS(SELECT 1 FROM public.lab_arena_runs
        WHERE round_id='arena-2026-09-20' AND kind='execute'
         AND submission_id='baseline-2026-09-20'
         AND assignment_id NOT LIKE '%:rerun328')
    OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
        WHERE round_id='arena-2026-09-20' AND kind='execute'
         AND submission_id<>'baseline-2026-09-20' AND status='accepted'
         AND terminal_cause='accepted' AND output_ref IS NOT NULL)<>80 THEN
   RAISE EXCEPTION 'existing Sep20 rerun332 differs' USING ERRCODE='55000';
  END IF;
  RETURN;
 END IF;

 IF ('{"benchmark_deadline":"2026-09-20T10:30:00Z","final_scoring_close":"2026-09-21T00:30:00Z","publication_deadline":"2026-09-21T00:30:01Z","stage_1_close":"2026-09-20T17:30:00Z","stage_1_scoring_close":"2026-09-20T20:30:00Z","stage_1_start":"2026-09-20T10:30:01Z","stage_2_close":"2026-09-20T21:30:00Z","stage_2_start":"2026-09-20T20:30:01Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'submission_open') IS DISTINCT FROM
      '2026-09-19T00:00:00Z'
 OR ('{"benchmark_deadline":"2026-09-20T10:30:00Z","final_scoring_close":"2026-09-21T00:30:00Z","publication_deadline":"2026-09-21T00:30:01Z","stage_1_close":"2026-09-20T17:30:00Z","stage_1_scoring_close":"2026-09-20T20:30:00Z","stage_1_start":"2026-09-20T10:30:01Z","stage_2_close":"2026-09-20T21:30:00Z","stage_2_start":"2026-09-20T20:30:01Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'submission_cutoff') IS DISTINCT FROM
      '2026-09-20T00:00:00Z'
  OR ('{"benchmark_deadline":"2026-09-20T10:30:00Z","final_scoring_close":"2026-09-21T00:30:00Z","publication_deadline":"2026-09-21T00:30:01Z","stage_1_close":"2026-09-20T17:30:00Z","stage_1_scoring_close":"2026-09-20T20:30:00Z","stage_1_start":"2026-09-20T10:30:01Z","stage_2_close":"2026-09-20T21:30:00Z","stage_2_start":"2026-09-20T20:30:01Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'benchmark_deadline')::TIMESTAMPTZ
      <=pg_catalog.clock_timestamp()
  OR ('{"benchmark_deadline":"2026-09-20T10:30:00Z","final_scoring_close":"2026-09-21T00:30:00Z","publication_deadline":"2026-09-21T00:30:01Z","stage_1_close":"2026-09-20T17:30:00Z","stage_1_scoring_close":"2026-09-20T20:30:00Z","stage_1_start":"2026-09-20T10:30:01Z","stage_2_close":"2026-09-20T21:30:00Z","stage_2_start":"2026-09-20T20:30:01Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'stage_1_start')::TIMESTAMPTZ
      IS DISTINCT FROM
      ('{"benchmark_deadline":"2026-09-20T10:30:00Z","final_scoring_close":"2026-09-21T00:30:00Z","publication_deadline":"2026-09-21T00:30:01Z","stage_1_close":"2026-09-20T17:30:00Z","stage_1_scoring_close":"2026-09-20T20:30:00Z","stage_1_start":"2026-09-20T10:30:01Z","stage_2_close":"2026-09-20T21:30:00Z","stage_2_start":"2026-09-20T20:30:01Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'benchmark_deadline')::TIMESTAMPTZ+
       INTERVAL '1 second'
  OR ('{"benchmark_deadline":"2026-09-20T10:30:00Z","final_scoring_close":"2026-09-21T00:30:00Z","publication_deadline":"2026-09-21T00:30:01Z","stage_1_close":"2026-09-20T17:30:00Z","stage_1_scoring_close":"2026-09-20T20:30:00Z","stage_1_start":"2026-09-20T10:30:01Z","stage_2_close":"2026-09-20T21:30:00Z","stage_2_start":"2026-09-20T20:30:01Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'stage_1_close')::TIMESTAMPTZ
      IS DISTINCT FROM
      ('{"benchmark_deadline":"2026-09-20T10:30:00Z","final_scoring_close":"2026-09-21T00:30:00Z","publication_deadline":"2026-09-21T00:30:01Z","stage_1_close":"2026-09-20T17:30:00Z","stage_1_scoring_close":"2026-09-20T20:30:00Z","stage_1_start":"2026-09-20T10:30:01Z","stage_2_close":"2026-09-20T21:30:00Z","stage_2_start":"2026-09-20T20:30:01Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'benchmark_deadline')::TIMESTAMPTZ+
       INTERVAL '7 hours'
  OR ('{"benchmark_deadline":"2026-09-20T10:30:00Z","final_scoring_close":"2026-09-21T00:30:00Z","publication_deadline":"2026-09-21T00:30:01Z","stage_1_close":"2026-09-20T17:30:00Z","stage_1_scoring_close":"2026-09-20T20:30:00Z","stage_1_start":"2026-09-20T10:30:01Z","stage_2_close":"2026-09-20T21:30:00Z","stage_2_start":"2026-09-20T20:30:01Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'stage_1_scoring_close')::TIMESTAMPTZ
      IS DISTINCT FROM
      ('{"benchmark_deadline":"2026-09-20T10:30:00Z","final_scoring_close":"2026-09-21T00:30:00Z","publication_deadline":"2026-09-21T00:30:01Z","stage_1_close":"2026-09-20T17:30:00Z","stage_1_scoring_close":"2026-09-20T20:30:00Z","stage_1_start":"2026-09-20T10:30:01Z","stage_2_close":"2026-09-20T21:30:00Z","stage_2_start":"2026-09-20T20:30:01Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'benchmark_deadline')::TIMESTAMPTZ+
       INTERVAL '10 hours'
  OR ('{"benchmark_deadline":"2026-09-20T10:30:00Z","final_scoring_close":"2026-09-21T00:30:00Z","publication_deadline":"2026-09-21T00:30:01Z","stage_1_close":"2026-09-20T17:30:00Z","stage_1_scoring_close":"2026-09-20T20:30:00Z","stage_1_start":"2026-09-20T10:30:01Z","stage_2_close":"2026-09-20T21:30:00Z","stage_2_start":"2026-09-20T20:30:01Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'stage_2_start')::TIMESTAMPTZ
      IS DISTINCT FROM
      ('{"benchmark_deadline":"2026-09-20T10:30:00Z","final_scoring_close":"2026-09-21T00:30:00Z","publication_deadline":"2026-09-21T00:30:01Z","stage_1_close":"2026-09-20T17:30:00Z","stage_1_scoring_close":"2026-09-20T20:30:00Z","stage_1_start":"2026-09-20T10:30:01Z","stage_2_close":"2026-09-20T21:30:00Z","stage_2_start":"2026-09-20T20:30:01Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'benchmark_deadline')::TIMESTAMPTZ+
       INTERVAL '10 hours 1 second'
  OR ('{"benchmark_deadline":"2026-09-20T10:30:00Z","final_scoring_close":"2026-09-21T00:30:00Z","publication_deadline":"2026-09-21T00:30:01Z","stage_1_close":"2026-09-20T17:30:00Z","stage_1_scoring_close":"2026-09-20T20:30:00Z","stage_1_start":"2026-09-20T10:30:01Z","stage_2_close":"2026-09-20T21:30:00Z","stage_2_start":"2026-09-20T20:30:01Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'stage_2_close')::TIMESTAMPTZ
      IS DISTINCT FROM
      ('{"benchmark_deadline":"2026-09-20T10:30:00Z","final_scoring_close":"2026-09-21T00:30:00Z","publication_deadline":"2026-09-21T00:30:01Z","stage_1_close":"2026-09-20T17:30:00Z","stage_1_scoring_close":"2026-09-20T20:30:00Z","stage_1_start":"2026-09-20T10:30:01Z","stage_2_close":"2026-09-20T21:30:00Z","stage_2_start":"2026-09-20T20:30:01Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'benchmark_deadline')::TIMESTAMPTZ+
       INTERVAL '11 hours'
  OR ('{"benchmark_deadline":"2026-09-20T10:30:00Z","final_scoring_close":"2026-09-21T00:30:00Z","publication_deadline":"2026-09-21T00:30:01Z","stage_1_close":"2026-09-20T17:30:00Z","stage_1_scoring_close":"2026-09-20T20:30:00Z","stage_1_start":"2026-09-20T10:30:01Z","stage_2_close":"2026-09-20T21:30:00Z","stage_2_start":"2026-09-20T20:30:01Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'final_scoring_close')::TIMESTAMPTZ
      IS DISTINCT FROM
      ('{"benchmark_deadline":"2026-09-20T10:30:00Z","final_scoring_close":"2026-09-21T00:30:00Z","publication_deadline":"2026-09-21T00:30:01Z","stage_1_close":"2026-09-20T17:30:00Z","stage_1_scoring_close":"2026-09-20T20:30:00Z","stage_1_start":"2026-09-20T10:30:01Z","stage_2_close":"2026-09-20T21:30:00Z","stage_2_start":"2026-09-20T20:30:01Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'benchmark_deadline')::TIMESTAMPTZ+
       INTERVAL '14 hours'
  OR ('{"benchmark_deadline":"2026-09-20T10:30:00Z","final_scoring_close":"2026-09-21T00:30:00Z","publication_deadline":"2026-09-21T00:30:01Z","stage_1_close":"2026-09-20T17:30:00Z","stage_1_scoring_close":"2026-09-20T20:30:00Z","stage_1_start":"2026-09-20T10:30:01Z","stage_2_close":"2026-09-20T21:30:00Z","stage_2_start":"2026-09-20T20:30:01Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'publication_deadline')::TIMESTAMPTZ
      IS DISTINCT FROM
      ('{"benchmark_deadline":"2026-09-20T10:30:00Z","final_scoring_close":"2026-09-21T00:30:00Z","publication_deadline":"2026-09-21T00:30:01Z","stage_1_close":"2026-09-20T17:30:00Z","stage_1_scoring_close":"2026-09-20T20:30:00Z","stage_1_start":"2026-09-20T10:30:01Z","stage_2_close":"2026-09-20T21:30:00Z","stage_2_start":"2026-09-20T20:30:01Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB->>'benchmark_deadline')::TIMESTAMPTZ+
       INTERVAL '14 hours 1 second' THEN
  RAISE EXCEPTION 'Sep20 rerun332 schedule differs' USING ERRCODE='22023';
 END IF;

 IF 'sha256:cc2dd55dec0b94db8cd0e184d9bf67014e2e83def22d5c5b4e28213c46fe0cd8' !~ '^sha256:[0-9a-f]{64}$'
  OR '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:cc2dd55dec0b94db8cd0e184d9bf67014e2e83def22d5c5b4e28213c46fe0cd8' IS DISTINCT FROM
    '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@'||
    'sha256:cc2dd55dec0b94db8cd0e184d9bf67014e2e83def22d5c5b4e28213c46fe0cd8' THEN
  RAISE EXCEPTION 'Sep20 rerun332 scorer identity differs' USING ERRCODE='22023';
 END IF;

 SELECT pg_catalog.count(*) INTO active_count FROM public.lab_arena_runs
  WHERE round_id='arena-2026-09-20'
   AND status IN('pending','leased','submitted');
 SELECT pg_catalog.count(*) INTO unsettled_count
 FROM public.lab_arena_submissions s
 CROSS JOIN(VALUES('execute'::TEXT),('score'::TEXT)) k(kind)
 CROSS JOIN LATERAL(SELECT public.lab_arena__successful_call_cost_state(
  s.submission_id,k.kind,NULL) state)c
 WHERE s.round_id='arena-2026-09-20'
  AND (COALESCE((c.state->>'inflight_calls')::BIGINT,-1)<>0
   OR COALESCE((c.state->>'success_unresolved_calls')::BIGINT,-1)<>0);
 SELECT pg_catalog.count(*) INTO reservation_count
 FROM public.lab_arena_company_judgment_reservations reservation
 JOIN public.lab_arena_runs run ON run.run_id=reservation.run_id
 WHERE run.round_id='arena-2026-09-20';
 SELECT pg_catalog.count(*),pg_catalog.count(*) FILTER(WHERE
    CASE WHEN pg_catalog.jsonb_typeof(entry->'final_score')='number'
      THEN (entry->>'final_score')::NUMERIC=0 ELSE FALSE END)
 INTO baseline_publication_count,zero_baseline_publication_count
 FROM pg_catalog.jsonb_array_elements(CASE
   WHEN pg_catalog.jsonb_typeof(active.publication_doc->'final_ranking')='array'
   THEN active.publication_doc->'final_ranking' ELSE '[]'::JSONB END) entry
 WHERE entry->>'submission_id'='baseline-2026-09-20';

 -- User-authorized rerun332 is a zero-score recovery only. A positive,
 -- negative, missing, nonnumeric or ambiguous published baseline is final.
 IF baseline_publication_count<>1 OR zero_baseline_publication_count<>1 THEN
  RAISE EXCEPTION 'Sep20 rerun332 requires exactly one zero published baseline'
   USING ERRCODE='55000';
 END IF;

 IF active.status IS DISTINCT FROM 'published'
  OR active.status IS DISTINCT FROM 'published'
  OR active.cancel_reason IS DISTINCT FROM NULL
  OR active_count<>0 OR unsettled_count<>0 OR reservation_count<>0
  OR active.effective_reward_epoch IS NULL
  OR active.reward_basis_hash IS NULL OR active.reward_basis_doc IS NULL
  OR active.signing_key_doc IS NULL OR active.reward_activated_at IS NULL
  OR active.king_outcome IS NULL OR active.king_hotkey IS NULL
  OR active.king_start_epoch IS NULL
  OR active.promotion_doc IS NULL OR active.baseline_promoted_at IS NULL
  OR active.rewards_enabled IS DISTINCT FROM TRUE
  OR active.promotion_required IS DISTINCT FROM TRUE
  OR active.configuration_doc->'rewards_enabled' IS DISTINCT FROM 'true'::JSONB
  OR active.configuration_doc ? 'sep20_rerun328_promotion_reward_hold'
  OR active.publication_doc#>>'{king_decision,outcome}' IS DISTINCT FROM 'crowned'
  OR COALESCE(active.publication_doc#>>'{king_decision,winner_submission_id}','')=''
  OR active.publication_doc#>>'{king_decision,king_hotkey}'
     IS DISTINCT FROM active.king_hotkey
  OR active.reward_basis_doc->>'king_hotkey' IS DISTINCT FROM active.king_hotkey
  OR pg_catalog.encode(extensions.digest(pg_catalog.jsonb_build_object(
      'rewards_enabled',active.rewards_enabled,
      'effective_reward_epoch',active.effective_reward_epoch,
      'reward_basis_hash',active.reward_basis_hash,
      'reward_basis_doc',active.reward_basis_doc,
      'signing_key_doc',active.signing_key_doc,
      'reward_activated_at',active.reward_activated_at,
      'king_outcome',active.king_outcome,'king_hotkey',active.king_hotkey,
      'king_start_epoch',active.king_start_epoch,
      'promotion_required',active.promotion_required,
      'promotion_doc',active.promotion_doc,
      'baseline_promoted_at',active.baseline_promoted_at)::TEXT,'sha256'),'hex')
     <>'38ca5046aa2200f555130dec321282ecac92fdf5e0a1a95936222ee210022e4f'
  OR active.benchmark_ref IS DISTINCT FROM
     'arena/arena-2026-09-20/benchmark.json'
  OR active.evaluation_date IS DISTINCT FROM '2026-09-20'
  OR active.icp_set_date IS DISTINCT FROM DATE '2026-09-19'
  OR pg_catalog.jsonb_array_length(active.participants)<>5
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
      WHERE round_id='arena-2026-09-20')<>8
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
      WHERE round_id='arena-2026-09-20' AND status='frozen')<>5
  OR active.configuration_doc->'call_quotas' IS DISTINCT FROM
     '{"deepline":200,"openrouter":500,"scrapingdog":200}'::JSONB
  OR active.configuration_doc->'scoring_call_quotas' IS DISTINCT FROM
     '{"deepline":40,"openrouter":120,"scrapingdog":150}'::JSONB
  OR active.configuration_doc->>'sourcing_cost_eligibility_policy'
     IS DISTINCT FROM 'successful_calls_per_icp_v1'
  OR (active.configuration_doc->>'execution_icp_cap_microusd')::BIGINT<>4000000
  OR (active.configuration_doc->>'cost_per_company_microusd')::BIGINT<>800000
  OR active.configuration_doc->>'checkpoint_deadline_policy' IS DISTINCT FROM
     'atomic_checkpoint_90m_v1'
  OR (active.configuration_doc->>'icp_wall_clock_seconds')::INTEGER<>5400
  OR (active.configuration_doc->>'lease_ttl_seconds')::INTEGER<>6300
  OR active.configuration_doc->>'scorer_image_digest' IS DISTINCT FROM
     'sha256:333ae499ede5eb51d60385bc2d11c80fed2c2a9ce6922111adde5fa52b40236f'
  OR active.configuration_doc->>'scorer_image_reference' IS DISTINCT FROM
     '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:333ae499ede5eb51d60385bc2d11c80fed2c2a9ce6922111adde5fa52b40236f'
  OR baseline.source_ref IS DISTINCT FROM
     'arena/arena-2026-09-20/sources/baseline-2026-09-20-rerun328-5d492f17.tar.gz'
  OR baseline.source_size_bytes IS DISTINCT FROM 854708
  OR baseline.submission_doc->>'source_sha256' IS DISTINCT FROM
     'f1f24e24e0cd736d8c9695640093f4d5f28c360e2773b9824dd5010fe5100f3e'
  OR baseline.submission_doc->>'source_commit' IS DISTINCT FROM
     '5d492f1715e27c8d9b919bcc3ad69d1be5160524'
  OR pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(active)::TEXT,
       'sha256'),'hex')<>'418609475683ca81fb0f96cbb0c20d5243b805e5ecfffa875106f4dcab654072'
  OR pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(baseline)::TEXT,
       'sha256'),'hex')<>'8cb1fb888a97c2694369f129461bf492e3ce9e0f0a652da68e1d9ebcfeed5809'
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-20')<>202
  OR (SELECT pg_catalog.count(DISTINCT assignment_id) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-20' AND kind='execute')<>100
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-20' AND kind='score')<>
     98
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-20' AND kind='score'
       AND (status<>'accepted' OR terminal_cause<>'accepted'
        OR assignment_id NOT LIKE '%:score:rerun328'))
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-20' AND kind='execute'
       AND submission_id='baseline-2026-09-20')<>
     24
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-20' AND kind='execute'
       AND submission_id='baseline-2026-09-20'
       AND status='accepted' AND terminal_cause='accepted')<>
     18
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-20' AND kind='execute'
       AND submission_id='baseline-2026-09-20' AND status='failed')<>
     6
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-20' AND kind='execute'
       AND submission_id='baseline-2026-09-20'
       AND assignment_id NOT LIKE '%:rerun328')
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-20' AND kind='execute'
       AND submission_id<>'baseline-2026-09-20')<>80
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-20' AND kind='execute'
       AND submission_id<>'baseline-2026-09-20' AND status='accepted'
       AND terminal_cause='accepted' AND output_ref IS NOT NULL)<>80
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-20' AND kind='execute'
       AND status='accepted' AND terminal_cause='accepted'
       AND pg_catalog.char_length(output_ref) BETWEEN 1 AND 1024)<>
     98
  OR '0892ee718698880c31476cb2bd87f052c62e92cd20af359a36b3145f3e4669ae' !~ '^[0-9a-f]{64}$'
  OR EXISTS(SELECT 1 FROM (
      SELECT assignment_id,
       pg_catalog.bool_or(status='accepted') has_accepted,
       pg_catalog.bool_or(COALESCE(terminal_cause,'') IN(
        'model_timeout','invalid_output','budget_exhausted',
        'credential_error','model_error')) has_model_zero,
       (pg_catalog.array_agg(attempt ORDER BY attempt DESC))[1] latest_attempt,
       (pg_catalog.array_agg(status ORDER BY attempt DESC))[1] latest_status,
       (pg_catalog.array_agg(COALESCE(terminal_cause,'')
         ORDER BY attempt DESC))[1] latest_cause
      FROM public.lab_arena_runs WHERE round_id='arena-2026-09-20'
       AND kind='execute' GROUP BY assignment_id) outcome
     WHERE NOT has_accepted AND NOT has_model_zero AND NOT(
       latest_attempt>=2 AND latest_status='failed'
       AND latest_cause='provider_error'))
  OR NOT EXISTS(SELECT 1 FROM public.lab_arena_rounds archived
      WHERE archived.round_id='arena-2026-09-20-r326archive'
       AND archived.configuration_doc->>'archived_bank_object_sha256'=
       '7aaa7fbb4b4ce078dd62217c4d594dafdc82161d30c7718bd4500fa659fe5ad9')
  OR NOT EXISTS(SELECT 1 FROM public.lab_arena_rounds archived
      WHERE archived.round_id='arena-2026-09-20-r328archive'
       AND archived.configuration_doc->>'archived_bank_object_sha256'=
       '7aaa7fbb4b4ce078dd62217c4d594dafdc82161d30c7718bd4500fa659fe5ad9') THEN
  RAISE EXCEPTION 'Sep20 rerun332 terminal preimage differs' USING ERRCODE='55000';
 END IF;

 IF pg_catalog.encode(extensions.digest(COALESCE((SELECT pg_catalog.string_agg(
    pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(s)::TEXT,'sha256'),'hex'),''
    ORDER BY submission_id) FROM public.lab_arena_submissions s
    WHERE round_id='arena-2026-09-20'),''),'sha256'),'hex')<>
       '80147fc44b90f3e139820c27007a577c9d63f76eff5d4fcb02f1abc194e411a9'
  OR pg_catalog.encode(extensions.digest(COALESCE((SELECT pg_catalog.string_agg(
    pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(r)::TEXT,'sha256'),'hex'),''
    ORDER BY run_id) FROM public.lab_arena_runs r
    WHERE round_id='arena-2026-09-20'),''),'sha256'),'hex')<>
       '482c43a1e1b85febec6d148667f1df65bdded118c6303443ccbda48f65e3fe1e'
  OR pg_catalog.encode(extensions.digest(COALESCE((SELECT pg_catalog.string_agg(
    pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(l)::TEXT,'sha256'),'hex'),''
    ORDER BY entry_id) FROM public.lab_arena_ledger l
    WHERE round_id='arena-2026-09-20'),''),'sha256'),'hex')<>
       'c5b4d28eaf2d11910de3ab316f38273caa004114696d40c9cd312e6efffe6f4e'
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
      WHERE round_id='arena-2026-09-20')<>40771
  OR pg_catalog.encode(extensions.digest(COALESCE((SELECT pg_catalog.string_agg(
    pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(j)::TEXT,'sha256'),'hex'),''
    ORDER BY cache_key,authority_slot) FROM public.lab_arena_company_judgments j),''),
    'sha256'),'hex')<>'0ef317eb17e2318bae2ef9d4bc8de770cc8cbbae941ab9456941b2c44b9c5c41'
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_company_judgments)<>
       67
  OR pg_catalog.encode(extensions.digest(COALESCE((SELECT pg_catalog.string_agg(
    pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(c)::TEXT,'sha256'),'hex'),''
    ORDER BY cache_key) FROM public.lab_arena_judgment_cache c),''),
    'sha256'),'hex')<>'09bf9030e831972b02474dc75606b39516733417db98d9989a0e2cbf9391fde3'
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_judgment_cache)<>
       866 THEN
  RAISE EXCEPTION 'Sep20 rerun332 terminal history differs' USING ERRCODE='55000';
 END IF;

 SELECT pg_catalog.jsonb_build_object(
  'rounds',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r) ORDER BY round_id)
    FROM public.lab_arena_rounds r WHERE round_id IN
     ('arena-2026-09-20-r326archive','arena-2026-09-20-r328archive')),
  'submissions',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(s)
    ORDER BY round_id,submission_id) FROM public.lab_arena_submissions s
    WHERE round_id IN
     ('arena-2026-09-20-r326archive','arena-2026-09-20-r328archive')),
  'runs',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r)
    ORDER BY round_id,run_id) FROM public.lab_arena_runs r WHERE round_id IN
     ('arena-2026-09-20-r326archive','arena-2026-09-20-r328archive')),
  'ledger',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(l)
    ORDER BY round_id,entry_id) FROM public.lab_arena_ledger l WHERE round_id IN
     ('arena-2026-09-20-r326archive','arena-2026-09-20-r328archive')))
 INTO prior_archive_before;
 SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256',
   pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(
    pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(j)::TEXT,'sha256'),'hex'),''
    ORDER BY cache_key,authority_slot),''),'sha256'),'hex'))
  INTO judgments_before FROM public.lab_arena_company_judgments j;
 SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256',
   pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(
    pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(c)::TEXT,'sha256'),'hex'),''
    ORDER BY cache_key),''),'sha256'),'hex'))
  INTO cache_before FROM public.lab_arena_judgment_cache c;

 authority_before:=pg_catalog.jsonb_build_object(
  'rewards_enabled',active.rewards_enabled,
  'effective_reward_epoch',active.effective_reward_epoch,
  'reward_basis_hash',active.reward_basis_hash,
  'reward_basis_doc',active.reward_basis_doc,
  'signing_key_doc',active.signing_key_doc,
  'reward_activated_at',active.reward_activated_at,
  'king_outcome',active.king_outcome,'king_hotkey',active.king_hotkey,
  'king_start_epoch',active.king_start_epoch,
  'promotion_required',active.promotion_required,
  'promotion_doc',active.promotion_doc,
  'baseline_promoted_at',active.baseline_promoted_at);

 SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(item) ORDER BY submission_id)
  INTO submissions_before FROM public.lab_arena_submissions item
  WHERE round_id='arena-2026-09-20';
 SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r)-'per_icp_score'-
   'qualification_doc' ORDER BY run_id) INTO execution_runs_before
  FROM public.lab_arena_runs r WHERE round_id='arena-2026-09-20'
   AND kind='execute';
 SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(l) ORDER BY entry_id)
  INTO execution_ledger_before FROM public.lab_arena_ledger l
  WHERE round_id='arena-2026-09-20' AND EXISTS(SELECT 1
   FROM public.lab_arena_runs r WHERE r.round_id='arena-2026-09-20'
    AND r.kind='execute' AND r.run_id=l.run_id);
 SELECT pg_catalog.jsonb_agg(item||pg_catalog.jsonb_build_object(
   'submission_id',(item->>'submission_id')||':r332archive') ORDER BY ordinal)
  INTO archive_participants FROM pg_catalog.jsonb_array_elements(active.participants)
   WITH ORDINALITY entries(item,ordinal);
 SELECT pg_catalog.jsonb_agg(pg_catalog.jsonb_build_object(
   'run_id',run_id,'per_icp_score',per_icp_score,
   'qualification_doc',qualification_doc,'updated_at',updated_at) ORDER BY run_id)
  INTO archived_execution_judgments FROM public.lab_arena_runs
  WHERE round_id='arena-2026-09-20' AND kind='execute';
 archive_config:=active.configuration_doc||pg_catalog.jsonb_build_object(
  'round_id','arena-2026-09-20-r332archive','mode','shadow','rewards_enabled',FALSE,
  'archived_terminal_status',active.status,
  'archived_terminal_cancel_reason',active.cancel_reason,
  'archived_terminal_round_sha256','418609475683ca81fb0f96cbb0c20d5243b805e5ecfffa875106f4dcab654072',
  'archived_terminal_baseline_sha256','8cb1fb888a97c2694369f129461bf492e3ce9e0f0a652da68e1d9ebcfeed5809',
  'archived_terminal_submissions_sha256','80147fc44b90f3e139820c27007a577c9d63f76eff5d4fcb02f1abc194e411a9',
  'archived_terminal_runs_sha256','482c43a1e1b85febec6d148667f1df65bdded118c6303443ccbda48f65e3fe1e',
  'archived_terminal_ledger_sha256','c5b4d28eaf2d11910de3ab316f38273caa004114696d40c9cd312e6efffe6f4e',
  'archived_terminal_company_judgments_sha256',
   '0ef317eb17e2318bae2ef9d4bc8de770cc8cbbae941ab9456941b2c44b9c5c41',
  'archived_terminal_judgment_cache_sha256','09bf9030e831972b02474dc75606b39516733417db98d9989a0e2cbf9391fde3',
  'archived_execution_artifacts_sha256','0892ee718698880c31476cb2bd87f052c62e92cd20af359a36b3145f3e4669ae',
  'archived_execution_artifact_count',98,
  'archived_authority_sha256','38ca5046aa2200f555130dec321282ecac92fdf5e0a1a95936222ee210022e4f',
  'archived_rewards_enabled',active.rewards_enabled,
  'archived_effective_reward_epoch',active.effective_reward_epoch,
  'archived_reward_basis_hash',active.reward_basis_hash,
  'archived_reward_basis_doc',active.reward_basis_doc,
  'archived_signing_key_doc',active.signing_key_doc,
  'archived_reward_activated_at',active.reward_activated_at,
  'archived_king_outcome',active.king_outcome,
  'archived_king_hotkey',active.king_hotkey,
  'archived_king_start_epoch',active.king_start_epoch,
  'archived_promotion_required',active.promotion_required,
  'archived_promotion_doc',active.promotion_doc,
  'archived_baseline_promoted_at',active.baseline_promoted_at,
  'archived_execution_judgments',archived_execution_judgments,
  'archived_bank_object_sha256',
   '7aaa7fbb4b4ce078dd62217c4d594dafdc82161d30c7718bd4500fa659fe5ad9');
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
  champion_fallback_providers)
 SELECT round_id,status,status_generation,stage_generation,configuration_doc,
  rewards_enabled,participants,benchmark_ref,evaluation_date,
  stage1_scoring_plan_doc,stage2_scoring_plan_doc,finalists,publication_doc,
  king_outcome,king_hotkey,king_start_epoch,effective_reward_epoch,
  reward_basis_hash,reward_basis_doc,signing_key_doc,reward_activated_at,
  cancel_reason,published_at,created_at,updated_at,promotion_required,
  promotion_doc,baseline_promoted_at,icp_set_date,confirmation_bank_ref,
  confirmation_bank_hash,confirmation_cohort,stage3_scoring_plan_doc,
  champion_funding_frozen,champion_submission_id,champion_hotkey,
  champion_fallback_providers FROM pg_catalog.jsonb_populate_record(
   NULL::public.lab_arena_rounds,pg_catalog.to_jsonb(active)||
   pg_catalog.jsonb_build_object(
    'round_id','arena-2026-09-20-r332archive','status','cancelled',
    'configuration_doc',archive_config,'participants',archive_participants,
    'rewards_enabled',FALSE,'reward_basis_hash',NULL,'reward_basis_doc',NULL,
    'signing_key_doc',NULL,'effective_reward_epoch',NULL,'reward_activated_at',NULL,
    'cancel_reason','authorized_sep20_authority_preserving_rejudge_archive'));
 INSERT INTO public.lab_arena_submissions SELECT (pg_catalog.jsonb_populate_record(
  NULL::public.lab_arena_submissions,pg_catalog.to_jsonb(s)||
  pg_catalog.jsonb_build_object('round_id','arena-2026-09-20-r332archive',
   'submission_id',s.submission_id||':r332archive'))).*
 FROM public.lab_arena_submissions s WHERE s.round_id='arena-2026-09-20';

 UPDATE public.lab_arena_runs SET round_id='arena-2026-09-20-r332archive',
  submission_id=submission_id||':r332archive'
 WHERE round_id='arena-2026-09-20' AND kind='score';
 GET DIAGNOSTICS moved_score_runs=ROW_COUNT;
 UPDATE public.lab_arena_ledger l SET round_id='arena-2026-09-20-r332archive',
  submission_id=submission_id||':r332archive'
 WHERE round_id='arena-2026-09-20' AND EXISTS(SELECT 1
  FROM public.lab_arena_runs s WHERE s.round_id='arena-2026-09-20-r332archive'
   AND s.kind='score' AND s.run_id=l.run_id);
 GET DIAGNOSTICS moved_score_ledger=ROW_COUNT;
 UPDATE public.lab_arena_runs SET per_icp_score=NULL,qualification_doc=NULL
 WHERE round_id='arena-2026-09-20' AND kind='execute';
 UPDATE public.lab_arena_rounds SET status='stage1',
  status_generation=status_generation+1,stage_generation=stage_generation+1,
  configuration_doc=configuration_doc||pg_catalog.jsonb_build_object(
   'schedule','{"benchmark_deadline":"2026-09-20T10:30:00Z","final_scoring_close":"2026-09-21T00:30:00Z","publication_deadline":"2026-09-21T00:30:01Z","stage_1_close":"2026-09-20T17:30:00Z","stage_1_scoring_close":"2026-09-20T20:30:00Z","stage_1_start":"2026-09-20T10:30:01Z","stage_2_close":"2026-09-20T21:30:00Z","stage_2_start":"2026-09-20T20:30:01Z","submission_cutoff":"2026-09-20T00:00:00Z","submission_open":"2026-09-19T00:00:00Z"}'::JSONB,
   'scorer_image_digest',
    'sha256:cc2dd55dec0b94db8cd0e184d9bf67014e2e83def22d5c5b4e28213c46fe0cd8',
   'scorer_image_reference',
    '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:cc2dd55dec0b94db8cd0e184d9bf67014e2e83def22d5c5b4e28213c46fe0cd8'),
  stage1_scoring_plan_doc=NULL,
  stage2_scoring_plan_doc=NULL,stage3_scoring_plan_doc=NULL,finalists=NULL,
  publication_doc=NULL,published_at=NULL,cancel_reason=NULL,
  updated_at=pg_catalog.clock_timestamp()
 WHERE round_id='arena-2026-09-20';
 IF NOT FOUND THEN RAISE EXCEPTION 'Sep20 rerun332 round update missing'; END IF;
 ALTER TABLE public.lab_arena_ledger ENABLE TRIGGER USER;
 ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER;
 ALTER TABLE public.lab_arena_submissions ENABLE TRIGGER USER;
 ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER;

 SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(item) ORDER BY submission_id)
  INTO submissions_after FROM public.lab_arena_submissions item
  WHERE round_id='arena-2026-09-20';
 SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r)-'per_icp_score'-
   'qualification_doc' ORDER BY run_id) INTO execution_runs_after
  FROM public.lab_arena_runs r WHERE round_id='arena-2026-09-20'
   AND kind='execute';
 SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(l) ORDER BY entry_id)
  INTO execution_ledger_after FROM public.lab_arena_ledger l
  WHERE round_id='arena-2026-09-20' AND EXISTS(SELECT 1
   FROM public.lab_arena_runs r WHERE r.round_id='arena-2026-09-20'
    AND r.kind='execute' AND r.run_id=l.run_id);
 SELECT pg_catalog.jsonb_build_object(
  'rounds',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r) ORDER BY round_id)
    FROM public.lab_arena_rounds r WHERE round_id IN
     ('arena-2026-09-20-r326archive','arena-2026-09-20-r328archive')),
  'submissions',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(s)
    ORDER BY round_id,submission_id) FROM public.lab_arena_submissions s
    WHERE round_id IN
     ('arena-2026-09-20-r326archive','arena-2026-09-20-r328archive')),
  'runs',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r)
    ORDER BY round_id,run_id) FROM public.lab_arena_runs r WHERE round_id IN
     ('arena-2026-09-20-r326archive','arena-2026-09-20-r328archive')),
  'ledger',(SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(l)
    ORDER BY round_id,entry_id) FROM public.lab_arena_ledger l WHERE round_id IN
     ('arena-2026-09-20-r326archive','arena-2026-09-20-r328archive')))
 INTO prior_archive_after;
 SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256',
   pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(
    pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(j)::TEXT,'sha256'),'hex'),''
    ORDER BY cache_key,authority_slot),''),'sha256'),'hex'))
  INTO judgments_after FROM public.lab_arena_company_judgments j;
 SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256',
   pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(
    pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(c)::TEXT,'sha256'),'hex'),''
    ORDER BY cache_key),''),'sha256'),'hex'))
  INTO cache_after FROM public.lab_arena_judgment_cache c;
 SELECT pg_catalog.jsonb_build_object(
  'rewards_enabled',rewards_enabled,
  'effective_reward_epoch',effective_reward_epoch,
  'reward_basis_hash',reward_basis_hash,
  'reward_basis_doc',reward_basis_doc,
  'signing_key_doc',signing_key_doc,
  'reward_activated_at',reward_activated_at,
  'king_outcome',king_outcome,'king_hotkey',king_hotkey,
  'king_start_epoch',king_start_epoch,
  'promotion_required',promotion_required,
  'promotion_doc',promotion_doc,
  'baseline_promoted_at',baseline_promoted_at)
 INTO authority_after FROM public.lab_arena_rounds
 WHERE round_id='arena-2026-09-20';
 IF moved_score_runs<>98
  OR moved_score_ledger<>1866
  OR submissions_after IS DISTINCT FROM submissions_before
  OR execution_runs_after IS DISTINCT FROM execution_runs_before
  OR execution_ledger_after IS DISTINCT FROM execution_ledger_before
  OR prior_archive_after IS DISTINCT FROM prior_archive_before
  OR judgments_after IS DISTINCT FROM judgments_before
  OR cache_after IS DISTINCT FROM cache_before
  OR authority_after IS DISTINCT FROM authority_before
  OR pg_catalog.encode(extensions.digest(authority_after::TEXT,'sha256'),'hex')<>
     '38ca5046aa2200f555130dec321282ecac92fdf5e0a1a95936222ee210022e4f'
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-20' AND kind='score')
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-20-r332archive' AND kind='execute')
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id='arena-2026-09-20' AND kind='execute'
       AND status='accepted' AND terminal_cause='accepted')<>
     98 THEN
  RAISE EXCEPTION 'Sep20 rerun332 preservation differs' USING ERRCODE='55000';
 END IF;
END $sep20_rerun332$;
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
COMMIT;
