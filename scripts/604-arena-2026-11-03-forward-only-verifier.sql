-- November 3 operator exception: preserve completed judgments, then use the
-- improved verifier only for unscored ICPs. This is not a uniform regrade.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
CREATE TEMP TABLE arena604_schema_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $helpers_preflight604$
DECLARE t RECORD; p RECORD;
BEGIN
 FOR t IN SELECT * FROM (VALUES
 ('public.lab_arena__nov03_preserved_judge_v1(public.lab_arena_rounds,public.lab_arena_runs)','d714778b53f14917052cdc16c7cab9e3f88413f7c0d2b6e9f9f727c71f4d823a','plpgsql',FALSE),
 ('public.lab_arena_preserved_judgment_policy_v1(text,text,text,jsonb)','839f8a86cd675995a4fa21e27c7b29bb8204a061d5fc27b807fc701cba7b3d34','sql',TRUE)
 ) AS helpers(signature,source_hash,language,api) LOOP
  SELECT proc.*,l.lanname INTO p FROM pg_proc proc JOIN pg_language l ON l.oid=proc.prolang
  WHERE proc.oid=to_regprocedure(t.signature);
  IF FOUND AND (encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex')<>t.source_hash
   OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR NOT p.prosecdef OR p.provolatile<>'s'
   OR p.proacl::TEXT IS DISTINCT FROM CASE WHEN t.api
    THEN '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
    ELSE '{lab_arena_owner=X/lab_arena_owner}' END
   OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] OR p.lanname<>t.language) THEN
   RAISE EXCEPTION 'Nov03 forward verifier helper differs: %',t.signature; END IF;
 END LOOP;
END;
$helpers_preflight604$;

-- Both readers fail closed unless the exact owner-created receipt, accepted
-- judgment, source and frozen old/new configurations still match.
CREATE OR REPLACE FUNCTION public.lab_arena__nov03_preserved_judge_v1(
 p_round public.lab_arena_rounds,p_judge public.lab_arena_runs)
RETURNS JSONB LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $preserved604$
DECLARE h public.lab_arena_scoring_supersessions; proof JSONB;
 source public.lab_arena_runs; old_round public.lab_arena_rounds;
BEGIN
 IF p_round.round_id IS DISTINCT FROM 'arena-2026-11-03'
  OR p_round.scoring_generation_floor IS DISTINCT FROM 0
  OR p_judge.round_id IS DISTINCT FROM p_round.round_id
  OR p_judge.kind IS DISTINCT FROM 'score' OR p_judge.status IS DISTINCT FROM 'accepted'
  OR p_judge.terminal_cause IS DISTINCT FROM 'accepted' OR p_judge.output_ref IS NULL
  OR p_judge.scoring_generation_floor IS DISTINCT FROM 0 THEN RETURN NULL; END IF;
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions
 WHERE round_id=p_round.round_id AND request_id='nov03-forward-only-verifier604-v1'
  AND recovery_doc->>'operation'='nov03_forward_only_verifier_v1'
  AND recovery_doc->'scoring_generation_floor'='0'::JSONB
  AND new_round_doc->'configuration_doc'=p_round.configuration_doc;
 IF NOT FOUND THEN RETURN NULL; END IF;
 proof:=h.recovery_doc->'preserved_judgments'->p_judge.run_id;
 IF proof IS NULL OR proof->>'judge_sha256' IS DISTINCT FROM
   encode(extensions.digest(to_jsonb(p_judge)::TEXT,'sha256'),'hex')
  OR NOT EXISTS(SELECT 1 FROM public.lab_arena_runs actual
   WHERE actual.run_id=p_judge.run_id AND to_jsonb(actual)=to_jsonb(p_judge)) THEN RETURN NULL; END IF;
 SELECT * INTO source FROM public.lab_arena_runs WHERE run_id=proof->>'source_run_id';
 IF NOT FOUND OR source.run_id IS DISTINCT FROM p_judge.scored_run_id
  OR source.kind IS DISTINCT FROM 'execute' OR source.status IS DISTINCT FROM 'accepted'
  OR source.terminal_cause IS DISTINCT FROM 'accepted' OR source.output_ref IS NULL
  OR (source.round_id,source.submission_id,source.miner_hotkey,source.stage,source.icp_position)
    IS DISTINCT FROM (p_judge.round_id,p_judge.submission_id,p_judge.miner_hotkey,p_judge.stage,p_judge.icp_position)
  OR proof->>'source_immutable_sha256' IS DISTINCT FROM
    encode(extensions.digest((to_jsonb(source)-ARRAY['per_icp_score','qualification_doc','updated_at'])::TEXT,'sha256'),'hex') THEN RETURN NULL; END IF;
 old_round:=jsonb_populate_record(NULL::public.lab_arena_rounds,h.old_round_doc);
 IF NOT public.lab_arena__company_judgment_refs_v1(old_round,p_judge.company_judgment_refs)
  OR p_judge.judgment_cache_key IS NOT NULL OR p_judge.judgment_input_hash IS NOT NULL
  OR p_judge.judgment_scope_doc IS NOT NULL THEN RETURN NULL; END IF;
 RETURN jsonb_build_object('operation','nov03_forward_only_verifier_v1','scoring_generation_floor',0,
  'scorer_policy',old_round.configuration_doc->'scorer_policy',
  'scorer_image_digest',old_round.configuration_doc->'scorer_image_digest',
  'scorer_image_reference',old_round.configuration_doc->'scorer_image_reference',
  'current_scorer_image_digest',p_round.configuration_doc->'scorer_image_digest',
  'current_scorer_image_reference',p_round.configuration_doc->'scorer_image_reference');
END;
$preserved604$;
ALTER FUNCTION public.lab_arena__nov03_preserved_judge_v1(public.lab_arena_rounds,public.lab_arena_runs) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__nov03_preserved_judge_v1(public.lab_arena_rounds,public.lab_arena_runs)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_preserved_judgment_policy_v1(
 p_round_id TEXT,p_judge_run_id TEXT,p_source_run_id TEXT,p_refs JSONB)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $reader604$
 SELECT public.lab_arena__nov03_preserved_judge_v1(r,j)
 FROM public.lab_arena_rounds r JOIN public.lab_arena_runs j ON j.round_id=r.round_id
 WHERE r.round_id=p_round_id AND j.run_id=p_judge_run_id
  AND j.scored_run_id=p_source_run_id AND j.company_judgment_refs=p_refs;
$reader604$;
ALTER FUNCTION public.lab_arena_preserved_judgment_policy_v1(TEXT,TEXT,TEXT,JSONB) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_preserved_judgment_policy_v1(TEXT,TEXT,TEXT,JSONB)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
GRANT EXECUTE ON FUNCTION public.lab_arena_preserved_judgment_policy_v1(TEXT,TEXT,TEXT,JSONB) TO lab_arena_service;

DO $helpers_verify604$
DECLARE t RECORD; p RECORD;
BEGIN
 FOR t IN SELECT * FROM (VALUES
 ('public.lab_arena__nov03_preserved_judge_v1(public.lab_arena_rounds,public.lab_arena_runs)','d714778b53f14917052cdc16c7cab9e3f88413f7c0d2b6e9f9f727c71f4d823a','plpgsql',FALSE),
 ('public.lab_arena_preserved_judgment_policy_v1(text,text,text,jsonb)','839f8a86cd675995a4fa21e27c7b29bb8204a061d5fc27b807fc701cba7b3d34','sql',TRUE)
 ) AS helpers(signature,source_hash,language,api) LOOP
  SELECT proc.*,l.lanname INTO p FROM pg_proc proc JOIN pg_language l ON l.oid=proc.prolang
  WHERE proc.oid=to_regprocedure(t.signature);
  IF FOUND AND (encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex')<>t.source_hash
   OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR NOT p.prosecdef OR p.provolatile<>'s'
   OR p.proacl::TEXT IS DISTINCT FROM CASE WHEN t.api
    THEN '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
    ELSE '{lab_arena_owner=X/lab_arena_owner}' END
   OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] OR p.lanname<>t.language) THEN
   RAISE EXCEPTION 'Nov03 forward verifier helper differs: %',t.signature; END IF;
 END LOOP;
END;
$helpers_verify604$;

DO $patch604$
DECLARE target RECORD; d TEXT; updated TEXT; metadata JSONB; function_oid OID;
BEGIN
 FOR target IN SELECT * FROM (VALUES
 ('public.lab_arena__all_judgments_ready_v1(text)','68a8126d0729337cd2e12adc8bb4be0b7cfd1eadc8ca9935032bf869fd7bd33e','90e28205344403f08f4bb6d9b58f7500d2c80ff2c148e9c378bba1c8a165a6e0',1,$old604$public.lab_arena__company_judgment_refs_v1(validation.round_row,judge.company_judgment_refs)$old604$,$new604$(public.lab_arena__company_judgment_refs_v1(validation.round_row,judge.company_judgment_refs) OR public.lab_arena__nov03_preserved_judge_v1(validation.round_row,judge) IS NOT NULL)$new604$),
 ('public.lab_arena__participant_incomplete_reason_v1(text,text)','f083615eea5f2b3e440ce7bed39ef7111b1ef5fe242bb685a79febaab0ca757a','6233456df0e4585839148f792f6b24eea33cc298fb708b0b5cc01668be224d2b',1,$old604$public.lab_arena__company_judgment_refs_v1(r,judge.company_judgment_refs)$old604$,$new604$(public.lab_arena__company_judgment_refs_v1(r,judge.company_judgment_refs) OR public.lab_arena__nov03_preserved_judge_v1(r,judge) IS NOT NULL)$new604$),
 ('public.lab_arena_record_run_scores(text,smallint,jsonb)','759119aafc1f0041da8faa38eb4066abc0d10206e7192956f9741ef989e94834','4b30dc9e41852938c66bb180d2b3a046dbb237ca75c5cececbb648901f37ea08',2,$old604$public.lab_arena__company_judgment_refs_v1(v_round,judged.company_judgment_refs)$old604$,$new604$(public.lab_arena__company_judgment_refs_v1(v_round,judged.company_judgment_refs) OR public.lab_arena__nov03_preserved_judge_v1(v_round,judged) IS NOT NULL)$new604$),
 ('public.lab_arena_scoring_policy_history_v1(text)','94b3338f73b2cdc5803fc9caa4c2d33bfb22a0d30cf0bcf6b9e0347b2c181b50','5d5c46fd06f1d3db37c5fadb0dea6ad13f25d0a3090fd2ead798e9d683c5b098',1,$old604$'generation_floor',h.generation_floor,$old604$,$new604$'generation_floor',CASE WHEN h.recovery_doc->>'operation'='nov03_forward_only_verifier_v1' THEN (h.new_round_doc->>'scoring_generation_floor')::BIGINT ELSE h.generation_floor END,
  'receipt_generation',h.generation_floor,$new604$)
 ) AS patches(signature,preimage,postimage,occurrences,old_text,new_text) LOOP
  function_oid:=to_regprocedure(target.signature);
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,metadata FROM pg_proc p WHERE p.oid=function_oid;
  IF d IS NULL OR (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,
   p.prosecdef,p.provolatile,p.proconfig,l.lanname) FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=function_oid)
   IS DISTINCT FROM jsonb_build_array('lab_arena_owner',
    CASE WHEN target.signature IN('public.lab_arena_record_run_scores(text,smallint,jsonb)','public.lab_arena_scoring_policy_history_v1(text)')
     THEN '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
     ELSE '{lab_arena_owner=X/lab_arena_owner}' END,TRUE,
    CASE WHEN target.signature='public.lab_arena_record_run_scores(text,smallint,jsonb)' THEN 'v' ELSE 's' END,
    ARRAY['search_path=pg_catalog, public']::TEXT[],
    CASE WHEN target.signature IN('public.lab_arena__all_judgments_ready_v1(text)','public.lab_arena_scoring_policy_history_v1(text)') THEN 'sql' ELSE 'plpgsql' END) THEN
   RAISE EXCEPTION 'Nov03 forward verifier security differs: %',target.signature; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')=target.postimage THEN CONTINUE; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')<>target.preimage
   OR length(d)-length(replace(d,target.old_text,''))<>target.occurrences*length(target.old_text) THEN
   RAISE EXCEPTION 'Nov03 forward verifier preimage differs: %',target.signature; END IF;
  updated:=replace(d,target.old_text,target.new_text);
  IF encode(extensions.digest(updated,'sha256'),'hex')<>target.postimage THEN
   RAISE EXCEPTION 'Nov03 forward verifier postimage differs: %',target.signature; END IF;
  EXECUTE updated;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=function_oid) IS DISTINCT FROM metadata
   OR encode(extensions.digest(pg_get_functiondef(function_oid),'sha256'),'hex')<>target.postimage THEN
   RAISE EXCEPTION 'Nov03 forward verifier readback differs: %',target.signature; END IF;
 END LOOP;
END;
$patch604$;
-- Exact one-time live preimage604. The catalog, schedule, roster, plans,
-- generations, source bytes, accepted judge rows, failures and costs do not change.
DO $nov03_forward604$
DECLARE rid CONSTANT TEXT:='arena-2026-11-03';
 request CONSTANT TEXT:='nov03-forward-only-verifier604-v1';
 manifest CONSTANT JSONB:=$manifest604${"preserved_judgments":{"arena-2026-11-03:baseline-2026-11-03:1:0:score:2":{"judge_sha256":"6b565a7636a2e2172b88eed86e11d2031c7806940cb621539d8556d5d6246905","source_immutable_sha256":"3c4e132c53b151000695624e05b01c24b91bbb01f01a58ae778da259bbff2ba6","source_original_sha256":"f0f56f5a5549a8870bdb3524d3edf7da8a7b5d2436ef8c2b3aed287b8d59ded1","source_run_id":"arena-2026-11-03:baseline-2026-11-03:1:0:1"},"arena-2026-11-03:baseline-2026-11-03:1:1:score:1":{"judge_sha256":"5fc70fd0a20cd9c02306d2e22aab4cef84752b4725ada1a3ba883be69513b1ba","source_immutable_sha256":"bc6795e4187c25e0811983ff7cf3dd18a3b490cb63c4105e0d57ec3d495b1c8d","source_original_sha256":"21e6cbfabeaafd0a9d00765ba2e183112262be291061a17d591da57f04c3c43c","source_run_id":"arena-2026-11-03:baseline-2026-11-03:1:1:1"},"arena-2026-11-03:baseline-2026-11-03:1:2:score:3":{"judge_sha256":"a766700350a6027856843abfd7a94e9da55143431ee23d0f36b9be3670aa4bb9","source_immutable_sha256":"8c3f8ba2c94d3ee2a6afbca33e8ce6130cff7915e1ea1b2ec8b0b3c638097f36","source_original_sha256":"c96d7f20a16f06a7ac28963fda7b00fdec975db21aceb14bca72e54e049e86a3","source_run_id":"arena-2026-11-03:baseline-2026-11-03:1:2:1"},"arena-2026-11-03:baseline-2026-11-03:1:3:score:3":{"judge_sha256":"11b990496e0a1cb89a9d69848e6bae8d11e51f3a73b51064dedf64c018d4b3ed","source_immutable_sha256":"376656a161c9fdebb4a922f3f150cc5fc80c085621ee27e1c97f7896fa762985","source_original_sha256":"17ea1a2746442aa001288c246663033f3764c8ea2cf2439bb498f5c2e588c721","source_run_id":"arena-2026-11-03:baseline-2026-11-03:1:3:1"},"arena-2026-11-03:baseline-2026-11-03:1:4:score:2":{"judge_sha256":"858b87a9d5ed6c79f1d6692bffd65ca8edfb25cd6253e157d79a1b464709197b","source_immutable_sha256":"4a8defea2ca26ea92b2600e7ee06a52d9b5eee5cc1b0dcd56eb3b3241b31d5eb","source_original_sha256":"17574ce56bde3ded7e1cd9e66b27cf3d4542b6daf354b7a29a8b76811a4d6ebe","source_run_id":"arena-2026-11-03:baseline-2026-11-03:1:4:1"},"arena-2026-11-03:baseline-2026-11-03:1:5:score:1":{"judge_sha256":"ebd01b18ad7e1f6a170e9f3db680cf5e837449f851ec88150f17981007783631","source_immutable_sha256":"9710a7d7129d435a64f81866286019bf5cde84767f7779e24e1db4899ea6374f","source_original_sha256":"fdd240ef90b2144a19642368fe8b34bd7f5df2012b88ad09754a1eaac8fba709","source_run_id":"arena-2026-11-03:baseline-2026-11-03:1:5:1"},"arena-2026-11-03:baseline-2026-11-03:1:6:score:2":{"judge_sha256":"9de653d48b31fb34e5d3283537ba64fec29336280c3de5740fc26e1e1e224a69","source_immutable_sha256":"4ea9888c92575b679f7e549ac21b2094c0b7f7e3cf91bc089a7e0b94f1b18b6a","source_original_sha256":"d814095362bbed7e646f13e19b30f79e6c473bf7bc01c29d8564749373a89214","source_run_id":"arena-2026-11-03:baseline-2026-11-03:1:6:1"},"arena-2026-11-03:baseline-2026-11-03:1:7:score:2":{"judge_sha256":"40fa3a56db53cba061597bc812d4c4a026b093233170e78a8a1d884dc3962afe","source_immutable_sha256":"ae6c7baf5df52ce2b98a1a84daa70fe7f678709b52e7b078c5e1b5f03f2713b7","source_original_sha256":"1896759b0a704caa7a4e0e23aa31f6ba8da0e4237f584909247e8f1d7b972c21","source_run_id":"arena-2026-11-03:baseline-2026-11-03:1:7:1"},"arena-2026-11-03:baseline-2026-11-03:1:9:score:1":{"judge_sha256":"7b16f7a8de4d02d65bc3f4b404ac139a181577f608468ac37e279d105060d4ed","source_immutable_sha256":"e76434fd686bce4b1b5e33e63e1ee057e57113f0a31f6814eced6fe133feca4c","source_original_sha256":"bdea728072e596d4aa4a890f701dcc9e790b66fc05fa8543c764ca071aedbfe5","source_run_id":"arena-2026-11-03:baseline-2026-11-03:1:9:1"}},"run_hashes":{"arena-2026-11-03:baseline-2026-11-03:1:0:1":"f0f56f5a5549a8870bdb3524d3edf7da8a7b5d2436ef8c2b3aed287b8d59ded1","arena-2026-11-03:baseline-2026-11-03:1:0:score:1":"1dd878b85c97182a3d01ca169d299633fb60ddb97990b5bb7a0d31c79cff2d16","arena-2026-11-03:baseline-2026-11-03:1:0:score:2":"6b565a7636a2e2172b88eed86e11d2031c7806940cb621539d8556d5d6246905","arena-2026-11-03:baseline-2026-11-03:1:1:1":"21e6cbfabeaafd0a9d00765ba2e183112262be291061a17d591da57f04c3c43c","arena-2026-11-03:baseline-2026-11-03:1:1:score:1":"5fc70fd0a20cd9c02306d2e22aab4cef84752b4725ada1a3ba883be69513b1ba","arena-2026-11-03:baseline-2026-11-03:1:2:1":"c96d7f20a16f06a7ac28963fda7b00fdec975db21aceb14bca72e54e049e86a3","arena-2026-11-03:baseline-2026-11-03:1:2:score:1":"8589c1bbaeb3516aa00e424d1b0eb9ad9fc258ad353e2472b8893ce6f2ad9df5","arena-2026-11-03:baseline-2026-11-03:1:2:score:2":"e44e762f3cf46394f02156e9769903883b27e39f735aa8dcc8374bf297edd8fb","arena-2026-11-03:baseline-2026-11-03:1:2:score:3":"a766700350a6027856843abfd7a94e9da55143431ee23d0f36b9be3670aa4bb9","arena-2026-11-03:baseline-2026-11-03:1:3:1":"17ea1a2746442aa001288c246663033f3764c8ea2cf2439bb498f5c2e588c721","arena-2026-11-03:baseline-2026-11-03:1:3:score:1":"d9e57873bc1f2b82aac57b3f01e01676d3455f792cc7e9dd8e47f247aed87813","arena-2026-11-03:baseline-2026-11-03:1:3:score:2":"c580c22a5cbe99c4c6f17eea5f180d737d1ead4a915f3ee41e475b7945dfecb8","arena-2026-11-03:baseline-2026-11-03:1:3:score:3":"11b990496e0a1cb89a9d69848e6bae8d11e51f3a73b51064dedf64c018d4b3ed","arena-2026-11-03:baseline-2026-11-03:1:4:1":"17574ce56bde3ded7e1cd9e66b27cf3d4542b6daf354b7a29a8b76811a4d6ebe","arena-2026-11-03:baseline-2026-11-03:1:4:score:1":"0a88fb9660872f94f3e5239dceaba9e85ad7528fa2c13425d808338f601c25d6","arena-2026-11-03:baseline-2026-11-03:1:4:score:2":"858b87a9d5ed6c79f1d6692bffd65ca8edfb25cd6253e157d79a1b464709197b","arena-2026-11-03:baseline-2026-11-03:1:5:1":"fdd240ef90b2144a19642368fe8b34bd7f5df2012b88ad09754a1eaac8fba709","arena-2026-11-03:baseline-2026-11-03:1:5:score:1":"ebd01b18ad7e1f6a170e9f3db680cf5e837449f851ec88150f17981007783631","arena-2026-11-03:baseline-2026-11-03:1:6:1":"d814095362bbed7e646f13e19b30f79e6c473bf7bc01c29d8564749373a89214","arena-2026-11-03:baseline-2026-11-03:1:6:score:1":"78ee09aa42ca3a6ccd30f1e12013da875ff450ad860fbabd65a6cfa7a9d0da5f","arena-2026-11-03:baseline-2026-11-03:1:6:score:2":"9de653d48b31fb34e5d3283537ba64fec29336280c3de5740fc26e1e1e224a69","arena-2026-11-03:baseline-2026-11-03:1:7:1":"1896759b0a704caa7a4e0e23aa31f6ba8da0e4237f584909247e8f1d7b972c21","arena-2026-11-03:baseline-2026-11-03:1:7:score:1":"a94c964d63195911dfcdc985c59c71247570a04425cdbee72bc1b6cb27ac56b3","arena-2026-11-03:baseline-2026-11-03:1:7:score:2":"40fa3a56db53cba061597bc812d4c4a026b093233170e78a8a1d884dc3962afe","arena-2026-11-03:baseline-2026-11-03:1:8:1":"7fa1f2a24b3276aefec06ea61bdbe782a1a7167ec9ba06f30dedd6379b453f64","arena-2026-11-03:baseline-2026-11-03:1:8:score:1":"ee17bb654544bc05143c6ba9fa1ec070711502edbd683a715876a029c4495cda","arena-2026-11-03:baseline-2026-11-03:1:8:score:2":"63c13e3902a7a3dcb4baf52376ac13b2e2080d1ed853a96f624255105682f376","arena-2026-11-03:baseline-2026-11-03:1:8:score:3":"91661ea7c81e8ad12cda9351faf6f960ae54d46301e8d970217bba2fea29aec0","arena-2026-11-03:baseline-2026-11-03:1:8:score:4":"c44d420a7393c251b23b6cb552f8344632607a2610ea90c1021986aeff4bd9c5","arena-2026-11-03:baseline-2026-11-03:1:8:score:5":"bcc7a37a15147fe70f5b8bf6ed33a216b601de5a63678ca76ae6434d43b3e031","arena-2026-11-03:baseline-2026-11-03:1:9:1":"bdea728072e596d4aa4a890f701dcc9e790b66fc05fa8543c764ca071aedbfe5","arena-2026-11-03:baseline-2026-11-03:1:9:score:1":"7b16f7a8de4d02d65bc3f4b404ac139a181577f608468ac37e279d105060d4ed","arena-2026-11-03:sub-0df2d0295c950d907e094e258bf98905:2:0:1":"40e350b0065057ae965233066232f55ae25d72e57855c32d3b6df1ca112f3c83","arena-2026-11-03:sub-0df2d0295c950d907e094e258bf98905:2:1:1":"754fe57f97f5eab5d5de63621fd46a679d4b3121542b00813b0e63371464e4c6","arena-2026-11-03:sub-0df2d0295c950d907e094e258bf98905:2:2:1":"f9931dd9cf792ddc36d7dfba03b44f5371631ae715aa82c0e5e03b93eea3fd90","arena-2026-11-03:sub-0df2d0295c950d907e094e258bf98905:2:3:1":"f3c065659cad58b7d1024cd7f4e665da983ae2d410de4a11d583b69a0784f398","arena-2026-11-03:sub-0df2d0295c950d907e094e258bf98905:2:4:1":"6b2845a10a20ecec26e11f47c1a5d8fd8c37b1a311e54f26dae9208063730921","arena-2026-11-03:sub-0df2d0295c950d907e094e258bf98905:2:5:1":"33061f810fb41d92f3d0e8337abf40f1e403711f18c027a467cf52717d958140","arena-2026-11-03:sub-0df2d0295c950d907e094e258bf98905:2:6:1":"55a08f4c2c206c4ea63b39fe4423ab9d259792c6e8f3513d42194f97f865ebbb","arena-2026-11-03:sub-0df2d0295c950d907e094e258bf98905:2:7:1":"8c38e00f051d235ce417b600c56b04706569d7a092f2c6f5da9d9d727c539a91","arena-2026-11-03:sub-0df2d0295c950d907e094e258bf98905:2:8:1":"0e041fbaae7607a70e17303159a858de02f96e60de8dd44c34bf3074ee109e0f","arena-2026-11-03:sub-0df2d0295c950d907e094e258bf98905:2:9:1":"ceaa37cd96cdfc1a8855461f95bcdec64734c86c0c55344972ccfe9779effa28","arena-2026-11-03:sub-6206272caf2b1a19ee2a7e843dc518cf:2:0:1":"e5e7870891b464c61377e28695b6c7efed9eeae1d9899dfbf00c7730d98e37d5","arena-2026-11-03:sub-6206272caf2b1a19ee2a7e843dc518cf:2:1:1":"0d5915903fb1e015f32be8b306bbd64bc97785b3bd4bd88006e33bc6affd1aaa","arena-2026-11-03:sub-6206272caf2b1a19ee2a7e843dc518cf:2:2:1":"ff68aca879e1b44f81509cb4b77d6b3c1a90dc093c96b9b5ee4c79c52407be9e","arena-2026-11-03:sub-6206272caf2b1a19ee2a7e843dc518cf:2:3:1":"2b30f712c0584a2f3e4372209763803020ca4a4691f18dcf4357bb1b90fdfc2c","arena-2026-11-03:sub-6206272caf2b1a19ee2a7e843dc518cf:2:4:1":"6577a5fdb8eb911fab3153f074eafe24d7cc2af2eef0b4c961d2b424f46307ed","arena-2026-11-03:sub-6206272caf2b1a19ee2a7e843dc518cf:2:5:1":"d2b7a4c57442b5a4f355b474db8b857a233f0e77c5f0466f8ca004d328016d8b","arena-2026-11-03:sub-6206272caf2b1a19ee2a7e843dc518cf:2:6:1":"af745faf084568bd900bf87c24813df38c76599d19c0f04f95c2f67cec862681","arena-2026-11-03:sub-6206272caf2b1a19ee2a7e843dc518cf:2:7:1":"d7dcf6324ef5b9b5da31cff3f9a69579e6890e97af47ba68bcf7c2a62d3905fb","arena-2026-11-03:sub-6206272caf2b1a19ee2a7e843dc518cf:2:8:1":"bf5107bb7f1c5d1f34e35456d55326ddc043729366aaefaf27e0743f52c16758","arena-2026-11-03:sub-6206272caf2b1a19ee2a7e843dc518cf:2:9:1":"d6f070eb8d93090a377cffda3f5fc7f6770344a6e76f107269c155c947262a42","arena-2026-11-03:sub-62e7cca860ed3b5db98ba8d7b2d4cb37:2:0:1":"26ea12b6fa52aac98e93f5485cbf0a862c1891ad61eb4659d8186da7d105b735","arena-2026-11-03:sub-62e7cca860ed3b5db98ba8d7b2d4cb37:2:1:1":"96bb2efca42fc5dc82a2e884c745f51fc3a4b100015028d8f601bce43abdb60c","arena-2026-11-03:sub-62e7cca860ed3b5db98ba8d7b2d4cb37:2:2:1":"9716cd40ca1c4f26ae001b949721e4f987da7759b5a4891e4620e5abf8c89ec8","arena-2026-11-03:sub-62e7cca860ed3b5db98ba8d7b2d4cb37:2:3:1":"a409a99810600361e57cd653a08a567269972ebaebc240a550d3f97bf0091d02","arena-2026-11-03:sub-62e7cca860ed3b5db98ba8d7b2d4cb37:2:4:1":"dac93d50baca80a9f9f7cf3a897b6f695e75d5e3210189f07462da9b11b1e3ff","arena-2026-11-03:sub-62e7cca860ed3b5db98ba8d7b2d4cb37:2:5:1":"66add099c267bf8bb3c474cd5b3121d3a2b7507018781cbfe4befd48e62b1904","arena-2026-11-03:sub-62e7cca860ed3b5db98ba8d7b2d4cb37:2:6:1":"124dca3d49f7c704029fa070e7d1fe7925ae8c057b6376a32e809208cd889401","arena-2026-11-03:sub-62e7cca860ed3b5db98ba8d7b2d4cb37:2:7:1":"c820a32f700bbdd8ae39e614601892fbd654803ffca1136d273acd9bc30702ef","arena-2026-11-03:sub-62e7cca860ed3b5db98ba8d7b2d4cb37:2:8:1":"91b55f8bbd64cbe91b6653788afa6aa1e3cd7cb363e58322e8152031b4b25707","arena-2026-11-03:sub-62e7cca860ed3b5db98ba8d7b2d4cb37:2:9:1":"2d985a85c078ce0841ebb23b46c5d32e2d6f92c343225407634baa8b74c49e12","arena-2026-11-03:sub-862332b7526026ae04645471c18a425c:2:0:1":"52d1aeb2c555b6f45142769a88c84584c2cd1165de987f20c43f42e6a30bac32","arena-2026-11-03:sub-862332b7526026ae04645471c18a425c:2:1:1":"c9e708ab67d3fe3f4cb01cb32a9df5f0c07bb205ff90281f0781dca3725dde1d","arena-2026-11-03:sub-862332b7526026ae04645471c18a425c:2:1:2":"ba00e73576d130ed80d55b40819391a45d67f6ac56890cfb0d6ccadb226c7336","arena-2026-11-03:sub-862332b7526026ae04645471c18a425c:2:2:1":"3a5be2a9a5c4b7d829c9f4babed84755b8676fccf04e268335386e5800708d91","arena-2026-11-03:sub-862332b7526026ae04645471c18a425c:2:2:2":"040e217c950b2349df7f90f9011b9cae7a3896808df26a247a434deb32aceb05","arena-2026-11-03:sub-862332b7526026ae04645471c18a425c:2:3:1":"75a05dfdff3481888c966dce213f5f90b7c368b9fdb339705c2aa2f36e5e0dd7","arena-2026-11-03:sub-862332b7526026ae04645471c18a425c:2:3:2":"e8358aeee9b0f36569d4422137b5532a601cb897190f353585a21dfc6c526716","arena-2026-11-03:sub-862332b7526026ae04645471c18a425c:2:4:1":"5501a4581f31c3d40620e1b172b5ee492f4734e4422afae8e3d80b97dcacb13e","arena-2026-11-03:sub-862332b7526026ae04645471c18a425c:2:4:2":"dadf163fd28a480bd80f5ed001d5b834e86470ccfde7d5926f2099ce8d650b2f","arena-2026-11-03:sub-862332b7526026ae04645471c18a425c:2:5:1":"4ae9ba38b4147521903632e07d536cbfa860d49ef7de2d569e9f87917859a533","arena-2026-11-03:sub-862332b7526026ae04645471c18a425c:2:5:2":"055cf4813ca7bce85c6de3f7beec5ecae0ffa41aec1a671642ef0c73b6928058","arena-2026-11-03:sub-862332b7526026ae04645471c18a425c:2:6:1":"61119dbecf8c013f79b0be4c714e60f0f77ae058352d84feba81a1da53eb5b22","arena-2026-11-03:sub-862332b7526026ae04645471c18a425c:2:7:1":"5bf06cbcfd5d94b607703f9e4f409cbf8d3fffbb05bef220f9a20adb015e8e50","arena-2026-11-03:sub-862332b7526026ae04645471c18a425c:2:8:1":"56e06719c6126ce8e13f3a12f2861a406acb06d0715edf23b1ff8bf6decf65f3","arena-2026-11-03:sub-862332b7526026ae04645471c18a425c:2:9:1":"d1a5aba409f51b33d7a6d5d61665937a87826f55ef4784354a96e00cad227a83","arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:0:1":"93d629ed6946ac8f4a99ab775f975a5ec41c71b2946207ef9f82e97a4eb0501d","arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:1:1":"ba74276c51287aa1f85619fd1c3cfd67ba0b9aa1bcdf2de27b7e78b2bf507115","arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:2:1":"22da94e685bb468b41f700c7918d03179d941ccb578baa78ffc59d9470c24c1e","arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:2:2":"4ad5c9e4f8c9bf657d9e479539b74da3f166844a1fb20fca49ef1e181047cacb","arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:3:1":"9476f764992b7f8c849bcb0fbf28139de0833822f88bae8cecb6567e21cebbe6","arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:3:2":"e031b2ef18e8f7f82c53d2906a18026ad46175d72082404e452ad7c42ec4b6bd","arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:4:1":"0e64f163a1414ef8a8d2d6bdd17658daaa2a533fc14333e451eb26c5cfe835c0","arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:4:2":"31ba8b22c4d46141d7248f1ba355ac73a78c29290a47a798bbdadc56f3e1040e","arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:5:1":"d54d49493f3aabdb0cfa6c76a8ce902e0b094b44232df30ab52df7ed290ba654","arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:5:2":"3c5e6533b0e4bd49a05f7372833b0171d747033d5eb671827737a4c892b04ce0","arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:6:1":"c4bdf55ae91a3cac261e385bff609486cef582e28b77154b899317675064b88c","arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:6:2":"e38757c8d5e0cd1ba9fdb332f5a0454f1dbca11fa11fca62249f5db06f47771b","arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:7:1":"a88c536e0bbbca9af30df8a0220498784a23aea505bf078a4b51eaee9f73d814","arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:7:2":"0b8aed9db50599021734f970b7d168feef44cb865c3e66a112f72379fa9bd5e8","arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:8:1":"a3a67aebb132109a8098f4e28cf4ceb5b757348cea4499e47932a6bd45b150e4","arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:8:2":"d8fabe65fa55318eec5aff60fa6420aed63d2b1a248afea173c62a9c7e6c3606","arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:9:1":"850aed80ca4f0887033b67163a56a13af2104a7907b9eae977dbb5e4df2f73c6","arena-2026-11-03:sub-a4fe920d6204d92824d54d9cd0149bc6:2:9:2":"1a1c40cc429bc42e7503b9b7c43a9f2c7a91fd6be2a379e783bf66c66abdd8eb","arena-2026-11-03:sub-b214ba8af13632081187b84b2a2be1aa:2:0:1":"b57e96726b9f1e1ab278f1636ec91db9ee05e6fbb66ac8b8626c659b1108fe53","arena-2026-11-03:sub-b214ba8af13632081187b84b2a2be1aa:2:1:1":"42a213890b676ce7f2266ed101b58d1c01f4ed542b488ce2c0e2258439088d4b","arena-2026-11-03:sub-b214ba8af13632081187b84b2a2be1aa:2:2:1":"281e86b52c7c980882d1a2a2299e32fab331011eaf3d070b438e0c1cb9304051","arena-2026-11-03:sub-b214ba8af13632081187b84b2a2be1aa:2:3:1":"12ef34ee29091ecb7b816eaa185a27489f719da2b0331630e06c8eabc2b824c1","arena-2026-11-03:sub-b214ba8af13632081187b84b2a2be1aa:2:4:1":"64e047156ff5fe0c102f93c8b4f4a9471b7163acce313eb7497f8a4dafe49d13","arena-2026-11-03:sub-b214ba8af13632081187b84b2a2be1aa:2:5:1":"822e109636c6f715b661527f1adeac1b70fa4521cfee57fb52e7022fba4d01b9","arena-2026-11-03:sub-b214ba8af13632081187b84b2a2be1aa:2:6:1":"cc54d02e6f15e7c4d8be0c1dfafe6fde95d06e2fd9b8788d0a413af760f631e8","arena-2026-11-03:sub-b214ba8af13632081187b84b2a2be1aa:2:7:1":"43f3f476512b74c221bf41e13874a92b21a142913d72c9676b516fcf31863a89","arena-2026-11-03:sub-b214ba8af13632081187b84b2a2be1aa:2:8:1":"52ef977a02250a5a58576716efe82b720eee3241ac27086440cff058a43b499a","arena-2026-11-03:sub-b214ba8af13632081187b84b2a2be1aa:2:9:1":"8c2324626619dd87f58fee2a9532aa899ce4b86d0333641ca82a3632236f6371","arena-2026-11-03:sub-edc9ae59687bddc1f9bb1b4e08e93e2c:2:0:1":"2700b82d3cdc9a533eaa73b68a02c745eed2588653b4246c408e6f8d341379d3","arena-2026-11-03:sub-edc9ae59687bddc1f9bb1b4e08e93e2c:2:1:1":"09b7aaa4f51f77827e3645cf8574057938d3db21c497897c5dccbee417eaa762","arena-2026-11-03:sub-edc9ae59687bddc1f9bb1b4e08e93e2c:2:2:1":"05d1be4b76200094b3938c872928c9482812e1cd4b202a6d479896b3abaf3376","arena-2026-11-03:sub-edc9ae59687bddc1f9bb1b4e08e93e2c:2:3:1":"cf95bbae1e2e5404377f165110fa7af78250fdb8b03d2a31c684df4b7406bdfd","arena-2026-11-03:sub-edc9ae59687bddc1f9bb1b4e08e93e2c:2:4:1":"231c8999a78b10b2e0e18cbd74f215ac42996e2d1fd7e746d863f08f5e0b3f4e","arena-2026-11-03:sub-edc9ae59687bddc1f9bb1b4e08e93e2c:2:5:1":"7ba7b5ddc1131091ea45710118f3b5a57f8df943c0fb902400acfaa9971b718d","arena-2026-11-03:sub-edc9ae59687bddc1f9bb1b4e08e93e2c:2:6:1":"257f8230933f5436d7bf94db91de81214b534924070363cb99c62df6c6c6cfce","arena-2026-11-03:sub-edc9ae59687bddc1f9bb1b4e08e93e2c:2:7:1":"4e788d585c0e7032f484d6d2255cadb750c15fe05ada63037d5f48f35d7e5648","arena-2026-11-03:sub-edc9ae59687bddc1f9bb1b4e08e93e2c:2:8:1":"e2b5d88be31271a799a233ef91399c12aaaf5a3ddcc2deaa107b1a2540eb5ca9","arena-2026-11-03:sub-edc9ae59687bddc1f9bb1b4e08e93e2c:2:9:1":"9d36159057a32d92a905677767e2db5a79eb6b8c60da14895cdc4224dbf9857c","arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:0:1":"73c20b64411d68ebd5eb87ce06efb1982503842bcef565bc4bf0ed690e1db7fc","arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:1:1":"6383bdece9b197d361a0e7dcc08ea636f60efae285843c53bf9c5cc71ee9d0d6","arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:1:2":"308b081e51dbfaa025f7f4b72f6b60e541f899047c23d3f1a56f730b919a18b8","arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:2:1":"fad09ada8469ab990461b90ce349b5ce1222849f742839c1c0515e03989e4a29","arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:2:2":"7ad07453efd196bf06d2636dee13cf03db5373422c8b5f57edb088593e885101","arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:3:1":"4068b222cc84096b0ba5236025f95a6826efb2149d766eba373768bd20b8b410","arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:4:1":"d72c986ce4ed06a3d0eaf7a246d397fd637c6f85ad3c9cdca3545c6dcb370df4","arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:4:2":"2b65a8e071ea44b5733ea9ab87ba91c65bfa42ac2465a51ead24473b013563b8","arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:5:1":"54745206a409c23b2f1c359e487bc283d299538e2fa7ca10b271b076780257a9","arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:5:2":"3dae7cc8b28c7dce5dcd9bf8f7950f255ccf14cf5110c9fc80beecbd6d13738c","arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:6:1":"bb6c5d6f2eb8561a617522a0672c8f3751be7d3023000006f99b62a4e674035f","arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:6:2":"7726f47dcc0e1cc7e55bf3933e20623ee92d4e20db654c2cd631a910e7dfb2c0","arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:7:1":"a558efc9bc4fbdf1804206bb1abf34d9ce470b42f969ac0c2f55d23a250a3085","arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:7:2":"7fa1408b524dfa2546bb611a89e9c764b187ac6f0809644050c4c706b5532dd9","arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:8:1":"6773c718ebe90211c5328428b0d3e19ec42b7d56486da338db7da213432675f2","arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:8:2":"6f0b64e3a1e65a4ca6b0ff6d8a346e3833becedda5cc00e2a1e55b8b99d7f262","arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:9:1":"bf73fbca525b0f21f7029cb596f753826d9648a4f94f2588ab91e65f1d36ff0b"}}$manifest604$::JSONB;
 r public.lab_arena_rounds; n public.lab_arena_rounds; c public.lab_arena_restart_claim_control;
 h public.lab_arena_scoring_supersessions; revision BIGINT; actual_hashes JSONB;
 expected_policy CONSTANT JSONB:=$policy604${"company_cap_rule":"icp_max_companies","company_judgment_policy":"company_judgment_v1","employee_bucket_rule":"lab_relaxed_buckets","env_bindings":{"ARENA_FIT_FP_CLASSIFICATION":"company_fit_receipt_v1","ARENA_PROVIDER_OBSERVATION_HANDOFF":"authenticated_first_observed_v1","ARENA_SCORE_NORMALIZATION":"available_intent_cap_v1"},"fp_penalty_icp_floor":0.0,"fp_penalty_points":10.0,"fp_unverified_primary_penalty_points":10.0,"intent_details_policy":"intent_details_v1","judge_models":{"company_fit_reverification":"perplexity/sonar","industry_evidence_review":"openai/gpt-6.1-sol","intent_details_review":"anthropic/claude-sonnet-5.5","intent_fetched_evidence":"openai/gpt-6-luna","intent_signal_judge":"openai/gpt-6-luna","intent_three_stage_stage3":"perplexity/sonar-pro","role_batch_check":"google/gemini-3.5-flash-lite"},"max_scored_companies":0,"pre_slice_rule":"first_n_model_order","provider_profile":"lab_arena","schema_version":"leadpoet.lab_arena.scorer_policy.v1","scoring_adapter_version":"qualification_integrity_v2","signal_judgment_policy":"signal_judgment_v1"}$policy604$::JSONB;
 new_digest CONSTANT TEXT:='sha256:1093b23cc475051ef43d0123062f0f494055e5d0c608237f62745532f5630b50'; new_reference CONSTANT TEXT:='493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:1093b23cc475051ef43d0123062f0f494055e5d0c608237f62745532f5630b50';
BEGIN
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 SELECT * INTO STRICT c FROM public.lab_arena_restart_claim_control WHERE singleton FOR UPDATE;
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=rid FOR UPDATE;
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions WHERE round_id=rid AND request_id=request;
 IF FOUND THEN
  IF h.request_hash IS DISTINCT FROM 'sha256:957e7a1c73c0dec7d438ccd4486434e0962eb5bb254191eafa4d37ce24fd4522'
   OR h.recovery_doc->>'operation' IS DISTINCT FROM 'nov03_forward_only_verifier_v1'
   OR h.recovery_doc->'preserved_judgments' IS DISTINCT FROM manifest->'preserved_judgments'
   OR h.recovery_doc->'original_run_hashes' IS DISTINCT FROM manifest->'run_hashes'
   OR r.configuration_doc IS DISTINCT FROM h.new_round_doc->'configuration_doc'
   OR r.scoring_generation_floor IS DISTINCT FROM 0 THEN
   RAISE EXCEPTION 'Nov03 forward verifier replay differs'; END IF;
  RETURN;
 END IF;
 IF current_setting('session_replication_role')<>'origin'
  OR NOT c.operator_paused OR c.pause_reason IS DISTINCT FROM 'nov03_improved_verifier_required603'
  OR c.guard_generation IS DISTINCT FROM 754
  OR c.guard_commitment IS DISTINCT FROM '' OR c.owner_commitment IS DISTINCT FROM ''
  OR c.guard_expires_at IS NOT NULL OR c.candidate_commit IS DISTINCT FROM ''
  OR c.restart_scope IS DISTINCT FROM '' OR c.restart_phase IS DISTINCT FROM ''
  OR c.captured_leases IS DISTINCT FROM '[]'::JSONB
  OR r.status IS DISTINCT FROM 'stage2' OR r.status_generation IS DISTINCT FROM 5
  OR r.stage_generation IS DISTINCT FROM 4 OR r.scoring_generation_floor IS DISTINCT FROM 0
  OR r.evaluation_date IS DISTINCT FROM '2026-11-03' OR r.icp_set_date IS DISTINCT FROM '2026-11-02'
  OR r.benchmark_ref IS DISTINCT FROM 'arena/arena-2026-11-03/benchmark.json'
  OR r.configuration_doc->>'mode' IS DISTINCT FROM 'live'
  OR encode(extensions.digest(r.configuration_doc::TEXT,'sha256'),'hex') IS DISTINCT FROM 'e91499fe3037f17d0363f94ffdcfe383f49ceace8a365dc40c4a72277e21337c'
  OR encode(extensions.digest(r.participants::TEXT,'sha256'),'hex') IS DISTINCT FROM 'a7b48bac1538146d8ec5ceacec4ac6b054a4537621e903ef516dd430ff5d4452'
  OR r.published_at IS NOT NULL OR r.publication_doc IS NOT NULL OR r.promotion_doc IS NOT NULL
  OR r.reward_activated_at IS NOT NULL OR r.reward_basis_hash IS NOT NULL
  OR r.effective_reward_epoch IS NOT NULL
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.status IN('leased','submitted')) THEN
  RAISE EXCEPTION 'Nov03 forward verifier round or pause preimage differs'; END IF;
 SELECT coalesce(jsonb_object_agg(a.run_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex')),'{}')
 INTO actual_hashes FROM public.lab_arena_runs a WHERE a.round_id=rid;
 IF actual_hashes IS DISTINCT FROM manifest->'run_hashes'
  OR (SELECT count(*) FROM public.lab_arena_runs WHERE round_id=rid AND kind='score' AND status='accepted')<>9
  OR (SELECT count(*) FROM jsonb_object_keys(manifest->'preserved_judgments'))<>9 THEN
  RAISE EXCEPTION 'Nov03 forward verifier original run preimage differs'; END IF;
 n:=r;
 n.configuration_doc:=jsonb_set(jsonb_set(jsonb_set(r.configuration_doc,
  '{scorer_policy,judge_models,industry_evidence_review}','"openai/gpt-6.1-sol"'::JSONB),
  '{scorer_image_digest}',to_jsonb(new_digest)),'{scorer_image_reference}',to_jsonb(new_reference));
 IF n.configuration_doc->'scorer_policy' IS DISTINCT FROM expected_policy
  OR encode(extensions.digest(n.configuration_doc::TEXT,'sha256'),'hex') IS DISTINCT FROM '7ec5660ffafd85922005cfad0c30c2e1a141899cf5dc6c47341d526c940c7e95'
  OR encode(extensions.digest((n.configuration_doc->'scorer_policy')::TEXT,'sha256'),'hex') IS DISTINCT FROM 'affc53521322c343401f266b5d2861b04efc4a96db335f9cea5a7a14ad8ccb2f'
  OR (to_jsonb(n)-'configuration_doc') IS DISTINCT FROM (to_jsonb(r)-'configuration_doc') THEN
  RAISE EXCEPTION 'Nov03 forward verifier exact configuration delta differs'; END IF;
 SELECT greatest(1,coalesce(max(generation_floor),0)+1) INTO revision
 FROM public.lab_arena_scoring_supersessions WHERE round_id=rid;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,
  actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(rid,revision,request,'sha256:957e7a1c73c0dec7d438ccd4486434e0962eb5bb254191eafa4d37ce24fd4522','nov03-forward-only-verifier604',txid_current(),pg_backend_pid(),
  to_jsonb(r)-'updated_at',to_jsonb(n)-'updated_at','{}',
  jsonb_build_object('operation','nov03_forward_only_verifier_v1','scoring_generation_floor',0,
   'preserved_judgments',manifest->'preserved_judgments','original_run_hashes',manifest->'run_hashes',
   'new_scorer_policy_hash','sha256:7d960d774ea473262da1bdd47ffd30b837e562083cefc868f078e27f7f7d8ab2',
   'reason','Preserve nine completed old-verifier judgments; use the improved verifier only for unscored work. No provider replay, execution retry, judge-attempt reset or deadline change.'));
 UPDATE public.lab_arena_rounds SET configuration_doc=n.configuration_doc WHERE round_id=rid;
 IF EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.round_id=rid
  AND manifest->'run_hashes'->>a.run_id IS DISTINCT FROM encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex'))
  OR (SELECT to_jsonb(control) FROM public.lab_arena_restart_claim_control control WHERE singleton) IS DISTINCT FROM to_jsonb(c) THEN
  RAISE EXCEPTION 'Nov03 forward verifier changed original work or hold'; END IF;
END;
$nov03_forward604$;

DO $restore604$
BEGIN
 IF NOT (SELECT had_create FROM pg_temp.arena604_schema_acl) THEN
  REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
 END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM
  (SELECT nspacl FROM pg_temp.arena604_schema_acl) THEN
  RAISE EXCEPTION 'Nov03 forward verifier schema ACL differs'; END IF;
END;
$restore604$;
NOTIFY pgrst,'reload schema';
COMMIT;
