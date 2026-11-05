-- Reuse exact existing score assignments when an execution plan is rebuilt.
-- A one-time approved Nov5 correction restores unused pending attempts and
-- queues ordinary second attempts. Accepted work and every bill remain intact.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='90s';
DO $idempotent633$
DECLARE target OID:='public.lab_arena_open_scoring_v3(text,smallint,jsonb)'::regprocedure;
 d TEXT; m JSONB; expected JSONB;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',jsonb_build_array(pg_get_userbyid(p.proowner),
  p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname) INTO d,m,expected
 FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target;
 IF expected IS DISTINCT FROM '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public","statement_timeout=60s"],"plpgsql"]'::JSONB THEN
  RAISE EXCEPTION 'Recovery633 open scoring security differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='373fad0500ff5a7cdeee04fd7524f827df0f5987081bd10eddc8e88e617b1dc0' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'0d2c9cd4aadea7d8de4f455dfc9d3d352527940b3833a9ad9af5ca967fce18e3'
  OR length(d)-length(replace(d,$old633$    INSERT INTO public.lab_arena_runs (
      run_id, assignment_id, round_id, submission_id, miner_hotkey, stage,
      icp_position, attempt, status, stage_generation, kind, scored_run_id,
      company_judgment_refs
$old633$,''))<>length($old633$    INSERT INTO public.lab_arena_runs (
      run_id, assignment_id, round_id, submission_id, miner_hotkey, stage,
      icp_position, attempt, status, stage_generation, kind, scored_run_id,
      company_judgment_refs
$old633$) THEN
  RAISE EXCEPTION 'Recovery633 open scoring preimage differs'; END IF;
 d:=replace(d,$old633$    INSERT INTO public.lab_arena_runs (
      run_id, assignment_id, round_id, submission_id, miner_hotkey, stage,
      icp_position, attempt, status, stage_generation, kind, scored_run_id,
      company_judgment_refs
$old633$,$new633$    IF (SELECT count(*) FROM pg_catalog.jsonb_array_elements(p_work_items) other
      WHERE (other->'submission_id',other->'icp_position') IS NOT DISTINCT FROM
        (v_item->'submission_id',v_item->'icp_position'))<>1 THEN
      RAISE EXCEPTION 'lab_arena_duplicate_scoring_item' USING ERRCODE='22023';
    END IF;
    -- Reopening an unchanged plan must not recreate accepted, active or
    -- exhausted judgments. Claim-time cache receipts and runtime pins can
    -- change later; the frozen plan input and source identity cannot.
    IF EXISTS (SELECT 1 FROM public.lab_arena_runs existing
      WHERE existing.assignment_id=v_assignment
        AND existing.scoring_generation_floor=v_round.scoring_generation_floor) THEN
      IF EXISTS (SELECT 1 FROM public.lab_arena_runs existing
        WHERE existing.assignment_id=v_assignment
          AND existing.scoring_generation_floor=v_round.scoring_generation_floor
          AND ((existing.round_id,existing.submission_id,existing.miner_hotkey,
                existing.stage,existing.icp_position,existing.kind,existing.scored_run_id)
            IS DISTINCT FROM (p_round_id,v_scored.submission_id,v_scored.miner_hotkey,
                p_stage,v_scored.icp_position,'score'::TEXT,v_scored.run_id)
            OR existing.company_judgment_refs IS DISTINCT FROM v_item->'company_judgment_refs'
            OR existing.judgment_cache_key IS NOT NULL OR existing.judgment_input_hash IS NOT NULL
            OR existing.judgment_scope_doc IS NOT NULL OR existing.judgment_group_leader IS NOT NULL
            OR existing.judgment_group_miner_hotkeys IS NOT NULL OR existing.judgment_cache_source_run_id IS NOT NULL)) THEN
        RAISE EXCEPTION 'lab_arena_existing_scoring_input_conflict' USING ERRCODE='22023';
      END IF;
      CONTINUE;
    END IF;
    INSERT INTO public.lab_arena_runs (
      run_id, assignment_id, round_id, submission_id, miner_hotkey, stage,
      icp_position, attempt, status, stage_generation, kind, scored_run_id,
      company_judgment_refs
$new633$);
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'373fad0500ff5a7cdeee04fd7524f827df0f5987081bd10eddc8e88e617b1dc0' THEN
  RAISE EXCEPTION 'Recovery633 open scoring readback differs'; END IF;
END;
$idempotent633$;
DO $history633$
DECLARE target OID:='public.lab_arena_scoring_policy_history_v1(text)'::regprocedure; d TEXT; m JSONB;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,m FROM pg_proc p WHERE p.oid=target;
 IF pg_get_userbyid((m->>'proowner')::OID)<>'lab_arena_owner'
  OR m->'proacl' IS DISTINCT FROM '["lab_arena_owner=X/lab_arena_owner","lab_arena_service=X/lab_arena_owner"]'::JSONB
  OR m->>'prosecdef'<>'true' OR m->>'provolatile'<>'s'
  OR m->'proconfig'<>'["search_path=pg_catalog, public"]'::JSONB THEN
  RAISE EXCEPTION 'Recovery633 history security differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='27da80592b12e32477348f106756d4e5bf9ca8dabe9e604291530519a3a93d43' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'1bc738f6e73c29ac756106ff233af894915a492547334f6ecb66815654f6031a'
  OR length(d)-length(replace(d,$old633$'baseline_donor_evidence630_v1') THEN$old633$,''))<>length($old633$'baseline_donor_evidence630_v1') THEN$old633$) THEN
  RAISE EXCEPTION 'Recovery633 history preimage differs'; END IF;
 EXECUTE replace(d,$old633$'baseline_donor_evidence630_v1') THEN$old633$,$new633$'baseline_donor_evidence630_v1','nov05_infrastructure_recovery633_v1') THEN$new633$);
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'27da80592b12e32477348f106756d4e5bf9ca8dabe9e604291530519a3a93d43' THEN
  RAISE EXCEPTION 'Recovery633 history readback differs'; END IF;
END;
$history633$;

-- Exact one-time data correction.
DO $data633$
DECLARE manifest CONSTANT JSONB:=$manifest633${"configuration_sha256":"3c9d44fcfc541663ecb7ad8de5d4a6178284741f37c39c7070fb2d6a155a54ab","plan_sha256":"9692d2b28d8ea829d27070c137288a1fb07b0713c3e34149b20b01976becbe02","round_id":"arena-2026-11-05","schema_version":"leadpoet.lab_arena.nov05_recovery633.v1","sources":{"sub-032f92a20d9d1c52b20e2d7a075fbe02":"efe839c179ed94ac7389ed5c99f63dfde56a4c40d82d11b1e588e5f6e7855775","sub-0b2072dffefde95bb2c0d73e4cd90afc":"1cb0bf785398c103a8aa883371213e9c1f0d85d04fa630722c09be38bf06a98c","sub-2ccf157ed581723860431b347f243716":"ba779404c204aa1bfad38ab94666388117225f3f32309e45c76bb4c9843f2ff0","sub-5b84868cfb6c885036697a1eb2e60a19":"d8ebd79f28fbdecac70865f54f9a3ae623376de9ffa067f6d3833ce783d01380","sub-b0a5f04755fc31314ce06a8c65a118d3":"e7ec65181ef843c5d796ccc01db9a953298483a0f5f90ea07732c7addb796d32","sub-d0b73c86bdd9d3ce0ef76c5c78a95067":"ea1dc1a7fa72107d494fd4302571726568308542750845dd1d7d313a8d75c8c8"},"stage_generation":7,"status_generation":8,"targets":[{"mode":"restore_unused","row_sha256":"2565487bf018efa9b46fbc9755b6851058a2c51b3092dd57384834406d5cbaf1","run_id":"arena-2026-11-05:sub-032f92a20d9d1c52b20e2d7a075fbe02:2:0:2"},{"mode":"restore_unused","row_sha256":"7d2d26c881a6f41511f9a98b69e0fadca37688a0386eee6b4d35c099fda973d4","run_id":"arena-2026-11-05:sub-032f92a20d9d1c52b20e2d7a075fbe02:2:1:2"},{"mode":"restore_unused","row_sha256":"b9681677f51a877e8e20ecc9f12c2f4bfa19d34c7b7b599c918fab951e776cad","run_id":"arena-2026-11-05:sub-032f92a20d9d1c52b20e2d7a075fbe02:2:2:2"},{"mode":"restore_unused","row_sha256":"9fde7a5b9687cbaade20aec44c135d2fb269a8dc2fed34dd874b60a045175464","run_id":"arena-2026-11-05:sub-032f92a20d9d1c52b20e2d7a075fbe02:2:3:2"},{"mode":"restore_unused","row_sha256":"616ec54eb276ccebae22df0da536e3c28023f950c8dcd490612ab4c30e57614a","run_id":"arena-2026-11-05:sub-032f92a20d9d1c52b20e2d7a075fbe02:2:4:2"},{"mode":"restore_unused","row_sha256":"b30f4fadd2d5c5a6c7cb183daf2846e69dc8db22e850521c026fd0ed4cbe5472","run_id":"arena-2026-11-05:sub-032f92a20d9d1c52b20e2d7a075fbe02:2:5:2"},{"mode":"restore_unused","row_sha256":"fe253b12afa12bca44c559d39cd3b222ddea5d30462778ed4da82a425b573704","run_id":"arena-2026-11-05:sub-032f92a20d9d1c52b20e2d7a075fbe02:2:6:2"},{"mode":"restore_unused","row_sha256":"877e84fc1226fc835e1d70ad2dcc61884acb68fef2553c981ff3165010f5142c","run_id":"arena-2026-11-05:sub-032f92a20d9d1c52b20e2d7a075fbe02:2:7:2"},{"mode":"restore_unused","row_sha256":"7ec4251b4c91a26675eda3d6e9ac927e77565993c4606f216bd2d8abe3154178","run_id":"arena-2026-11-05:sub-032f92a20d9d1c52b20e2d7a075fbe02:2:8:2"},{"mode":"restore_unused","row_sha256":"8f763322a9df873513bb2899eb3d4fdf83df14de87895504d729fa147780cadb","run_id":"arena-2026-11-05:sub-032f92a20d9d1c52b20e2d7a075fbe02:2:9:2"},{"mode":"restore_unused","row_sha256":"8f42ff53133e7d98206963b9ab885fab1cfc389fcbe00145c96efb3e8d5a31e5","run_id":"arena-2026-11-05:sub-0b2072dffefde95bb2c0d73e4cd90afc:2:0:1"},{"mode":"restore_unused","row_sha256":"223a5f03deb7ef7ee189fbb1674f59586ea6b9fa3f96b7243f30fbe5123aaa95","run_id":"arena-2026-11-05:sub-0b2072dffefde95bb2c0d73e4cd90afc:2:1:1"},{"mode":"restore_unused","row_sha256":"002c358b2aa6c962995a52ec23a1e385cfef74fe2fbf1283d497ef5e4d7fa38f","run_id":"arena-2026-11-05:sub-0b2072dffefde95bb2c0d73e4cd90afc:2:2:1"},{"mode":"restore_unused","row_sha256":"5cb5f05064bf448b95ddbb81cf36ee576629607facc4a3b805f93081992349dd","run_id":"arena-2026-11-05:sub-0b2072dffefde95bb2c0d73e4cd90afc:2:3:1"},{"mode":"restore_unused","row_sha256":"92b1004d995737c821c46445449bcd369ce2a36ca5d47dc84d9e96c1e78d0d00","run_id":"arena-2026-11-05:sub-0b2072dffefde95bb2c0d73e4cd90afc:2:4:1"},{"mode":"restore_unused","row_sha256":"ad49e644934aa35d358048c225105f80063de39990b96f143ab725ea1b3f95ea","run_id":"arena-2026-11-05:sub-0b2072dffefde95bb2c0d73e4cd90afc:2:5:1"},{"mode":"restore_unused","row_sha256":"83afb94afc6f5d9df7f040a2a7b0bb74134c23d05723c8dfb1627edb83650372","run_id":"arena-2026-11-05:sub-0b2072dffefde95bb2c0d73e4cd90afc:2:6:1"},{"mode":"restore_unused","row_sha256":"3e36f193adf28f25b1d1b102a1639abdb0529094c78eda59e460f804bbecdede","run_id":"arena-2026-11-05:sub-0b2072dffefde95bb2c0d73e4cd90afc:2:7:1"},{"mode":"restore_unused","row_sha256":"faff6d72e6eb6d06c6bc0c8c5d72a9a7d62a504b99d931c734ad102136b788a0","run_id":"arena-2026-11-05:sub-0b2072dffefde95bb2c0d73e4cd90afc:2:8:1"},{"mode":"restore_unused","row_sha256":"c32da151ba09b98f168e43bd93d883a682f92b6ffe5cce441c526daf44f1892a","run_id":"arena-2026-11-05:sub-0b2072dffefde95bb2c0d73e4cd90afc:2:9:1"},{"mode":"ordinary_retry","row_sha256":"b2762c19384c54581e576449629396daad39d0609e84e2adf9dfa5fd996116d5","run_id":"arena-2026-11-05:sub-2ccf157ed581723860431b347f243716:2:0:1"},{"mode":"ordinary_retry","row_sha256":"d6134b095161d29346a2daf072a40957d92735a84a203a5755ead62fb98ff3d9","run_id":"arena-2026-11-05:sub-2ccf157ed581723860431b347f243716:2:1:1"},{"mode":"ordinary_retry","row_sha256":"f5f64ea0d137a0b4fa0b4a44f53d30d9c635bf6a2795e775be6dacaa02350a48","run_id":"arena-2026-11-05:sub-2ccf157ed581723860431b347f243716:2:2:1"},{"mode":"ordinary_retry","row_sha256":"20e41bad684ac5ffe80d78ac8db1d450e147b7887554a8c4313ddb4ec6499a26","run_id":"arena-2026-11-05:sub-2ccf157ed581723860431b347f243716:2:3:1"},{"mode":"ordinary_retry","row_sha256":"220e3ba4a8760386cc39144e7918df7b76e25cfc28deb8f51b44b901880e4262","run_id":"arena-2026-11-05:sub-2ccf157ed581723860431b347f243716:2:4:1"},{"mode":"ordinary_retry","row_sha256":"d426b8c17d76494354ee5472b00f1056ec683e6d3e45e71659c5c7c07e51cf81","run_id":"arena-2026-11-05:sub-2ccf157ed581723860431b347f243716:2:5:1"},{"mode":"ordinary_retry","row_sha256":"ff24515504c51e6eb49d24ee9b1e5b0972acce3657d94085f231b41dc101651d","run_id":"arena-2026-11-05:sub-2ccf157ed581723860431b347f243716:2:6:1"},{"mode":"ordinary_retry","row_sha256":"bb2db252610e2a245152f47748399b527875cefb73d774e72dd8088eff59af0b","run_id":"arena-2026-11-05:sub-2ccf157ed581723860431b347f243716:2:7:1"},{"mode":"ordinary_retry","row_sha256":"7c06363ef3e9bb08b99df36daad450a6cefa44c5a7df2cb1bec8934c400ced5e","run_id":"arena-2026-11-05:sub-2ccf157ed581723860431b347f243716:2:8:1"},{"mode":"ordinary_retry","row_sha256":"47f254186d63b7eead87081e4a543395d2887000288a5583e2079c57b6af5f0c","run_id":"arena-2026-11-05:sub-2ccf157ed581723860431b347f243716:2:9:1"},{"mode":"restore_unused","row_sha256":"04f36297cfb78c23786bbf8b5526d24298426116d81a6422c7562ea502960030","run_id":"arena-2026-11-05:sub-5b84868cfb6c885036697a1eb2e60a19:2:7:2"},{"mode":"restore_unused","row_sha256":"ca3df8d3a0381069d6065d451b62672002ae5ad6b3471c6730144266066ecea6","run_id":"arena-2026-11-05:sub-5b84868cfb6c885036697a1eb2e60a19:2:8:2"},{"mode":"restore_unused","row_sha256":"274f14795b9017cbde808b55a2b96572f7c57f558f7cf37b08f6aadddb7db5a5","run_id":"arena-2026-11-05:sub-5b84868cfb6c885036697a1eb2e60a19:2:9:2"},{"mode":"restore_unused","row_sha256":"8ea89d76390d0b4ba506c92da92b4af09b94853d11c9499f0884c1a2390d70bb","run_id":"arena-2026-11-05:sub-b0a5f04755fc31314ce06a8c65a118d3:2:3:1"},{"mode":"restore_unused","row_sha256":"31542795a2bfb5b3dd02c615bb51e97c1e6b5119b48a3e19bf1843e7de2cfa81","run_id":"arena-2026-11-05:sub-b0a5f04755fc31314ce06a8c65a118d3:2:4:1"},{"mode":"restore_unused","row_sha256":"e947e4bf1d0f63efe6798807a1c72187212ce334bff70b4c3baa4a6dfaf3350d","run_id":"arena-2026-11-05:sub-b0a5f04755fc31314ce06a8c65a118d3:2:5:1"},{"mode":"restore_unused","row_sha256":"ed2e8c8a15ba8d22e18bcc68422c2433ded78e78118f044d128370d1dca2f5ef","run_id":"arena-2026-11-05:sub-b0a5f04755fc31314ce06a8c65a118d3:2:6:1"},{"mode":"restore_unused","row_sha256":"c6658be0eb3a2f97f438121b79d313f6365b9ceeda510baab5d9a872d33b0a10","run_id":"arena-2026-11-05:sub-b0a5f04755fc31314ce06a8c65a118d3:2:7:1"},{"mode":"restore_unused","row_sha256":"34308db95b6205a53d8d0728e974d64e69de2c2fa8ccda26cf0926884bc43103","run_id":"arena-2026-11-05:sub-b0a5f04755fc31314ce06a8c65a118d3:2:8:1"},{"mode":"restore_unused","row_sha256":"24a0a6cc7c15c770feddb6c684dbb57461cd93b5fc50523e68f4cfefc8112896","run_id":"arena-2026-11-05:sub-b0a5f04755fc31314ce06a8c65a118d3:2:9:1"},{"mode":"ordinary_retry","row_sha256":"aef72a4a48dca5b50ca462dd589b03e804a03a2c3af3d28d212f39f5194ce97f","run_id":"arena-2026-11-05:sub-d0b73c86bdd9d3ce0ef76c5c78a95067:2:0:1"},{"mode":"ordinary_retry","row_sha256":"f39d099d8150d523491c65dc52e03ad8d83e8fdbd16cc5a8d3dc77f68c0a74f9","run_id":"arena-2026-11-05:sub-d0b73c86bdd9d3ce0ef76c5c78a95067:2:1:1"},{"mode":"ordinary_retry","row_sha256":"0c7170a843847925210e69d9248d4deb4f34da8b066d4c2f73a372134d13a1b2","run_id":"arena-2026-11-05:sub-d0b73c86bdd9d3ce0ef76c5c78a95067:2:2:1"},{"mode":"ordinary_retry","row_sha256":"3c65882ac8b24628a5b01b5fe38933b93cef9618ed19f78b04edff94b48f65ec","run_id":"arena-2026-11-05:sub-d0b73c86bdd9d3ce0ef76c5c78a95067:2:3:1"},{"mode":"restore_unused","row_sha256":"41cead941776631d01f34c2fa7c9847de0b5942aa925dabe73d99f034868c394","run_id":"arena-2026-11-05:sub-d0b73c86bdd9d3ce0ef76c5c78a95067:2:4:1"},{"mode":"restore_unused","row_sha256":"3829b5df82b9a58f61b307216ea62372b6ca98cc449baa34207324b0767de025","run_id":"arena-2026-11-05:sub-d0b73c86bdd9d3ce0ef76c5c78a95067:2:5:1"},{"mode":"restore_unused","row_sha256":"3b0543a0ab74062210620986239dabfa72f2299014c2020f280644bfef2b4839","run_id":"arena-2026-11-05:sub-d0b73c86bdd9d3ce0ef76c5c78a95067:2:6:1"},{"mode":"restore_unused","row_sha256":"f9ff799c8979352fa989f7728d0dded2cf8d68ffa4b44ab07e9c31a5337487e5","run_id":"arena-2026-11-05:sub-d0b73c86bdd9d3ce0ef76c5c78a95067:2:7:1"},{"mode":"restore_unused","row_sha256":"e5e10f81a1fa3ede52adc2403ccd3bc2b27b94c2cab3681caffbb6d1d428a4b6","run_id":"arena-2026-11-05:sub-d0b73c86bdd9d3ce0ef76c5c78a95067:2:8:1"},{"mode":"restore_unused","row_sha256":"8ab9b7255e650f0b47f5194959ab6b0cc575b7f4735ad4b1908fe90735a27ae5","run_id":"arena-2026-11-05:sub-d0b73c86bdd9d3ce0ef76c5c78a95067:2:9:1"}]}$manifest633$;
 r public.lab_arena_rounds; n public.lab_arena_rounds; h public.lab_arena_scoring_supersessions;
 parent public.lab_arena_runs; source public.lab_arena_submissions; target JSONB;
 originals JSONB; before_runs JSONB; before_accepted JSONB; before_submissions JSONB;
 before_triggers JSONB; after_triggers JSONB; before_rounds JSONB;
 cost JSONB; provider RECORD; available BOOLEAN; created INTEGER:=0; restored INTEGER:=0;
 revision BIGINT; request_hash TEXT; expected_config JSONB; known_ids TEXT[]; successor_ids TEXT[];
BEGIN
 IF current_setting('session_replication_role') IS DISTINCT FROM 'origin' THEN
  RAISE EXCEPTION 'Recovery633 origin triggers required'; END IF;
 request_hash:='sha256:'||encode(extensions.digest(manifest::TEXT,'sha256'),'hex');
 PERFORM pg_advisory_xact_lock(hashtextextended('lab_arena.baseline_promotion',0));
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id='arena-2026-11-05' FOR UPDATE;
 expected_config:=jsonb_set(jsonb_set(jsonb_set(r.configuration_doc,
  '{schedule,stage_2_close}','"2026-11-05T20:30:00Z"'::JSONB,FALSE),
  '{schedule,final_scoring_close}','"2026-11-05T23:58:59Z"'::JSONB,FALSE),
  '{schedule,publication_deadline}','"2026-11-05T23:59:00Z"'::JSONB,FALSE);
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions
 WHERE round_id=r.round_id AND request_id='nov05-approved-infrastructure633-v1';
 IF FOUND THEN
  IF h.request_hash IS DISTINCT FROM request_hash
   OR h.recovery_doc->>'operation' IS DISTINCT FROM 'nov05_infrastructure_recovery633_v1'
   OR h.recovery_doc->'manifest' IS DISTINCT FROM manifest
   OR h.new_round_doc->'configuration_doc' IS DISTINCT FROM r.configuration_doc
   OR r.scoring_generation_floor<>0
   OR (SELECT count(*) FROM jsonb_object_keys(h.recovery_doc->'original_runs'))<>50
   OR (SELECT count(*) FROM public.lab_arena_runs a,jsonb_array_elements(manifest->'targets') t
     WHERE t->>'mode'='ordinary_retry' AND a.run_id=regexp_replace(t->>'run_id',':1$',':2')
      AND a.kind='execute' AND a.attempt=2 AND a.round_id=r.round_id)<>14 THEN
   RAISE EXCEPTION 'Recovery633 replay differs'; END IF;
  RETURN; -- Normal claims, completions, scoring and publication are never reset.
 END IF;
 IF r.status IS DISTINCT FROM 'stage2_judged' OR r.evaluation_date<>'2026-11-05'
  OR r.status_generation IS DISTINCT FROM (manifest->>'status_generation')::BIGINT
  OR r.stage_generation IS DISTINCT FROM (manifest->>'stage_generation')::BIGINT
  OR r.scoring_generation_floor<>0
  OR encode(extensions.digest(r.configuration_doc::TEXT,'sha256'),'hex') IS DISTINCT FROM manifest->>'configuration_sha256'
  OR encode(extensions.digest(r.stage2_scoring_plan_doc::TEXT,'sha256'),'hex') IS DISTINCT FROM manifest->>'plan_sha256'
  OR jsonb_array_length(r.stage2_scoring_plan_doc->'work_items')<>165
  OR r.configuration_doc->>'max_attempts_per_assignment'<>'2'
  OR r.configuration_doc->>'sourcing_cost_eligibility_policy'<>'successful_calls_flat_per_icp_v1'
  OR r.published_at IS NOT NULL OR r.publication_doc IS NOT NULL OR r.promotion_doc IS NOT NULL
  OR r.reward_activated_at IS NOT NULL OR r.reward_basis_doc IS NOT NULL OR r.reward_basis_hash IS NOT NULL
  OR clock_timestamp()>='2026-11-05T20:30:00Z'::TIMESTAMPTZ
  OR jsonb_array_length(manifest->'targets')<>50
  OR (SELECT count(*) FROM jsonb_array_elements(manifest->'targets')t WHERE t->>'mode'='restore_unused')<>36
  OR (SELECT count(*) FROM jsonb_array_elements(manifest->'targets')t WHERE t->>'mode'='ordinary_retry')<>14
  OR (SELECT count(DISTINCT t->>'run_id') FROM jsonb_array_elements(manifest->'targets')t)<>50
  OR (SELECT count(*) FROM public.lab_arena_restart_claim_control WHERE singleton)<>1
  OR EXISTS(SELECT 1 FROM public.lab_arena_restart_claim_control WHERE operator_paused OR guard_commitment<>''
   OR owner_commitment<>'' OR guard_expires_at IS NOT NULL OR candidate_commit<>'' OR restart_scope<>'' OR restart_phase<>'')
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.round_id=r.round_id AND a.status IN('pending','leased','submitted')) THEN
  RAISE EXCEPTION 'Recovery633 round, clock, scope or control differs'; END IF;
 -- Claim-control ownership prevents new admissions. The exclusive table lock
 -- makes the short terminal-guard suspension invisible to concurrent writers.
 LOCK TABLE public.lab_arena_runs IN ACCESS EXCLUSIVE MODE;
 SELECT jsonb_object_agg(a.run_id,to_jsonb(a)) INTO before_runs FROM public.lab_arena_runs a WHERE a.round_id=r.round_id;
 SELECT jsonb_object_agg(a.run_id,to_jsonb(a)) INTO before_accepted FROM public.lab_arena_runs a WHERE a.round_id=r.round_id AND a.status='accepted';
 SELECT jsonb_object_agg(s.submission_id,to_jsonb(s)) INTO before_submissions FROM public.lab_arena_submissions s WHERE s.round_id=r.round_id;
 SELECT jsonb_object_agg(a.round_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex')) INTO before_rounds FROM public.lab_arena_rounds a WHERE a.round_id<>r.round_id;
 SELECT jsonb_agg(jsonb_build_array(t.tgname,t.tgenabled,pg_get_triggerdef(t.oid),pg_get_functiondef(t.tgfoid)) ORDER BY t.tgname)
 INTO before_triggers FROM pg_trigger t WHERE t.tgrelid='public.lab_arena_runs'::REGCLASS AND NOT t.tgisinternal;
 IF NOT EXISTS(SELECT 1 FROM pg_trigger t JOIN pg_proc f ON f.oid=t.tgfoid
  WHERE t.tgrelid='public.lab_arena_runs'::REGCLASS AND t.tgname='lab_arena_runs_terminal'
   AND t.tgenabled='O' AND t.tgtype=27 AND NOT t.tgisinternal
   AND t.tgfoid='public.lab_arena_runs_terminal_v1()'::REGPROCEDURE
   AND encode(extensions.digest(pg_get_triggerdef(t.oid),'sha256'),'hex')='d088b4d54b5e639b8f736435a3ea83d5e1509f374b3cc99e433fd61386fbe449'
   AND encode(extensions.digest(pg_get_functiondef(f.oid),'sha256'),'hex')='5f795a01c8ed4bf2280597d1f42404c4df46291886be9e7b1a83d35b0e09ea49'
   AND pg_get_userbyid(f.proowner)='lab_arena_owner' AND NOT f.prosecdef
   AND f.proconfig=ARRAY['search_path=pg_catalog']::TEXT[])
  OR encode(extensions.digest(pg_get_functiondef('public.lab_arena_rounds_write_once_v1()'::REGPROCEDURE),'sha256'),'hex')<>'fc99c04a17205b97c55f522b445cd6533847eaaf48d2807ebd05e55ea9de5d25' THEN
  RAISE EXCEPTION 'Recovery633 immutable guards differ'; END IF;
 FOR target IN SELECT value FROM jsonb_array_elements(manifest->'targets') LOOP
  SELECT * INTO STRICT parent FROM public.lab_arena_runs WHERE run_id=target->>'run_id';
  SELECT * INTO STRICT source FROM public.lab_arena_submissions WHERE submission_id=parent.submission_id;
  IF parent.round_id<>r.round_id OR parent.kind<>'execute' OR parent.stage<>2 OR parent.status<>'failed'
   OR parent.result_doc IS NOT NULL OR parent.output_ref IS NOT NULL OR parent.per_icp_score IS NOT NULL
   OR parent.qualification_doc IS NOT NULL OR parent.champion_restart_required
   OR parent.run_id<>parent.assignment_id||':'||parent.attempt::TEXT
   OR encode(extensions.digest(to_jsonb(parent)::TEXT,'sha256'),'hex') IS DISTINCT FROM target->>'row_sha256'
   OR source.round_id<>r.round_id OR source.is_king OR source.status<>'frozen' OR source.code_review_status<>'passed'
   OR parent.miner_hotkey<>source.miner_hotkey
   OR encode(extensions.digest(jsonb_build_object('round_id',source.round_id,'submission_id',source.submission_id,
    'miner_hotkey',source.miner_hotkey,'source_ref',source.source_ref,'source_size_bytes',source.source_size_bytes,
    'source_archive_sha256',source.source_archive_sha256)::TEXT,'sha256'),'hex') IS DISTINCT FROM manifest#>>ARRAY['sources',parent.submission_id]
   OR NOT EXISTS(SELECT 1 FROM jsonb_array_elements(r.participants)p
    WHERE p->>'submission_id'=parent.submission_id AND p->>'miner_hotkey'=parent.miner_hotkey AND p->'is_king'='false'::JSONB)
   OR EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.assignment_id=parent.assignment_id
    AND (a.attempt>parent.attempt OR a.status IN('pending','leased','submitted','accepted')))
   OR EXISTS(SELECT 1 FROM jsonb_array_elements(r.stage2_scoring_plan_doc->'work_items')i
    WHERE i->>'submission_id'=parent.submission_id AND i->'icp_position'=to_jsonb(parent.icp_position)) THEN
   RAISE EXCEPTION 'Recovery633 exact failed source differs: %',parent.run_id; END IF;
  IF target->>'mode'='restore_unused' THEN
   IF parent.terminal_cause<>'stage_closed' OR parent.terminal_doc->>'previous_status'<>'pending'
    OR parent.terminal_doc->'infrastructure_incomplete' IS DISTINCT FROM 'true'::JSONB
    OR parent.claim_response IS NOT NULL OR parent.claim_request_id IS NOT NULL OR parent.claim_request_hash IS NOT NULL
    OR parent.runner_hotkey IS NOT NULL OR parent.lease_token_hash IS NOT NULL OR parent.lease_expires_at IS NOT NULL
    OR parent.participation_accepted_at IS NOT NULL
    OR EXISTS(SELECT 1 FROM public.lab_arena_trajectory_events e WHERE e.run_id=parent.run_id)
    OR EXISTS(SELECT 1 FROM public.lab_arena_ledger l WHERE l.run_id=parent.run_id) THEN
    RAISE EXCEPTION 'Recovery633 pending identity was used: %',parent.run_id; END IF;
  ELSIF target->>'mode'='ordinary_retry' THEN
   IF parent.terminal_cause<>'lease_expired' OR parent.attempt<>1
    OR public.lab_arena_execution_attempt_count(parent)<>1
    OR parent.claim_response->>'status'<>'leased' OR parent.lease_generation<1
    OR parent.lease_expires_at IS NULL OR parent.lease_expires_at>=clock_timestamp()
    OR parent.claim_response->'lease_generation' IS DISTINCT FROM to_jsonb(parent.lease_generation) THEN
    RAISE EXCEPTION 'Recovery633 ordinary retry lineage differs'; END IF;
  ELSE RAISE EXCEPTION 'Recovery633 unsupported target mode'; END IF;
  cost:=public.lab_arena__successful_icp_cost_state(r.round_id,parent.submission_id,parent.icp_position);
  IF (cost->>'inflight_calls')::BIGINT IS DISTINCT FROM 0::BIGINT
   OR (cost->>'success_unresolved_calls')::BIGINT IS DISTINCT FROM 0::BIGINT
   OR coalesce(cost->>'competition_settled_microusd',cost->>'settled_microusd')::BIGINT>=
      (r.configuration_doc->>'execution_icp_cap_microusd')::BIGINT THEN
   RAISE EXCEPTION 'Recovery633 unresolved billing or successful-cost cap'; END IF;
  available:=FALSE;
  FOR provider IN SELECT key,value FROM jsonb_each_text(r.configuration_doc->'call_quotas') LOOP
   IF provider.value::BIGINT=0 AND provider.key='deepline'
      AND r.configuration_doc#>>'{deepline_catalog,schema_version}'='leadpoet.lab_arena.deepline_catalog.v1' THEN available:=TRUE;
   ELSIF provider.value::BIGINT>0
    AND public.lab_arena__execute_assignment_calls_v1(parent.assignment_id,provider.key)<provider.value::BIGINT*2
    AND public.lab_arena__submission_stage_consumed(parent.submission_id,2::SMALLINT,provider.key,'execute')<
      provider.value::BIGINT*2*(r.configuration_doc->>'stage_2_icp_count')::BIGINT THEN available:=TRUE;
   END IF;
  END LOOP;
  IF NOT available THEN RAISE EXCEPTION 'Recovery633 original quota exhausted'; END IF;
 END LOOP;
 SELECT jsonb_object_agg(a.run_id,to_jsonb(a)) INTO originals FROM public.lab_arena_runs a
 WHERE a.run_id IN(SELECT t->>'run_id' FROM jsonb_array_elements(manifest->'targets')t);
 n:=r;n.configuration_doc:=expected_config;n.status:='stage2';n.status_generation:=r.status_generation+1;
 n.stage2_scoring_plan_doc:=NULL; -- Archived below; normal builder merges old and new accepted sources.
 SELECT coalesce(max(generation_floor),0)+1 INTO revision FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,
  transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(r.round_id,revision,'nov05-approved-infrastructure633-v1',request_hash,'nov05-user-approved-infrastructure633',
  txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(n)-'updated_at','{}',
  jsonb_build_object('operation','nov05_infrastructure_recovery633_v1','scoring_generation_floor',0,
   'manifest',manifest,'original_runs',originals,'old_stage2_scoring_plan',r.stage2_scoring_plan_doc));
 UPDATE public.lab_arena_rounds SET configuration_doc=n.configuration_doc,status=n.status,
  status_generation=n.status_generation,stage2_scoring_plan_doc=NULL WHERE round_id=r.round_id;
 ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal;
 UPDATE public.lab_arena_runs a SET status='pending',terminal_cause=NULL,terminal_doc=NULL,stage_generation=r.stage_generation
 FROM jsonb_array_elements(manifest->'targets')t WHERE a.run_id=t->>'run_id' AND t->>'mode'='restore_unused';
 GET DIAGNOSTICS restored=ROW_COUNT;
 ALTER TABLE public.lab_arena_runs ENABLE TRIGGER lab_arena_runs_terminal;
 INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,
  attempt,status,lease_generation,stage_generation,kind,previous_runner_hotkey)
 SELECT a.assignment_id||':2',a.assignment_id,a.round_id,a.submission_id,a.miner_hotkey,a.stage,a.icp_position,
  2,'pending',a.lease_generation,r.stage_generation,'execute',a.runner_hotkey
 FROM public.lab_arena_runs a,jsonb_array_elements(manifest->'targets')t
 WHERE a.run_id=t->>'run_id' AND t->>'mode'='ordinary_retry';
 GET DIAGNOSTICS created=ROW_COUNT;
 SELECT array_agg(t->>'run_id') INTO known_ids FROM jsonb_array_elements(manifest->'targets')t;
 SELECT array_agg(regexp_replace(t->>'run_id',':1$',':2')) INTO successor_ids
 FROM jsonb_array_elements(manifest->'targets')t WHERE t->>'mode'='ordinary_retry';
 SELECT jsonb_agg(jsonb_build_array(t.tgname,t.tgenabled,pg_get_triggerdef(t.oid),pg_get_functiondef(t.tgfoid)) ORDER BY t.tgname)
 INTO after_triggers FROM pg_trigger t WHERE t.tgrelid='public.lab_arena_runs'::REGCLASS AND NOT t.tgisinternal;
 IF restored<>36 OR created<>14 OR before_triggers IS DISTINCT FROM after_triggers
  OR (SELECT count(*) FROM public.lab_arena_runs a WHERE a.run_id=ANY(known_ids) AND a.status='pending'
    AND a.terminal_cause IS NULL AND a.terminal_doc IS NULL AND a.stage_generation=r.stage_generation)<>36
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.round_id=r.round_id AND NOT a.run_id=ANY(successor_ids)
   AND (CASE WHEN a.run_id=ANY(known_ids) AND manifest->'targets' @>jsonb_build_array(jsonb_build_object('run_id',a.run_id,'mode','restore_unused'))
    THEN (to_jsonb(a)-ARRAY['status','terminal_cause','terminal_doc','stage_generation','updated_at']) IS DISTINCT FROM
      ((before_runs->a.run_id)-ARRAY['status','terminal_cause','terminal_doc','stage_generation','updated_at'])
    ELSE to_jsonb(a) IS DISTINCT FROM before_runs->a.run_id END))
  OR (SELECT jsonb_object_agg(a.run_id,to_jsonb(a)) FROM public.lab_arena_runs a WHERE a.round_id=r.round_id AND a.status='accepted') IS DISTINCT FROM before_accepted
  OR (SELECT jsonb_object_agg(s.submission_id,to_jsonb(s)) FROM public.lab_arena_submissions s WHERE s.round_id=r.round_id) IS DISTINCT FROM before_submissions
  OR (SELECT jsonb_object_agg(a.round_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex')) FROM public.lab_arena_rounds a WHERE a.round_id<>r.round_id) IS DISTINCT FROM before_rounds
  OR (SELECT to_jsonb(a)-'updated_at' FROM public.lab_arena_rounds a WHERE a.round_id=r.round_id) IS DISTINCT FROM to_jsonb(n)-'updated_at'
  OR clock_timestamp()>='2026-11-05T20:30:00Z'::TIMESTAMPTZ THEN
  RAISE EXCEPTION 'Recovery633 exact readback differs'; END IF;
END;
$data633$;
NOTIFY pgrst,'reload schema';
COMMIT;
