-- Exact October 29 operator exception. Training eligibility remains false.
-- The unchanged Arena service must perform Git promotion and reward signing.
-- The archive preserves the original publication and signed basis for old epochs.
BEGIN;
SET LOCAL TIME ZONE 'UTC';
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

CREATE TEMP TABLE oct29_schema592 ON COMMIT DROP AS
  SELECT has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create;

DO $oct29_champion592$
DECLARE
  r public.lab_arena_rounds;
  a public.lab_arena_rounds;
  s public.lab_arena_submissions;
  archive_id CONSTANT TEXT := 'arena-2026-10-29-rewardhistory592';
  corrected_publication JSONB;
  original_triggers JSONB;
  insert_columns TEXT;
  archive_doc JSONB;
BEGIN
  PERFORM pg_advisory_xact_lock(hashtextextended('lab_arena.baseline_promotion', 0));
  PERFORM pg_advisory_xact_lock(hashtextextended('lab_arena.reward_epoch', 0));
  LOCK TABLE public.lab_arena_rounds IN SHARE ROW EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_submissions IN SHARE MODE;
  SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id='arena-2026-10-29' FOR UPDATE;
  SELECT * INTO STRICT s FROM public.lab_arena_submissions
    WHERE submission_id='sub-cb3105f830853c1452b889ab5cadee7e';
  SELECT * INTO a FROM public.lab_arena_rounds WHERE round_id=archive_id;

  IF encode(extensions.digest(to_jsonb(s)::TEXT,'sha256'),'hex') <> '17653ea9ba6b7abfe6d90f56a2e06b9721265424683a231d8c5188648a33a52d'
    OR s.round_id IS DISTINCT FROM r.round_id
    OR s.miner_hotkey IS DISTINCT FROM '5HSutSjNe6Mpfxjvifv7y2p4HbkTenVhZ75rRgHRdmSrXKu9'
    OR s.status IS DISTINCT FROM 'frozen' OR s.is_king
    OR s.source_archive_sha256 IS DISTINCT FROM 'sha256:64c963c4f9e39cf51a2b3492e9fe46d7bfb770a097620febe927400ebdfd4ec8'
    OR s.source_normalized_sha256 IS DISTINCT FROM 'sha256:99ce4041e7d25fe6d9620d138b9e89f016c886f06df8371ddc65cd9a6099999c'
    OR s.source_size_bytes IS DISTINCT FROM 458335 THEN
    RAISE EXCEPTION 'Oct29 champion submission differs' USING ERRCODE='55000';
  END IF;
  -- Idempotent after preparation, canonical promotion, and canonical activation.
  IF a.round_id IS NOT NULL THEN
    IF r.status IS DISTINCT FROM 'published'
      OR encode(extensions.digest(r.publication_doc::TEXT,'sha256'),'hex') <> '6edcfdaab11785f386d52881b95db08a27cac9e555c8d867c6b33889d77000c1'
      OR r.king_outcome IS DISTINCT FROM 'crowned'
      OR r.king_hotkey IS DISTINCT FROM s.miner_hotkey
      OR encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex') <> 'a4f2d15f0b9c0d640ea93bc516001bfd73ac3b238112fe32a6526f6f38df01b4'
      OR a.status IS DISTINCT FROM 'cancelled'
      OR a.cancel_reason IS DISTINCT FROM 'authorized_oct29_champion_reward_archive592'
      OR encode(extensions.digest(a.publication_doc::TEXT,'sha256'),'hex') <> '5ea1629e523ba39591946043ef283824c182a9022892ac3f69471dedf62a72e3'
      OR a.reward_basis_hash IS DISTINCT FROM 'sha256:fc38359b4a7c89939d6b927420b266d3c8b76483451291cc4808b875056f2009'
      OR a.reward_basis_doc->>'reward_basis_hash' IS DISTINCT FROM a.reward_basis_hash
      OR a.effective_reward_epoch IS DISTINCT FROM 26078
      OR a.configuration_doc->>'recovery_source_round_id' IS DISTINCT FROM r.round_id
      OR (a.configuration_doc-'recovery_source_round_id'-'authorized_champion_override')
         IS DISTINCT FROM jsonb_set(r.configuration_doc,'{round_id}',to_jsonb(archive_id))
      OR a.signing_key_doc IS NULL OR a.reward_activated_at IS NULL THEN
      RAISE EXCEPTION 'Oct29 champion replay differs' USING ERRCODE='55000';
    END IF;
    RETURN;
  END IF;
  IF encode(extensions.digest(to_jsonb(r)::TEXT,'sha256'),'hex') <> '834b65d9ea2e84ac5c89872a2f5e2b40fd46dc105b80c5b34e8aed6c22257fb8'
    OR r.status IS DISTINCT FROM 'published' OR NOT r.rewards_enabled OR NOT r.promotion_required
    OR r.arena_network_name IS DISTINCT FROM 'finney' OR r.arena_netuid IS DISTINCT FROM 71
    OR r.configuration_doc->>'mode' IS DISTINCT FROM 'live'
    OR r.evaluation_date IS DISTINCT FROM '2026-10-29'
    OR r.king_outcome IS DISTINCT FROM 'no_king' OR r.king_hotkey IS NOT NULL
    OR r.promotion_doc IS NOT NULL OR r.baseline_promoted_at IS NOT NULL
    OR r.reward_basis_hash IS DISTINCT FROM 'sha256:fc38359b4a7c89939d6b927420b266d3c8b76483451291cc4808b875056f2009'
    OR r.effective_reward_epoch IS DISTINCT FROM 26078
    OR encode(extensions.digest(r.publication_doc::TEXT,'sha256'),'hex') <> '5ea1629e523ba39591946043ef283824c182a9022892ac3f69471dedf62a72e3'
    OR EXISTS (SELECT 1 FROM public.lab_arena_rounds newer
      WHERE newer.status='published' AND newer.configuration_doc->>'mode'='live'
        AND newer.arena_network_name=r.arena_network_name AND newer.arena_netuid=r.arena_netuid
        AND newer.evaluation_date>r.evaluation_date)
    OR NOT public.lab_arena__baseline_final_ready_v1(r.round_id) THEN
    RAISE EXCEPTION 'Oct29 champion preimage differs' USING ERRCODE='55000';
  END IF;
  SELECT jsonb_agg(jsonb_build_array(t.tgname,t.tgenabled,pg_get_triggerdef(t.oid)) ORDER BY t.tgname)
    INTO original_triggers FROM pg_trigger t
    WHERE t.tgrelid='public.lab_arena_rounds'::regclass AND NOT t.tgisinternal;
  IF EXISTS (SELECT 1 FROM pg_trigger WHERE tgrelid='public.lab_arena_rounds'::regclass
      AND NOT tgisinternal AND tgenabled<>'O') THEN
    RAISE EXCEPTION 'Oct29 champion trigger state differs' USING ERRCODE='55000';
  END IF;
  corrected_publication := jsonb_set(r.publication_doc,'{king_decision}',
    jsonb_build_object('outcome','crowned','king_hotkey',s.miner_hotkey,
      'king_submission_id',s.submission_id,'winner_submission_id',s.submission_id,
      'operator_override',jsonb_build_object('reason','authorized_oct29_training_evidence_exception',
        'migration',592,'original_publication_sha256','5ea1629e523ba39591946043ef283824c182a9022892ac3f69471dedf62a72e3')));
  IF encode(extensions.digest(corrected_publication::TEXT,'sha256'),'hex') <> '6edcfdaab11785f386d52881b95db08a27cac9e555c8d867c6b33889d77000c1'
    OR corrected_publication-'king_decision' IS DISTINCT FROM r.publication_doc-'king_decision' THEN
    RAISE EXCEPTION 'Oct29 corrected publication differs' USING ERRCODE='55000';
  END IF;
  archive_doc := to_jsonb(r) || jsonb_build_object('round_id',archive_id,'status','cancelled',
    'promotion_required',FALSE,'cancel_reason','authorized_oct29_champion_reward_archive592',
    'configuration_doc',r.configuration_doc || jsonb_build_object('round_id',archive_id,
      'recovery_source_round_id',r.round_id,'authorized_champion_override',
      corrected_publication#>'{king_decision,operator_override}'));
  SELECT string_agg(quote_ident(attname),',' ORDER BY attnum) INTO insert_columns
    FROM pg_attribute WHERE attrelid='public.lab_arena_rounds'::regclass
      AND attnum>0 AND NOT attisdropped AND attgenerated='';
  -- Owner-only, transaction-bounded exception to immutable publication fields.
  -- No callable recovery RPC or alternate promotion/signing implementation.
  ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER;
  UPDATE public.lab_arena_rounds SET effective_reward_epoch=NULL,king_start_epoch=NULL,
    reward_basis_hash=NULL,reward_basis_doc=NULL,signing_key_doc=NULL,reward_activated_at=NULL,
    publication_doc=corrected_publication,king_outcome='crowned',king_hotkey=s.miner_hotkey
    WHERE round_id=r.round_id;
  EXECUTE format('INSERT INTO public.lab_arena_rounds(%s) SELECT %s FROM jsonb_populate_record(NULL::public.lab_arena_rounds,$1)',
    insert_columns,insert_columns) USING archive_doc;
  ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER;
  IF (SELECT jsonb_agg(jsonb_build_array(t.tgname,t.tgenabled,pg_get_triggerdef(t.oid)) ORDER BY t.tgname)
      FROM pg_trigger t WHERE t.tgrelid='public.lab_arena_rounds'::regclass AND NOT t.tgisinternal)
      IS DISTINCT FROM original_triggers
    OR (SELECT to_jsonb(current)-ARRAY['publication_doc','king_outcome','king_hotkey','effective_reward_epoch',
      'king_start_epoch','reward_basis_hash','reward_basis_doc','signing_key_doc','reward_activated_at']
      FROM public.lab_arena_rounds current WHERE round_id=r.round_id)
      IS DISTINCT FROM to_jsonb(r)-ARRAY['publication_doc','king_outcome','king_hotkey','effective_reward_epoch',
      'king_start_epoch','reward_basis_hash','reward_basis_doc','signing_key_doc','reward_activated_at']
    OR (SELECT reward_basis_doc FROM public.lab_arena_rounds WHERE round_id=archive_id)
       IS DISTINCT FROM r.reward_basis_doc
    OR (SELECT signing_key_doc FROM public.lab_arena_rounds WHERE round_id=archive_id)
       IS DISTINCT FROM r.signing_key_doc
    OR (SELECT encode(extensions.digest(to_jsonb(archived)::TEXT,'sha256'),'hex')
        FROM public.lab_arena_rounds archived WHERE round_id=archive_id) <> 'a4f2d15f0b9c0d640ea93bc516001bfd73ac3b238112fe32a6526f6f38df01b4' THEN
    RAISE EXCEPTION 'Oct29 champion postimage differs' USING ERRCODE='55000';
  END IF;
END;
$oct29_champion592$;

-- Retain only this exact audited achievement in normal historical selection.
DO $oct29_snapshot592$
DECLARE
  v_signature REGPROCEDURE := 'public.lab_arena_reward_slot_snapshot(text,jsonb)'::regprocedure;
  definition TEXT;
  updated TEXT;
  identity JSONB;
  anchor CONSTANT TEXT := $old$      AND winner.entry -> 'eligible' = 'true'::JSONB$old$;
  replacement CONSTANT TEXT := $new$      AND (winner.entry -> 'eligible' = 'true'::JSONB
        OR (c.round_id = 'arena-2026-10-29'
          AND c.winner_id = 'sub-cb3105f830853c1452b889ab5cadee7e'
          AND c.winner_hotkey = '5HSutSjNe6Mpfxjvifv7y2p4HbkTenVhZ75rRgHRdmSrXKu9'
          AND c.evaluation_date = '2026-10-29'
          AND winner.entry -> 'eligible' = 'false'::JSONB
          AND winner.entry ->> 'eligibility_reason' = 'training_data_incomplete'
          AND encode(extensions.digest(c.publication_doc::TEXT,'sha256'),'hex') = '6edcfdaab11785f386d52881b95db08a27cac9e555c8d867c6b33889d77000c1'))$new$;
BEGIN
  SELECT pg_get_functiondef(p.oid),jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
    INTO definition,identity FROM pg_proc p WHERE p.oid=v_signature;
  IF identity IS DISTINCT FROM '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"]]'::JSONB THEN
    RAISE EXCEPTION 'Oct29 snapshot authority differs' USING ERRCODE='55000';
  END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex')='0870c5ab466388cdc89b8396e7565e547f4278b2870ea18c3e8bd2bb4dfca5c6' THEN RETURN; END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex')<>'670235bc84b77ebc11ca69be3729a1bdb430bb5119a32021db862e90f453f499'
    OR length(definition)-length(replace(definition,anchor,''))<>length(anchor) THEN
    RAISE EXCEPTION 'Oct29 snapshot preimage differs' USING ERRCODE='55000';
  END IF;
  updated:=replace(definition,anchor,replacement);
  IF encode(extensions.digest(updated,'sha256'),'hex')<>'0870c5ab466388cdc89b8396e7565e547f4278b2870ea18c3e8bd2bb4dfca5c6' THEN
    RAISE EXCEPTION 'Oct29 snapshot postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE updated;
  IF (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
      FROM pg_proc p WHERE p.oid=v_signature) IS DISTINCT FROM identity THEN
    RAISE EXCEPTION 'Oct29 snapshot permissions changed' USING ERRCODE='55000';
  END IF;
END;
$oct29_snapshot592$;

-- A stale chain read must not place the replacement in an already issued epoch.
DO $temporary_schema592$
BEGIN
  IF NOT (SELECT had_create FROM pg_temp.oct29_schema592) THEN
    GRANT CREATE ON SCHEMA public TO lab_arena_owner;
  END IF;
END;
$temporary_schema592$;
CREATE OR REPLACE FUNCTION public.lab_arena_oct29_reward_guard592()
RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
AS $oct29_guard592$
DECLARE
  prior public.lab_arena_rounds;
  slot JSONB;
BEGIN
  IF NEW.round_id<>'arena-2026-10-29' OR OLD.reward_activated_at IS NOT NULL
    OR NEW.reward_activated_at IS NULL THEN RETURN NEW; END IF;
  PERFORM pg_advisory_xact_lock(hashtextextended('lab_arena.reward_epoch',0));
  SELECT * INTO STRICT prior FROM public.lab_arena_rounds
    WHERE round_id='arena-2026-10-29-rewardhistory592';
  slot:=NEW.reward_basis_doc#>'{reward_slots,2}';
  IF encode(extensions.digest(NEW.publication_doc::TEXT,'sha256'),'hex')<>'6edcfdaab11785f386d52881b95db08a27cac9e555c8d867c6b33889d77000c1'
    OR NEW.baseline_promoted_at IS NULL OR NEW.promotion_doc IS NULL
    OR NEW.effective_reward_epoch<=prior.effective_reward_epoch
    OR NEW.reward_basis_doc->>'king_hotkey' IS DISTINCT FROM '5HSutSjNe6Mpfxjvifv7y2p4HbkTenVhZ75rRgHRdmSrXKu9'
    OR NEW.reward_basis_doc->>'king_outcome' IS DISTINCT FROM 'crowned'
    OR NEW.reward_basis_doc->'slot_policy' IS DISTINCT FROM prior.reward_basis_doc->'slot_policy'
    OR NEW.reward_basis_doc#>'{reward_slots,0}' IS DISTINCT FROM prior.reward_basis_doc#>'{reward_slots,0}'
    OR NEW.reward_basis_doc#>'{reward_slots,1}' IS DISTINCT FROM prior.reward_basis_doc#>'{reward_slots,1}'
    OR slot IS DISTINCT FROM '{"round_id":"arena-2026-10-29","submission_id":"sub-cb3105f830853c1452b889ab5cadee7e","miner_hotkey":"5HSutSjNe6Mpfxjvifv7y2p4HbkTenVhZ75rRgHRdmSrXKu9","baseline_submission_id":"baseline-2026-10-29","baseline_score":8.55,"winner_score":20.275,"start_epoch":null}'::JSONB
    OR EXISTS (SELECT 1 FROM public.lab_arena_accepted_weight_states
      WHERE network=OLD.arena_network_name AND netuid=OLD.arena_netuid AND epoch>=NEW.effective_reward_epoch)
    OR EXISTS (SELECT 1 FROM public.lab_arena_chain_outcomes
      WHERE network=OLD.arena_network_name AND netuid=OLD.arena_netuid AND epoch>=NEW.effective_reward_epoch) THEN
    RAISE EXCEPTION 'Oct29 replacement reward conflicts with immutable history' USING ERRCODE='55000';
  END IF;
  RETURN NEW;
END;
$oct29_guard592$;
ALTER FUNCTION public.lab_arena_oct29_reward_guard592() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_oct29_reward_guard592()
  FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
DROP TRIGGER IF EXISTS lab_arena_oct29_reward_guard592 ON public.lab_arena_rounds;
CREATE TRIGGER lab_arena_oct29_reward_guard592 BEFORE UPDATE ON public.lab_arena_rounds
  FOR EACH ROW EXECUTE FUNCTION public.lab_arena_oct29_reward_guard592();
-- Serialize new issued evidence with the same reward lock in both orderings.
-- Reject only the archived October 29 basis after its replacement takes effect.
CREATE OR REPLACE FUNCTION public.lab_arena_oct29_issued_guard592()
RETURNS trigger LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
AS $oct29_issued592$
DECLARE active public.lab_arena_rounds; archived_hash TEXT;
BEGIN
  IF NEW.network<>'finney' OR NEW.netuid<>71 THEN RETURN NEW; END IF;
  PERFORM pg_advisory_xact_lock(hashtextextended('lab_arena.reward_epoch',0));
  SELECT * INTO active FROM public.lab_arena_rounds WHERE round_id='arena-2026-10-29';
  IF active.reward_activated_at IS NULL OR NEW.epoch<active.effective_reward_epoch THEN RETURN NEW; END IF;
  SELECT reward_basis_hash INTO archived_hash FROM public.lab_arena_rounds
    WHERE round_id='arena-2026-10-29-rewardhistory592';
  IF archived_hash IS NULL THEN
    RAISE EXCEPTION 'Oct29 issued history archive missing' USING ERRCODE='55000';
  END IF;
  IF TG_TABLE_NAME='lab_arena_accepted_weight_states' THEN
    IF NEW.state_doc#>>'{reward_basis,reward_basis_hash}'=archived_hash THEN
      RAISE EXCEPTION 'Oct29 archived basis cannot acquire a replacement epoch' USING ERRCODE='55000';
    END IF;
  ELSIF NOT EXISTS (SELECT 1 FROM public.lab_arena_accepted_weight_states accepted
    WHERE accepted.network=NEW.network AND accepted.netuid=NEW.netuid AND accepted.epoch=NEW.epoch
      AND accepted.state_hash=NEW.outcome_doc->>'state_hash'
      AND accepted.state_doc#>>'{reward_basis,reward_basis_hash}' IS DISTINCT FROM archived_hash) THEN
    RAISE EXCEPTION 'Oct29 replacement chain outcome requires its accepted state' USING ERRCODE='55000';
  END IF;
  RETURN NEW;
END;
$oct29_issued592$;
ALTER FUNCTION public.lab_arena_oct29_issued_guard592() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_oct29_issued_guard592()
  FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
DO $issued_triggers592$
DECLARE name TEXT; relation REGCLASS;
BEGIN
  FOR name,relation IN VALUES
    ('lab_arena_oct29_accepted_guard592','public.lab_arena_accepted_weight_states'::regclass),
    ('lab_arena_oct29_chain_guard592','public.lab_arena_chain_outcomes'::regclass)
  LOOP
    IF EXISTS(SELECT 1 FROM pg_trigger t WHERE t.tgrelid=relation AND t.tgname=name) THEN
      IF NOT EXISTS(SELECT 1 FROM pg_trigger t WHERE t.tgrelid=relation AND t.tgname=name
        AND t.tgenabled='O' AND t.tgtype=7
        AND t.tgqual IS NULL AND t.tgnargs=0 AND NOT t.tgisinternal
        AND t.tgfoid='public.lab_arena_oct29_issued_guard592()'::regprocedure) THEN
        RAISE EXCEPTION 'Oct29 issued trigger differs' USING ERRCODE='55000';
      END IF;
    ELSE
      EXECUTE format('CREATE TRIGGER %I BEFORE INSERT ON %s FOR EACH ROW EXECUTE FUNCTION public.lab_arena_oct29_issued_guard592()',name,relation);
    END IF;
  END LOOP;
END;
$issued_triggers592$;
DO $restore_schema592$
BEGIN
  IF NOT (SELECT had_create FROM pg_temp.oct29_schema592) THEN
    REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
  END IF;
END;
$restore_schema592$;
NOTIFY pgrst,'reload schema';
COMMIT;
