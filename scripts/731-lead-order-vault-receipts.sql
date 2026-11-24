-- Lead-order key vault receipts (docs/gamma-lead-order-vault.md, "Receipts").
-- Each call the vault forwards on a miner's key returns a signed receipt
-- with a per-key sequence number. The gateway keeps every one, for the call
-- or the free cost read-back it belongs to, and shows a miner their own, so
-- a gap in the sequence is a call Leadpoet did not report.
-- Adds one append-only table and two functions; changes nothing existing.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
CREATE TEMP TABLE lead_orders731_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $prereq731$
BEGIN
 IF to_regclass('public.lead_order_calls') IS NULL
  OR to_regprocedure('public.lab_arena_append_only_v1()') IS NULL THEN
  RAISE EXCEPTION 'lead order vault receipts need lead orders (707)' USING ERRCODE='55000';
 END IF;
END;
$prereq731$;

-- One row per receipt. Its own name (key, vault boot, sequence) is the key,
-- so recording the same receipt twice is a no-op.
CREATE TABLE public.lead_order_vault_receipts (
  sequence BIGSERIAL NOT NULL UNIQUE,
  key_fingerprint TEXT NOT NULL CHECK(key_fingerprint ~ '^sha256:[0-9a-f]{64}$'),
  boot_id TEXT NOT NULL CHECK(length(boot_id) BETWEEN 1 AND 128),
  key_sequence BIGINT NOT NULL CHECK(key_sequence>0),
  call_id UUID NOT NULL REFERENCES public.lead_order_calls(call_id),
  kind TEXT NOT NULL CHECK(kind IN ('call','readback')),
  miner_hotkey TEXT NOT NULL,
  receipt JSONB NOT NULL CHECK(jsonb_typeof(receipt)='object' AND octet_length(receipt::TEXT)<=8192),
  recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY(key_fingerprint,boot_id,key_sequence)
);
CREATE INDEX lead_order_vault_receipts_hotkey ON public.lead_order_vault_receipts(miner_hotkey,sequence);
-- A call has one receipt, so a hidden call's receipt cannot be filed under it.
CREATE UNIQUE INDEX lead_order_vault_receipts_one_per_call ON public.lead_order_vault_receipts(call_id)
 WHERE kind='call';

ALTER TABLE public.lead_order_vault_receipts OWNER TO lab_arena_owner;
ALTER SEQUENCE public.lead_order_vault_receipts_sequence_seq OWNER TO lab_arena_owner;
ALTER TABLE public.lead_order_vault_receipts ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.lead_order_vault_receipts FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
CREATE TRIGGER lead_order_vault_receipts_append_only BEFORE UPDATE OR DELETE ON public.lead_order_vault_receipts
 FOR EACH ROW EXECUTE FUNCTION public.lab_arena_append_only_v1();
CREATE TRIGGER lead_order_vault_receipts_no_truncate BEFORE TRUNCATE ON public.lead_order_vault_receipts
 FOR EACH STATEMENT EXECUTE FUNCTION public.lab_arena_append_only_v1();

-- The receipt must name the call's own miner and key, and its path must fit
-- its kind: a read-back is the free generation lookup, a call never is. Its
-- signature is checked by the miner, who trusts the vault's key, not this table.
CREATE FUNCTION public.lead_order_record_vault_receipt_v1(p_call_id UUID,p_kind TEXT,p_receipt JSONB)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $record731$
DECLARE hotkey TEXT; fingerprint TEXT; doc JSONB := p_receipt->'document';
BEGIN
 SELECT c.miner_hotkey,k.internal_fingerprint INTO hotkey,fingerprint
  FROM public.lead_order_calls c JOIN public.lead_order_keys k ON k.key_id=c.key_id WHERE c.call_id=p_call_id;
 IF hotkey IS NULL OR p_kind IS NULL OR p_kind NOT IN ('call','readback') OR jsonb_typeof(doc) IS DISTINCT FROM 'object'
  OR doc->>'miner_hotkey' IS DISTINCT FROM hotkey OR doc->>'key_fingerprint' IS DISTINCT FROM fingerprint
  -- A readback is the one GET route the vault allows (vault.OPENROUTER_ROUTES).
  OR ((doc->>'method')='GET' AND (doc->>'path')='/api/v1/generation') IS DISTINCT FROM (p_kind='readback')
  OR length(coalesce(doc->>'boot_id','')) NOT BETWEEN 1 AND 128
  OR jsonb_typeof(doc->'key_sequence') IS DISTINCT FROM 'number' OR doc->>'key_sequence' !~ '^[1-9][0-9]{0,17}$'
  OR octet_length(p_receipt::TEXT)>8192 THEN
  RETURN jsonb_build_object('status','not_recordable');
 END IF;
 IF p_kind='call' AND EXISTS(SELECT 1 FROM public.lead_order_vault_receipts r
   WHERE r.call_id=p_call_id AND r.kind='call' AND (r.boot_id,r.key_sequence)<>(doc->>'boot_id',(doc->>'key_sequence')::BIGINT)) THEN
  RETURN jsonb_build_object('status','not_recordable');
 END IF;
 INSERT INTO public.lead_order_vault_receipts(key_fingerprint,boot_id,key_sequence,call_id,kind,miner_hotkey,receipt)
 VALUES(doc->>'key_fingerprint',doc->>'boot_id',(doc->>'key_sequence')::BIGINT,p_call_id,p_kind,hotkey,p_receipt)
 ON CONFLICT DO NOTHING;
 IF FOUND THEN
  RETURN jsonb_build_object('status','recorded');
 END IF;
 -- A replay of the same receipt for the same call; any other conflict is a
 -- receipt this table did not keep, which the broker logs.
 RETURN jsonb_build_object('status',CASE WHEN EXISTS(SELECT 1 FROM public.lead_order_vault_receipts r
   WHERE r.key_fingerprint=doc->>'key_fingerprint' AND r.boot_id=doc->>'boot_id'
     AND r.key_sequence=(doc->>'key_sequence')::BIGINT AND r.call_id=p_call_id AND r.kind=p_kind
     AND r.receipt=p_receipt) THEN 'exists' ELSE 'not_recordable' END);
END;
$record731$;

-- A miner's receipts after ``p_since`` (this table's sequence), plus the
-- envelope of each of their active keys; the service reads from it which
-- vault key it is sealed to, so the miner can see which need sealing again.
CREATE FUNCTION public.lead_order_vault_receipts_v1(p_hotkey TEXT,p_since BIGINT,p_limit INTEGER)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $receipts731$
 SELECT jsonb_build_object(
  'receipts',(SELECT coalesce(jsonb_agg(jsonb_build_object('sequence',r.sequence,'call_id',r.call_id,'kind',r.kind,
     'receipt',r.receipt) ORDER BY r.sequence),'[]'::JSONB)
   FROM (SELECT * FROM public.lead_order_vault_receipts WHERE miner_hotkey=p_hotkey AND sequence>greatest(p_since,0)
         ORDER BY sequence LIMIT least(greatest(p_limit,1),500)) r),
  'keys',(SELECT coalesce(jsonb_agg(jsonb_build_object('key_id',k.key_id,'provider',k.provider,
     'key_fingerprint',k.internal_fingerprint,'envelope',k.ciphertext_b64) ORDER BY k.sequence),'[]'::JSONB)
   FROM public.lead_order_keys k
   WHERE k.miner_hotkey=p_hotkey
     AND NOT EXISTS(SELECT 1 FROM public.lead_order_key_revocations v WHERE v.key_id=k.key_id)));
$receipts731$;

DO $functions731$
DECLARE f REGPROCEDURE;
BEGIN
 FOREACH f IN ARRAY ARRAY['public.lead_order_record_vault_receipt_v1(uuid,text,jsonb)'::REGPROCEDURE,
   'public.lead_order_vault_receipts_v1(text,bigint,integer)'::REGPROCEDURE] LOOP
  EXECUTE format('ALTER FUNCTION %s OWNER TO lab_arena_owner',f);
  EXECUTE format('REVOKE ALL ON FUNCTION %s FROM PUBLIC,anon,authenticated,service_role,lab_arena_service',f);
  EXECUTE format('GRANT EXECUTE ON FUNCTION %s TO lab_arena_service',f);
 END LOOP;
END;
$functions731$;

DO $restore731$
BEGIN
 IF NOT(SELECT had_create FROM pg_temp.lead_orders731_acl)THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;END IF;
 IF(SELECT nspacl FROM pg_namespace WHERE nspname='public')IS DISTINCT FROM(SELECT nspacl FROM pg_temp.lead_orders731_acl)THEN
  RAISE EXCEPTION 'lead_orders731 schema ACL differs';END IF;
END;
$restore731$;
COMMIT;
