-- Gamma tool market (docs/gamma-tool-market.md). Off by default: nothing
-- calls these functions unless LEADPOET_TOOL_MARKET_MODE is live. Miners
-- list vendor keys sealed to the key vault with a price per unit; buyers open
-- a tab against their Gamma balance and each call goes to the cheapest
-- eligible key. Every table is append-only (no UPDATE, DELETE or TRUNCATE):
-- state moves forward by inserting asks, events, closures and settlements.
-- RLS is on with no policy and no table grant; the Arena service role reaches
-- rows only through the SECURITY DEFINER functions below, as 707 does.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
CREATE TEMP TABLE tool_market770_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $prereq770$
BEGIN
 IF to_regprocedure('public.lab_arena_append_only_v1()') IS NULL
  OR to_regrole('lab_arena_owner') IS NULL OR to_regrole('lab_arena_service') IS NULL THEN
  RAISE EXCEPTION 'the tool market needs the Lab Arena roles and append-only trigger (179)' USING ERRCODE='55000';
 END IF;
END;
$prereq770$;

-- ---------------------------------------------------------------------------
-- Tables
-- ---------------------------------------------------------------------------

-- One cumulative Gamma namespace funds this ledger: the latest row. Startup
-- may repoint it only while no reading or tab exists, so a mistyped service
-- can be corrected before any credit depends on it.
CREATE TABLE public.tool_market_gamma_source (
  source_id BIGSERIAL PRIMARY KEY,
  genesis TEXT NOT NULL CHECK(genesis ~ '^0x[0-9a-f]{64}$'),
  netuid INTEGER NOT NULL CHECK(netuid BETWEEN 1 AND 65535),
  service TEXT NOT NULL CHECK(service ~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'),
  registered_at BIGINT NOT NULL CHECK(registered_at>0)
);

CREATE TABLE public.tool_market_gamma_readings (
  reading_id BIGSERIAL PRIMARY KEY,
  hotkey TEXT NOT NULL CHECK(hotkey ~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'),
  block_number BIGINT NOT NULL CHECK(block_number>=0),
  block_hash TEXT NOT NULL CHECK(block_hash ~ '^0x[0-9a-f]{64}$'),
  spent_units NUMERIC(39,0) NOT NULL CHECK(spent_units>=0),
  available BOOLEAN NOT NULL,
  read_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX tool_market_gamma_readings_hotkey ON public.tool_market_gamma_readings(hotkey,block_number DESC,reading_id DESC);

CREATE TABLE public.tool_market_nonces (
  hotkey TEXT NOT NULL,
  signature_hash TEXT NOT NULL CHECK(signature_hash ~ '^sha256:[0-9a-f]{64}$'),
  signed_at BIGINT NOT NULL,
  used_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY(hotkey,signature_hash)
);

-- A signed acceptance of the terms text whose sha256 the gateway is
-- configured with. ``signed_body``, ``host``, ``signed_at`` and
-- ``signature`` let anyone verify it again.
CREATE TABLE public.tool_market_terms (
  role TEXT NOT NULL CHECK(role IN ('buyer','miner')),
  hotkey TEXT NOT NULL CHECK(hotkey ~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'),
  terms_sha256 TEXT NOT NULL CHECK(terms_sha256 ~ '^[0-9a-f]{64}$'),
  terms_version TEXT NOT NULL CHECK(terms_version ~ '^[A-Za-z0-9._-]{1,32}$'),
  host TEXT NOT NULL CHECK(length(host) BETWEEN 1 AND 253),
  signed_at BIGINT NOT NULL,
  signed_body TEXT NOT NULL CHECK(length(signed_body) BETWEEN 2 AND 512),
  signature TEXT NOT NULL CHECK(length(signature) BETWEEN 128 AND 130),
  accepted_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY(role,hotkey,terms_sha256)
);

-- ``envelope`` is the miner's key sealed to the vault; only the vault can
-- open it. ``key_fingerprint`` is the vault's public name for the key.
-- ``signed_body`` is the register request as the miner's hotkey signed it,
-- with their attestation that the key is theirs to offer; ``host``,
-- ``signed_at`` and ``signature`` let anyone verify it again.
CREATE TABLE public.tool_market_keys (
  key_id UUID PRIMARY KEY,
  tool TEXT NOT NULL CHECK(tool ~ '^[a-z_]{1,40}$'),
  miner_hotkey TEXT NOT NULL CHECK(miner_hotkey ~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'),
  envelope TEXT NOT NULL CHECK(envelope LIKE 'lov1.%' AND length(envelope)<=4096),
  key_fingerprint TEXT NOT NULL CHECK(key_fingerprint ~ '^sha256:[0-9a-f]{64}$'),
  daily_cap_microusd BIGINT NOT NULL CHECK(daily_cap_microusd>0),
  host TEXT NOT NULL CHECK(length(host) BETWEEN 1 AND 253),
  signed_at BIGINT NOT NULL,
  signed_body TEXT NOT NULL CHECK(length(signed_body) BETWEEN 2 AND 8192),
  signature TEXT NOT NULL CHECK(length(signature) BETWEEN 128 AND 130),
  registered_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX tool_market_keys_tool ON public.tool_market_keys(tool);
CREATE INDEX tool_market_keys_fingerprint ON public.tool_market_keys(key_fingerprint);
CREATE INDEX tool_market_keys_miner ON public.tool_market_keys(miner_hotkey,tool);

CREATE TABLE public.tool_market_key_revocations (
  key_id UUID PRIMARY KEY REFERENCES public.tool_market_keys(key_id),
  reason TEXT NOT NULL CHECK(reason ~ '^[a-z_]{1,64}$'),
  revoked_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

-- A key's health: valid, unchecked (the vendor has no free check) or
-- invalid, from a vault check or a call that showed the key unusable. The
-- latest event decides whether the key trades.
CREATE TABLE public.tool_market_key_events (
  event_id BIGSERIAL PRIMARY KEY,
  key_id UUID NOT NULL REFERENCES public.tool_market_keys(key_id),
  status TEXT NOT NULL CHECK(status IN ('valid','unchecked','invalid')),
  reason TEXT CHECK(reason ~ '^[a-z_]{1,64}$'),
  recorded_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX tool_market_key_events_key ON public.tool_market_key_events(key_id,event_id DESC);

-- The latest ask is the key's price; a NULL ask withdraws it.
CREATE TABLE public.tool_market_asks (
  ask_sequence BIGSERIAL PRIMARY KEY,
  operation TEXT CHECK(operation ~ '^[a-z_]{1,40}$'),  -- NULL is the original tool-wide ask
  key_id UUID NOT NULL REFERENCES public.tool_market_keys(key_id),
  ask_microusd BIGINT CHECK(ask_microusd IS NULL OR ask_microusd BETWEEN 1 AND 9007199254740991),
  posted_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX tool_market_asks_key ON public.tool_market_asks(key_id,operation DESC NULLS LAST,ask_sequence DESC);

-- A buyer's budget, held against their Gamma balance while open. A tab is
-- open until it is closed or expires; then only what its calls committed
-- stays held.
CREATE TABLE public.tool_market_tabs (
  tab_id UUID PRIMARY KEY,
  hotkey TEXT NOT NULL CHECK(hotkey ~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'),
  token_hash TEXT NOT NULL UNIQUE CHECK(token_hash ~ '^sha256:[0-9a-f]{64}$'),
  budget_microusd BIGINT NOT NULL CHECK(budget_microusd>0),
  expires_at TIMESTAMPTZ NOT NULL,
  opened_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX tool_market_tabs_hotkey ON public.tool_market_tabs(hotkey,opened_at);

CREATE TABLE public.tool_market_tab_closures (
  tab_id UUID PRIMARY KEY REFERENCES public.tool_market_tabs(tab_id),
  closed_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE public.tool_market_calls (
  call_id UUID PRIMARY KEY,
  sequence BIGSERIAL NOT NULL UNIQUE,
  session_digest TEXT CHECK(session_digest IS NULL OR (session_digest ~ '^[0-9a-f]{64}$' AND tool='harvestapi' AND operation='search_leads')),
  tab_id UUID NOT NULL REFERENCES public.tool_market_tabs(tab_id),
  tool TEXT NOT NULL,
  operation TEXT NOT NULL CHECK(operation ~ '^[a-z_]{1,40}$'),
  key_id UUID NOT NULL REFERENCES public.tool_market_keys(key_id),
  ask_microusd BIGINT NOT NULL CHECK(ask_microusd>0),
  reserve_microusd BIGINT NOT NULL CHECK(reserve_microusd>0),
  -- The most the miner can be owed for this call; the daily cap counts it.
  owed_cap_microusd BIGINT NOT NULL CHECK(owed_cap_microusd>=0),
  opened_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX tool_market_calls_session ON public.tool_market_calls(session_digest,sequence) WHERE session_digest IS NOT NULL;
CREATE INDEX tool_market_calls_tab ON public.tool_market_calls(tab_id,sequence);
CREATE INDEX tool_market_calls_tab_tool ON public.tool_market_calls(tab_id,tool,opened_at);
CREATE INDEX tool_market_calls_key ON public.tool_market_calls(key_id,opened_at);

-- settled: the vendor answered and its usage is known. uncertain: it may
-- have charged the key but the usage is unknown (5xx, timeout, no cost in
-- the answer); the buyer pays nothing and the miner bears it. skipped: the
-- request was not sent, or the key failed and the next one was tried.
CREATE TABLE public.tool_market_call_settlements (
  call_id UUID PRIMARY KEY REFERENCES public.tool_market_calls(call_id),
  status TEXT NOT NULL CHECK(status IN ('settled','uncertain','skipped')),
  http_status INTEGER CHECK(http_status IS NULL OR http_status BETWEEN 100 AND 599),
  units BIGINT NOT NULL CHECK(units>=0),
  owed_microusd BIGINT NOT NULL CHECK(owed_microusd>=0),
  charge_microusd BIGINT NOT NULL CHECK(charge_microusd>=owed_microusd),
  epoch BIGINT NOT NULL CHECK(epoch>=0),
  settled_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  CHECK(status='settled' OR (units=0 AND owed_microusd=0 AND charge_microusd=0))
);

DO $tables770$
DECLARE t TEXT;
BEGIN
 FOREACH t IN ARRAY ARRAY['tool_market_gamma_source','tool_market_gamma_readings','tool_market_nonces','tool_market_terms','tool_market_keys',
   'tool_market_key_revocations','tool_market_key_events','tool_market_asks','tool_market_tabs',
   'tool_market_tab_closures','tool_market_calls','tool_market_call_settlements'] LOOP
  EXECUTE format('ALTER TABLE public.%I OWNER TO lab_arena_owner',t);
  EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY',t);
  EXECUTE format('REVOKE ALL ON public.%I FROM PUBLIC,anon,authenticated,service_role,lab_arena_service',t);
  EXECUTE format('CREATE TRIGGER %I BEFORE UPDATE OR DELETE ON public.%I FOR EACH ROW EXECUTE FUNCTION public.lab_arena_append_only_v1()',t||'_append_only',t);
  EXECUTE format('CREATE TRIGGER %I BEFORE TRUNCATE ON public.%I FOR EACH STATEMENT EXECUTE FUNCTION public.lab_arena_append_only_v1()',t||'_no_truncate',t);
 END LOOP;
END;
$tables770$;
ALTER SEQUENCE public.tool_market_gamma_source_source_id_seq OWNER TO lab_arena_owner;
ALTER SEQUENCE public.tool_market_gamma_readings_reading_id_seq OWNER TO lab_arena_owner;
ALTER SEQUENCE public.tool_market_key_events_event_id_seq OWNER TO lab_arena_owner;
ALTER SEQUENCE public.tool_market_asks_ask_sequence_seq OWNER TO lab_arena_owner;
ALTER SEQUENCE public.tool_market_calls_sequence_seq OWNER TO lab_arena_owner;

-- ---------------------------------------------------------------------------
-- Internal helpers (double underscore: never granted)
-- ---------------------------------------------------------------------------

-- A call left open this long is counted as uncertain: it holds nothing and
-- can no longer settle.
CREATE FUNCTION public.tool_market__open_seconds_v1() RETURNS INTEGER LANGUAGE sql IMMUTABLE AS $$ SELECT 600 $$;

-- What a tab's calls hold: each its charge once settled, its reserve while
-- young and open, nothing after that. It reads every call on the tab.
CREATE FUNCTION public.tool_market__committed_v1(p_tab_id UUID)
RETURNS BIGINT LANGUAGE sql STABLE SET search_path=pg_catalog,public AS $committed770$
 SELECT coalesce(sum(coalesce(s.charge_microusd,
   CASE WHEN c.opened_at>clock_timestamp()-make_interval(secs=>public.tool_market__open_seconds_v1())
    THEN c.reserve_microusd ELSE 0 END)),0)::BIGINT
 FROM public.tool_market_calls c LEFT JOIN public.tool_market_call_settlements s ON s.call_id=c.call_id
 WHERE c.tab_id=p_tab_id;
$committed770$;

-- A buyer's calls to one tool in the last hour that the miner may bear:
-- settled with unknown usage, or never settled (in flight, or cut off by a
-- restart; each may end that way).
-- A buyer can cause unknown usage (a timeout, an oversized answer), so at 20
-- the buyer's next call to that tool is refused. Counting calls in flight
-- keeps a burst of parallel calls inside the limit. Per tool, so one
-- vendor's outage does not stop a buyer's other tools. A call in the last
-- hour was made on a tab open then, so tabs expired before that are skipped
-- (and tabs live at most 30 days, which the hotkey index can use).
CREATE FUNCTION public.tool_market__buyer_uncertain_v1(p_hotkey TEXT,p_tool TEXT)
RETURNS BIGINT LANGUAGE sql STABLE SET search_path=pg_catalog,public AS $uncertain770$
 SELECT count(*) FROM public.tool_market_tabs t JOIN public.tool_market_calls c ON c.tab_id=t.tab_id
  LEFT JOIN public.tool_market_call_settlements s ON s.call_id=c.call_id
 WHERE t.hotkey=p_hotkey AND t.opened_at>clock_timestamp()-interval '30 days 1 hour'
  AND t.expires_at>clock_timestamp()-interval '1 hour' AND c.tool=p_tool
  AND c.opened_at>clock_timestamp()-interval '1 hour'
  AND (s.status='uncertain' OR s.call_id IS NULL);
$uncertain770$;

CREATE FUNCTION public.tool_market__tab_open_v1(p_tab public.tool_market_tabs)
RETURNS BOOLEAN LANGUAGE sql STABLE SET search_path=pg_catalog,public AS $open770$
 SELECT p_tab.expires_at>clock_timestamp()
  AND NOT EXISTS(SELECT 1 FROM public.tool_market_tab_closures x WHERE x.tab_id=p_tab.tab_id);
$open770$;

-- One scan of a key's last 24 h of calls, in owed (miner) terms:
-- * ``day``: settled calls count what they owe; unknown usage and calls
--   never settled (in flight, or cut off by a restart) the most they can owe. So the daily cap bounds what unknown usage
--   can cost a miner, however many buyers cause it.
-- * ``buyer_unknown``: the same for ``p_hotkey``'s unknown usage and open
--   calls. Past a quarter of the cap, that buyer's calls skip the key, so
--   no one buyer can fill a key's cap for free.
-- * ``demoted``: the key goes to the back of its book, still in it, after a
--   429 in the last minute, or in the last hour a 402 it was not failed for,
--   a 403 or unknown usage (5xx, timeout, oversized answer), each for 3
--   different buyers. A buyer can shape every one of those, so no one buyer
--   can push a rival's key back; a key really limited, empty or flaky is
--   seen by many.
CREATE FUNCTION public.tool_market__key_ledger_v1(p_key_id UUID,p_hotkey TEXT,
 OUT day BIGINT,OUT buyer_unknown BIGINT,OUT demoted BOOLEAN)
LANGUAGE sql STABLE SET search_path=pg_catalog,public AS $ledger770$
 WITH calls AS (
  SELECT t.hotkey,s.status,s.http_status,s.settled_at,s.owed_microusd,c.owed_cap_microusd cap,c.opened_at,
   s.call_id IS NULL OR s.status='uncertain' unknown
  FROM public.tool_market_calls c JOIN public.tool_market_tabs t ON t.tab_id=c.tab_id
   LEFT JOIN public.tool_market_call_settlements s ON s.call_id=c.call_id
  WHERE c.key_id=p_key_id AND c.opened_at>clock_timestamp()-interval '24 hours')
 SELECT coalesce(sum(CASE WHEN unknown THEN cap ELSE owed_microusd END),0)::BIGINT,
  coalesce(sum(cap) FILTER (WHERE hotkey=p_hotkey AND unknown),0)::BIGINT,
  count(DISTINCT hotkey) FILTER (WHERE status='skipped' AND http_status=429
    AND settled_at>clock_timestamp()-interval '1 minute')>=3
   OR count(DISTINCT hotkey) FILTER (WHERE opened_at>clock_timestamp()-interval '1 hour'
    AND (status='uncertain' OR (status='skipped' AND http_status IN (402,403))))>=3
 FROM calls;
$ledger770$;

CREATE FUNCTION public.tool_market__key_active_v1(p_key_id UUID)
RETURNS BOOLEAN LANGUAGE sql STABLE SET search_path=pg_catalog,public AS $active770$
 SELECT NOT EXISTS(SELECT 1 FROM public.tool_market_key_revocations r WHERE r.key_id=p_key_id);
$active770$;

CREATE FUNCTION public.tool_market__lock_v1(p_kind TEXT,p_value TEXT)
RETURNS VOID LANGUAGE sql SET search_path=pg_catalog,public AS $lock770$
 SELECT pg_advisory_xact_lock(hashtextextended('tool_market:'||p_kind||':'||p_value,0));
$lock770$;

-- ---------------------------------------------------------------------------
-- Buyers: balance, terms, tabs
-- ---------------------------------------------------------------------------

CREATE FUNCTION public.tool_market__balance_v1(p_hotkey TEXT)
RETURNS JSONB LANGUAGE plpgsql STABLE SET search_path=pg_catalog,public AS $balance770$
DECLARE r public.tool_market_gamma_readings; held BIGINT; credit BIGINT;
BEGIN
 SELECT * INTO r FROM public.tool_market_gamma_readings WHERE hotkey=p_hotkey
  ORDER BY block_number DESC,reading_id DESC LIMIT 1;
 -- An open tab holds its budget; a closed one what its calls committed
 -- (as tool_market__committed_v1), in one join, since each signed call is a tab.
 WITH tabs AS (SELECT t.tab_id,t.budget_microusd,public.tool_market__tab_open_v1(t) open
   FROM public.tool_market_tabs t WHERE t.hotkey=p_hotkey)
 SELECT coalesce((SELECT sum(budget_microusd) FROM tabs WHERE open),0)
  +coalesce((SELECT sum(coalesce(s.charge_microusd,
    CASE WHEN c.opened_at>clock_timestamp()-make_interval(secs=>public.tool_market__open_seconds_v1())
     THEN c.reserve_microusd ELSE 0 END))
   FROM tabs JOIN public.tool_market_calls c ON c.tab_id=tabs.tab_id
    LEFT JOIN public.tool_market_call_settlements s ON s.call_id=c.call_id WHERE NOT tabs.open),0)
  INTO held;
 -- 10^9 Gamma units per USD: 1000 units per micro-USD, rounded down.
 credit:=CASE WHEN r.available THEN floor(r.spent_units/1000)::BIGINT ELSE 0 END;
 RETURN jsonb_build_object('available',coalesce(r.available,FALSE),'block_number',r.block_number,
  'credit_microusd',credit,'held_microusd',held,'balance_microusd',credit-held);
END;
$balance770$;

CREATE FUNCTION public.tool_market_bind_gamma_source_v1(
 p_genesis TEXT,p_netuid INTEGER,p_service TEXT,p_registered_at BIGINT)
RETURNS BOOLEAN LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $bind770$
DECLARE actual public.tool_market_gamma_source;
BEGIN
 -- Each gateway process calls this on its first tool-market request.
 -- Readings hold this lock shared.
 PERFORM pg_advisory_xact_lock(hashtextextended('tool_market:gamma:source',0));
 SELECT * INTO actual FROM public.tool_market_gamma_source ORDER BY source_id DESC LIMIT 1;
 IF (actual.genesis,actual.netuid,actual.service,actual.registered_at)
   IS NOT DISTINCT FROM (p_genesis,p_netuid,p_service,p_registered_at) THEN
  RETURN TRUE;
 END IF;
 -- Any tab needed credit, so a reading: readings alone show the source in use.
 IF EXISTS(SELECT FROM public.tool_market_gamma_readings) THEN
  RAISE EXCEPTION 'Gamma credit source in use; repointing requires reconciliation' USING ERRCODE='55000';
 END IF;
 INSERT INTO public.tool_market_gamma_source(genesis,netuid,service,registered_at)
  VALUES(p_genesis,p_netuid,p_service,p_registered_at);
 RETURN TRUE;
END;
$bind770$;

CREATE FUNCTION public.tool_market_record_gamma_reading_v1(
 p_hotkey TEXT,p_block_number BIGINT,p_block_hash TEXT,p_units NUMERIC,p_available BOOLEAN,
 p_genesis TEXT,p_netuid INTEGER,p_service TEXT,p_registered_at BIGINT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $reading770$
DECLARE prior public.tool_market_gamma_readings; latest public.tool_market_gamma_readings;
BEGIN
 PERFORM pg_advisory_xact_lock_shared(hashtextextended('tool_market:gamma:source',0));
 IF NOT EXISTS (SELECT FROM (SELECT * FROM public.tool_market_gamma_source ORDER BY source_id DESC LIMIT 1) g WHERE
   (g.genesis,g.netuid,g.service,g.registered_at)=(p_genesis,p_netuid,p_service,p_registered_at)
   AND p_block_number>=g.registered_at) THEN
  RAISE EXCEPTION 'invalid Gamma credit source' USING ERRCODE='55000';
 END IF;
 -- Finalized readings and tab openings for one buyer share the same lock.
 PERFORM public.tool_market__lock_v1('buyer',p_hotkey);
 SELECT * INTO latest FROM public.tool_market_gamma_readings WHERE hotkey=p_hotkey
  ORDER BY block_number DESC,reading_id DESC LIMIT 1;
 IF latest.reading_id IS NOT NULL AND p_block_number<=latest.block_number THEN
  IF p_block_number=latest.block_number AND
    (lower(p_block_hash),p_available,CASE WHEN p_available THEN p_units ELSE 0 END)
      IS DISTINCT FROM (latest.block_hash,latest.available,latest.spent_units) THEN
   RETURN jsonb_build_object('status','reading_conflict');
  END IF;
  RETURN public.tool_market__balance_v1(p_hotkey);
 END IF;
 SELECT * INTO prior FROM public.tool_market_gamma_readings WHERE hotkey=p_hotkey AND available
  ORDER BY block_number DESC,reading_id DESC LIMIT 1;
 -- The spent counter only grows; a smaller value is a broken read.
 IF p_available AND prior.reading_id IS NOT NULL AND p_units<prior.spent_units THEN
  RETURN jsonb_build_object('status','counter_backwards');
 END IF;
 -- No row for what changes nothing: the same available total again (a
 -- client polling its balance), a hotkey that never spent here (anyone may
 -- ask for any balance), or an unavailable read after one.
 IF (p_available AND coalesce(latest.available,FALSE) AND latest.spent_units=p_units)
   OR (p_available AND latest.reading_id IS NULL AND p_units=0)
   OR (NOT p_available AND NOT coalesce(latest.available,FALSE)) THEN
  RETURN public.tool_market__balance_v1(p_hotkey);
 END IF;
 INSERT INTO public.tool_market_gamma_readings(hotkey,block_number,block_hash,spent_units,available)
 VALUES(p_hotkey,p_block_number,lower(p_block_hash),CASE WHEN p_available THEN p_units ELSE 0 END,p_available);
 RETURN public.tool_market__balance_v1(p_hotkey);
END;
$reading770$;

CREATE FUNCTION public.tool_market_use_nonce_v1(p_hotkey TEXT,p_signature_hash TEXT,p_signed_at BIGINT)
RETURNS BOOLEAN LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $nonce770$
BEGIN
 INSERT INTO public.tool_market_nonces(hotkey,signature_hash,signed_at) VALUES(p_hotkey,p_signature_hash,p_signed_at)
 ON CONFLICT DO NOTHING;
 RETURN FOUND;
END;
$nonce770$;

CREATE FUNCTION public.tool_market_accept_terms_v1(p_role TEXT,p_hotkey TEXT,p_version TEXT,p_terms_sha256 TEXT,
 p_host TEXT,p_signed_at BIGINT,p_signed_body TEXT,p_signature TEXT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $terms770$
BEGIN
 INSERT INTO public.tool_market_terms(role,hotkey,terms_sha256,terms_version,host,signed_at,signed_body,signature)
 VALUES(p_role,p_hotkey,p_terms_sha256,p_version,p_host,p_signed_at,p_signed_body,p_signature)
 ON CONFLICT DO NOTHING;
 RETURN jsonb_build_object('status',CASE WHEN FOUND THEN 'accepted' ELSE 'exists' END);
END;
$terms770$;

CREATE FUNCTION public.tool_market_open_tab_v1(p_tab_id UUID,p_hotkey TEXT,p_token_hash TEXT,p_budget BIGINT,
 p_seconds INTEGER,p_terms_sha256 TEXT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $open_tab770$
DECLARE balance JSONB; expires TIMESTAMPTZ;
BEGIN
 PERFORM public.tool_market__lock_v1('buyer',p_hotkey);
 -- NULL terms: a one-call tab for a btauth/1 call, whose terms the services
 -- document says the paid request accepts (the Gamma standard allows no
 -- step before a funded hotkey's first call).
 IF p_terms_sha256 IS NOT NULL AND NOT EXISTS(SELECT 1 FROM public.tool_market_terms WHERE role='buyer'
   AND hotkey=p_hotkey AND terms_sha256=p_terms_sha256) THEN
  RETURN jsonb_build_object('status','terms_required');
 END IF;
 balance:=public.tool_market__balance_v1(p_hotkey);
 IF (balance->>'balance_microusd')::BIGINT<p_budget THEN  -- an unavailable reading credits 0
  RETURN jsonb_build_object('status','insufficient_balance','balance',balance);
 END IF;
 expires:=clock_timestamp()+make_interval(secs=>p_seconds);
 INSERT INTO public.tool_market_tabs(tab_id,hotkey,token_hash,budget_microusd,expires_at)
 VALUES(p_tab_id,p_hotkey,p_token_hash,p_budget,expires);
 RETURN jsonb_build_object('status','opened','tab_id',p_tab_id,'budget_microusd',p_budget,'expires_at',expires);
END;
$open_tab770$;

-- Whose tab a token opens and whether it is open: the call path's first
-- read. A tab keeps the terms it was opened under (terms section 9).
CREATE FUNCTION public.tool_market_tab_state_v1(p_token_hash TEXT)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $tab_state770$
 SELECT jsonb_build_object('hotkey',t.hotkey,'open',public.tool_market__tab_open_v1(t))
 FROM public.tool_market_tabs t WHERE t.token_hash=p_token_hash;
$tab_state770$;

CREATE FUNCTION public.tool_market_tab_v1(p_token_hash TEXT)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $tab770$
 SELECT jsonb_build_object('tab_id',t.tab_id,'hotkey',t.hotkey,'budget_microusd',t.budget_microusd,
  'committed_microusd',public.tool_market__committed_v1(t.tab_id),'expires_at',t.expires_at,
  'open',public.tool_market__tab_open_v1(t),
  'calls',(SELECT count(*) FROM public.tool_market_calls c WHERE c.tab_id=t.tab_id))
 FROM public.tool_market_tabs t WHERE t.token_hash=p_token_hash;
$tab770$;

CREATE FUNCTION public.tool_market_close_tab_v1(p_token_hash TEXT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $close770$
DECLARE tab public.tool_market_tabs;
BEGIN
 SELECT * INTO tab FROM public.tool_market_tabs WHERE token_hash=p_token_hash;
 IF tab.tab_id IS NULL THEN RETURN NULL; END IF;
 -- The buyer lock, as reserve and open_tab: no call is reserved on a tab
 -- whose close another tab's budget already counts on.
 PERFORM public.tool_market__lock_v1('buyer',tab.hotkey);
 INSERT INTO public.tool_market_tab_closures(tab_id) VALUES(tab.tab_id) ON CONFLICT DO NOTHING;
 RETURN public.tool_market_tab_v1(p_token_hash);
END;
$close770$;

-- ---------------------------------------------------------------------------
-- Miners: keys, health, asks
-- ---------------------------------------------------------------------------

-- One active listing per key: the same vendor key sealed twice, by one miner
-- or two, is refused while the first is listed. For security, a hotkey
-- registers at most one key per tool every 30 days, revoked keys included.
-- A revoked key that never took a call does not count, but only one such
-- retry is allowed: at most 2 registrations per hotkey and tool in 30 days.
CREATE FUNCTION public.tool_market_register_key_v1(p_key_id UUID,p_tool TEXT,p_hotkey TEXT,p_envelope TEXT,
 p_fingerprint TEXT,p_daily_cap BIGINT,p_status TEXT,p_host TEXT,p_signed_at BIGINT,
 p_signed_body TEXT,p_signature TEXT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $register770$
DECLARE next_at TIMESTAMPTZ;
BEGIN
 IF p_status NOT IN ('valid','unchecked') THEN
  RAISE EXCEPTION 'only a valid or unchecked key is registered' USING ERRCODE='22023';
 END IF;
 PERFORM public.tool_market__lock_v1('miner_tool',p_hotkey||':'||p_tool);
 SELECT greatest(
   max(k.registered_at) FILTER (WHERE public.tool_market__key_active_v1(k.key_id) OR EXISTS(
    SELECT 1 FROM public.tool_market_calls c WHERE c.key_id=k.key_id)),
   (SELECT r.registered_at FROM public.tool_market_keys r WHERE r.miner_hotkey=p_hotkey AND r.tool=p_tool
    ORDER BY r.registered_at DESC OFFSET 1 LIMIT 1))+interval '30 days'
  INTO next_at FROM public.tool_market_keys k WHERE k.miner_hotkey=p_hotkey AND k.tool=p_tool;
 IF next_at>clock_timestamp() THEN
  RETURN jsonb_build_object('status','too_soon','next_at',next_at);
 END IF;
 PERFORM public.tool_market__lock_v1('fingerprint',p_fingerprint);
 IF EXISTS(SELECT 1 FROM public.tool_market_keys k WHERE k.key_fingerprint=p_fingerprint
   AND public.tool_market__key_active_v1(k.key_id)) THEN
  RETURN jsonb_build_object('status','duplicate');
 END IF;
 INSERT INTO public.tool_market_keys(key_id,tool,miner_hotkey,envelope,key_fingerprint,daily_cap_microusd,
  host,signed_at,signed_body,signature)
 VALUES(p_key_id,p_tool,p_hotkey,p_envelope,p_fingerprint,p_daily_cap,p_host,p_signed_at,p_signed_body,p_signature);
 INSERT INTO public.tool_market_key_events(key_id,status) VALUES(p_key_id,p_status);
 RETURN jsonb_build_object('status','registered','key_id',p_key_id);
END;
$register770$;

CREATE FUNCTION public.tool_market_key_v1(p_key_id UUID)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $key770$
 SELECT jsonb_build_object('key_id',k.key_id,'tool',k.tool,'miner_hotkey',k.miner_hotkey,'envelope',k.envelope,
  'daily_cap_microusd',k.daily_cap_microusd,'active',public.tool_market__key_active_v1(k.key_id))
 FROM public.tool_market_keys k WHERE k.key_id=p_key_id;
$key770$;

CREATE FUNCTION public.tool_market_revoke_key_v1(p_key_id UUID,p_hotkey TEXT,p_reason TEXT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $revoke770$
BEGIN
 PERFORM public.tool_market__lock_v1('key',p_key_id::TEXT);  -- after any reserve in flight
 IF NOT EXISTS(SELECT 1 FROM public.tool_market_keys WHERE key_id=p_key_id AND miner_hotkey=p_hotkey) THEN
  RETURN jsonb_build_object('status','not_found');
 END IF;
 INSERT INTO public.tool_market_key_revocations(key_id,reason) VALUES(p_key_id,p_reason) ON CONFLICT DO NOTHING;
 RETURN jsonb_build_object('status','revoked');
END;
$revoke770$;

CREATE FUNCTION public.tool_market_record_key_event_v1(p_key_id UUID,p_status TEXT,p_reason TEXT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $event770$
BEGIN
 PERFORM public.tool_market__lock_v1('key',p_key_id::TEXT);
 INSERT INTO public.tool_market_key_events(key_id,status,reason) VALUES(p_key_id,p_status,p_reason);
 RETURN jsonb_build_object('status','recorded');
END;
$event770$;

CREATE FUNCTION public.tool_market_post_ask_v1(p_key_id UUID,p_hotkey TEXT,p_ask BIGINT,p_operation TEXT DEFAULT NULL)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $ask770$
DECLARE seq BIGINT;
BEGIN
 PERFORM public.tool_market__lock_v1('key',p_key_id::TEXT);  -- after any reserve in flight
 IF NOT EXISTS(SELECT 1 FROM public.tool_market_keys WHERE key_id=p_key_id AND miner_hotkey=p_hotkey)
  OR NOT public.tool_market__key_active_v1(p_key_id) THEN
  RETURN jsonb_build_object('status','not_found');
 END IF;
 INSERT INTO public.tool_market_asks(key_id,operation,ask_microusd) VALUES(p_key_id,p_operation,p_ask) RETURNING ask_sequence INTO seq;
 RETURN jsonb_build_object('status',CASE WHEN p_ask IS NULL THEN 'withdrawn' ELSE 'posted' END,'ask_sequence',seq);
END;
$ask770$;

-- A signed general offer covers only the original operations. A narrower
-- offer, even a withdrawal, always takes precedence. Read it both when
-- quoting and under the key lock immediately before reservation.
CREATE FUNCTION public.tool_market__effective_ask_v1(p_key_id UUID,p_tool TEXT,p_operation TEXT)
RETURNS public.tool_market_asks LANGUAGE sql STABLE SET search_path=pg_catalog,public AS $effective770$
 SELECT a.* FROM public.tool_market_asks a WHERE a.key_id=p_key_id
  AND (a.operation=p_operation OR (a.operation IS NULL AND
   (p_operation IS NULL OR (p_tool,p_operation) IN (
    ('zerobounce','validate'),('prospeo','enrich_person'),('findymail','find_from_linkedin'),
    ('findymail','find_from_name'),('hunter','email_finder'),('exa','search'),('exa','contents'),
    ('harvestapi','linkedin_profile'),('harvestapi','linkedin_company')))))
 ORDER BY a.operation DESC NULLS LAST, a.ask_sequence DESC LIMIT 1;
$effective770$;

-- The tool's tradeable supply. A key trades while it is listed, its miner
-- accepted the current terms, its latest ask is a price, its latest event
-- is valid or unchecked, and its last 24 h are under its daily cap. The
-- caller ranks it (``book.rank``) and skips a key without room for a call.
CREATE FUNCTION public.tool_market_book_v1(p_tool TEXT,p_terms_sha256 TEXT,p_operation TEXT DEFAULT NULL)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $book770$
 -- MATERIALIZED, so each lookup runs once per key, and the 24 h scan only
 -- for keys with a price and good health.
 WITH latest AS MATERIALIZED (
  SELECT k.key_id,k.miner_hotkey,k.envelope,k.daily_cap_microusd,
   public.tool_market__effective_ask_v1(k.key_id,p_tool,p_operation) ask,
   (SELECT e.status FROM public.tool_market_key_events e WHERE e.key_id=k.key_id ORDER BY e.event_id DESC LIMIT 1) health
  FROM public.tool_market_keys k
  WHERE k.tool=p_tool AND public.tool_market__key_active_v1(k.key_id)
   AND EXISTS(SELECT 1 FROM public.tool_market_terms m WHERE m.role='miner' AND m.hotkey=k.miner_hotkey
    AND m.terms_sha256=p_terms_sha256)),
 keyed AS MATERIALIZED (
  SELECT l.key_id,l.miner_hotkey,l.envelope,(l.ask).ask_microusd,(l.ask).ask_sequence,
   l.daily_cap_microusd-g.day remaining,g.demoted
  FROM latest l CROSS JOIN LATERAL public.tool_market__key_ledger_v1(l.key_id,NULL) g
  WHERE (l.ask).ask_microusd IS NOT NULL AND l.health IN ('valid','unchecked'))
 SELECT coalesce(jsonb_agg(jsonb_build_object('key_id',key_id,'miner_hotkey',miner_hotkey,'envelope',envelope,
   'ask_microusd',ask_microusd,'ask_sequence',ask_sequence,'remaining_microusd',remaining,
   'demoted',demoted)),'[]'::JSONB)
 FROM keyed WHERE remaining>0;
$book770$;

-- ---------------------------------------------------------------------------
-- Calls
-- ---------------------------------------------------------------------------

-- A vendor's pagination session belongs to the account that first served a
-- page of it; until one has, page one may fail over to any key. Tabs rotate;
-- the authenticated buyer is the stable owner of the request.
CREATE FUNCTION public.tool_market_session_key_v1(p_buyer TEXT,p_digest TEXT)
RETURNS UUID LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $session770$
 SELECT c.key_id FROM public.tool_market_calls c
 JOIN public.tool_market_tabs t USING(tab_id) JOIN public.tool_market_call_settlements s USING(call_id)
 WHERE c.session_digest=p_digest AND t.hotkey=p_buyer AND s.status='settled'
   AND s.http_status BETWEEN 200 AND 299 AND s.units>0 ORDER BY s.settled_at,c.sequence LIMIT 1
$session770$;

-- Hold one call's reserve against its tab, at the key's current ask.
-- p_claim, a purchase's request_id digest, is spent only with the reserve it
-- is committed with, so a purchase refused before then can be retried. Every
-- claim is written under the buyer lock, so checking it first is exact.
CREATE FUNCTION public.tool_market_reserve_call_v1(p_call_id UUID,p_token_hash TEXT,p_tool TEXT,p_operation TEXT,
 p_key_id UUID,p_ask BIGINT,p_owed_cap BIGINT,p_session TEXT DEFAULT NULL,p_first BOOLEAN DEFAULT true,
 p_claim TEXT DEFAULT NULL)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $reserve770$
DECLARE tab public.tool_market_tabs; k public.tool_market_keys; current_ask BIGINT; session_key UUID; g RECORD;
 reserve BIGINT:=p_owed_cap+(p_owed_cap+9)/10;  -- the fee on the most the call can owe
BEGIN
 SELECT * INTO tab FROM public.tool_market_tabs WHERE token_hash=p_token_hash;
 IF tab.tab_id IS NULL THEN RETURN jsonb_build_object('status','tab_closed'); END IF;
 -- Lock order: buyer, then key. The buyer lock orders this reserve against
 -- a close of this tab and the balance check of a new tab, so a tab is
 -- open, checked under the lock, whenever a call is added to it.
 PERFORM public.tool_market__lock_v1('buyer',tab.hotkey);
 IF NOT public.tool_market__tab_open_v1(tab) THEN
  RETURN jsonb_build_object('status','tab_closed');
 END IF;
 IF p_claim IS NOT NULL AND EXISTS(SELECT 1 FROM public.tool_market_nonces n
   WHERE n.hotkey=tab.hotkey AND n.signature_hash=p_claim) THEN
  RETURN jsonb_build_object('status','request_already_used');
 END IF;
 IF p_session IS NOT NULL THEN
  IF p_tool<>'harvestapi' OR p_operation<>'search_leads' OR p_session !~ '^[0-9a-f]{64}$' OR p_first IS NULL THEN
   RAISE EXCEPTION 'invalid supplier session';
  END IF;
  session_key:=public.tool_market_session_key_v1(tab.hotkey,p_session);
  IF session_key IS NOT NULL AND session_key<>p_key_id THEN RETURN jsonb_build_object('status','session_key_changed'); END IF;
  IF NOT p_first AND session_key IS NULL THEN RETURN jsonb_build_object('status','session_page_one_required'); END IF;
 END IF;
 -- Key rows never change, so this read needs no key lock.
 SELECT * INTO k FROM public.tool_market_keys WHERE key_id=p_key_id AND tool=p_tool;
 IF k.key_id IS NULL THEN
  RETURN jsonb_build_object('status','ask_changed');
 END IF;
 IF k.miner_hotkey=tab.hotkey THEN  -- a miner's own tab never buys from their own key
  RETURN jsonb_build_object('status','own_key');
 END IF;
 IF public.tool_market__committed_v1(tab.tab_id)+reserve>tab.budget_microusd THEN
  RETURN jsonb_build_object('status','tab_budget');
 END IF;
 IF public.tool_market__buyer_uncertain_v1(tab.hotkey,p_tool)>=20 THEN
  RETURN jsonb_build_object('status','uncertain_limit');
 END IF;
 PERFORM public.tool_market__lock_v1('key',p_key_id::TEXT);
 SELECT a.ask_microusd INTO current_ask FROM public.tool_market__effective_ask_v1(p_key_id,p_tool,p_operation) a;
 IF NOT public.tool_market__key_active_v1(p_key_id) OR current_ask IS DISTINCT FROM p_ask
  OR NOT coalesce((SELECT e.status IN ('valid','unchecked') FROM public.tool_market_key_events e
      WHERE e.key_id=p_key_id ORDER BY e.event_id DESC LIMIT 1),FALSE) THEN
  RETURN jsonb_build_object('status','ask_changed');
 END IF;
 SELECT * INTO g FROM public.tool_market__key_ledger_v1(p_key_id,tab.hotkey);
 -- The quarter matches ToolMarket.post_ask, which makes a cap hold 4 of the key's largest calls.
 IF g.day+p_owed_cap>k.daily_cap_microusd OR g.buyer_unknown+p_owed_cap>k.daily_cap_microusd/4 THEN
  RETURN jsonb_build_object('status','key_cap');
 END IF;
 IF p_claim IS NOT NULL THEN
  INSERT INTO public.tool_market_nonces(hotkey,signature_hash,signed_at)
  VALUES(tab.hotkey,p_claim,extract(epoch FROM clock_timestamp())::BIGINT);
 END IF;
 INSERT INTO public.tool_market_calls(call_id,tab_id,tool,operation,key_id,ask_microusd,reserve_microusd,
  owed_cap_microusd,session_digest)
 VALUES(p_call_id,tab.tab_id,p_tool,p_operation,p_key_id,p_ask,reserve,p_owed_cap,p_session);
 RETURN jsonb_build_object('status','reserved');
END;
$reserve770$;

-- Settle once, within the open window. A settlement may not charge more
-- than the call reserved. It takes the buyer and key locks, as a reserve
-- does, so a reserve never sees a call that has just aged out of the window
-- before its settlement commits.
CREATE FUNCTION public.tool_market_settle_call_v1(p_call_id UUID,p_status TEXT,p_http_status INTEGER,p_units BIGINT,
 p_owed BIGINT,p_charge BIGINT,p_epoch BIGINT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $settle770$
DECLARE c public.tool_market_calls; previous public.tool_market_call_settlements;
BEGIN
 SELECT * INTO c FROM public.tool_market_calls WHERE call_id=p_call_id;
 IF c.call_id IS NULL THEN RETURN jsonb_build_object('status','not_found'); END IF;
 PERFORM public.tool_market__lock_v1('buyer',(SELECT t.hotkey FROM public.tool_market_tabs t WHERE t.tab_id=c.tab_id));
 PERFORM public.tool_market__lock_v1('key',c.key_id::TEXT);
 SELECT * INTO previous FROM public.tool_market_call_settlements WHERE call_id=p_call_id;
 IF FOUND THEN
  IF (p_status,p_http_status,p_units,p_owed,p_charge,p_epoch) IS DISTINCT FROM
     (previous.status,previous.http_status,previous.units,previous.owed_microusd,previous.charge_microusd,previous.epoch) THEN
   RAISE EXCEPTION 'conflicting call settlement' USING ERRCODE='22023';
  END IF;
  RETURN jsonb_build_object('status','exists');
 END IF;
 IF c.opened_at<=clock_timestamp()-make_interval(secs=>public.tool_market__open_seconds_v1()) THEN
  RETURN jsonb_build_object('status','expired');
 END IF;
 -- The reserve is the fee on the owed cap, so these also keep the charge within it.
 IF p_owed>c.owed_cap_microusd OR p_charge<>p_owed+(p_owed+9)/10 THEN
  RAISE EXCEPTION 'call settlement exceeds its reservation or differs from the market fee' USING ERRCODE='22023';
 END IF;
 INSERT INTO public.tool_market_call_settlements(call_id,status,http_status,units,owed_microusd,charge_microusd,epoch)
 VALUES(p_call_id,p_status,p_http_status,p_units,p_owed,p_charge,p_epoch);
 RETURN jsonb_build_object('status','settled');
END;
$settle770$;

-- A miner's keys, their state and what each has earned.
CREATE FUNCTION public.tool_market_statement_v1(p_hotkey TEXT)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $statement770$
 SELECT jsonb_build_object('keys',coalesce(jsonb_agg(jsonb_build_object(
   'key_id',k.key_id,'tool',k.tool,'key_fingerprint',k.key_fingerprint,
   'active',public.tool_market__key_active_v1(k.key_id),'daily_cap_microusd',k.daily_cap_microusd,
   'ask_microusd',(public.tool_market__effective_ask_v1(k.key_id,k.tool,NULL)).ask_microusd,
   'operation_asks',coalesce((SELECT jsonb_object_agg(o.operation,o.ask_microusd) FROM
     (SELECT DISTINCT ON (a.operation) a.operation,a.ask_microusd FROM public.tool_market_asks a
      WHERE a.key_id=k.key_id AND a.operation IS NOT NULL ORDER BY a.operation,a.ask_sequence DESC) o),'{}'::jsonb),
   'health',(SELECT jsonb_build_object('status',e.status,'reason',e.reason,'at',e.recorded_at)
     FROM public.tool_market_key_events e WHERE e.key_id=k.key_id ORDER BY e.event_id DESC LIMIT 1),
   'calls',u.calls,'uncertain_calls',u.uncertain,'owed_microusd',u.owed)
   ORDER BY k.registered_at,k.key_id),'[]'::JSONB))
 FROM public.tool_market_keys k, LATERAL (
  SELECT count(*) calls,count(*) FILTER (WHERE s.status='uncertain' OR s.call_id IS NULL) uncertain,
   coalesce(sum(s.owed_microusd),0) owed
  FROM public.tool_market_calls c LEFT JOIN public.tool_market_call_settlements s ON s.call_id=c.call_id
  WHERE c.key_id=k.key_id) u
 WHERE k.miner_hotkey=p_hotkey;
$statement770$;

-- A buyer's own calls, oldest first, after a watermark (a call sequence).
-- Only calls opened more than a minute past the late-answer window are
-- listed, settled or expired: by then every call with a lower sequence has
-- committed and finished, so a watermark never passes one that may still
-- change. Usage only, never the vendor's answer.
CREATE FUNCTION public.tool_market_usage_v1(p_hotkey TEXT,p_after BIGINT,p_limit INTEGER)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $usage770$
 -- Two calls can take their sequence and their opened_at in opposite
 -- orders, so a page stops before the first call still too young.
 WITH mine AS MATERIALIZED (
  SELECT c.*,c.opened_at<clock_timestamp()-make_interval(secs=>public.tool_market__open_seconds_v1()+60) settled_by_now
  FROM public.tool_market_tabs t JOIN public.tool_market_calls c ON c.tab_id=t.tab_id
  WHERE t.hotkey=p_hotkey AND c.sequence>p_after),
 page AS (
  SELECT c.*,s.status settled,s.http_status,s.units,s.charge_microusd,s.settled_at
  FROM mine c LEFT JOIN public.tool_market_call_settlements s ON s.call_id=c.call_id
  WHERE c.sequence<coalesce((SELECT min(y.sequence) FROM mine y WHERE NOT y.settled_by_now),9223372036854775807)
  ORDER BY c.sequence LIMIT greatest(1,least(p_limit,500)))
 SELECT jsonb_build_object(
  'calls',coalesce((SELECT jsonb_agg(jsonb_build_object('call_id',call_id,'sequence',sequence,'tool',tool,
    'operation',operation,'opened_at',opened_at,'status',coalesce(settled,'expired'),'http_status',http_status,
    'units',coalesce(units,0),'charge_microusd',coalesce(charge_microusd,0),'settled_at',settled_at)
    ORDER BY sequence) FROM page),'[]'::JSONB),
  'watermark',coalesce((SELECT max(sequence) FROM page),p_after));
$usage770$;

-- Every tool_market function: owner, no PUBLIC execute; the service may
-- execute all but the double-underscore helpers.
DO $functions770$
DECLARE f REGPROCEDURE; name TEXT;
BEGIN
 FOR f,name IN SELECT p.oid::REGPROCEDURE,p.proname FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
  WHERE n.nspname='public' AND p.proname LIKE 'tool\_market%\_v1' LOOP
  EXECUTE format('ALTER FUNCTION %s OWNER TO lab_arena_owner',f);
  EXECUTE format('REVOKE ALL ON FUNCTION %s FROM PUBLIC,anon,authenticated,service_role,lab_arena_service',f);
  IF name NOT LIKE 'tool\_market\_\_%' THEN
   EXECUTE format('GRANT EXECUTE ON FUNCTION %s TO lab_arena_service',f);
  END IF;
 END LOOP;
END;
$functions770$;

DO $restore770$
BEGIN
 IF NOT(SELECT had_create FROM pg_temp.tool_market770_acl)THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;END IF;
 IF(SELECT nspacl FROM pg_namespace WHERE nspname='public')IS DISTINCT FROM(SELECT nspacl FROM pg_temp.tool_market770_acl)THEN
  RAISE EXCEPTION 'tool_market770 schema ACL differs';END IF;
END;
$restore770$;
NOTIFY pgrst,'reload schema';
COMMIT;
