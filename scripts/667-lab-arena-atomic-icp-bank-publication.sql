-- Persist and activate each daily ICP bank in one transaction. A bank date is
-- write-once, activation is serialized, and an older generation cannot replace
-- a bank for a newer UTC date.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $icp_bank_preflight$
DECLARE active_count BIGINT;
BEGIN
  IF pg_catalog.to_regclass('public.qualification_private_icp_sets') IS NULL THEN
    RAISE EXCEPTION 'qualification_private_icp_sets is missing';
  END IF;
  IF NOT EXISTS (
       SELECT 1 FROM information_schema.columns
       WHERE table_schema='public' AND table_name='qualification_private_icp_sets'
         AND column_name='industry_distribution' AND data_type='jsonb'
     ) OR NOT EXISTS (
       SELECT 1 FROM information_schema.columns
       WHERE table_schema='public' AND table_name='qualification_private_icp_sets'
         AND column_name='generation_seed'
     ) OR NOT EXISTS (
       SELECT 1 FROM information_schema.columns
       WHERE table_schema='public' AND table_name='qualification_private_icp_sets'
         AND column_name='icp_set_hash'
     ) THEN
    RAISE EXCEPTION 'qualification_private_icp_sets schema is incomplete';
  END IF;
  SELECT count(*) INTO active_count
  FROM public.qualification_private_icp_sets WHERE is_active;
  IF active_count > 1 THEN
    RAISE EXCEPTION 'multiple daily ICP banks are active; reconcile before applying';
  END IF;
END;
$icp_bank_preflight$;

-- This invariant prevents two active rows. It does not fence old gateway
-- writers from leaving zero active rows, so deployment must quiesce every old
-- ICP writer before applying this migration and starting the new code.
DO $icp_bank_active_index$
DECLARE
  index_oid OID;
  valid_index BOOLEAN;
BEGIN
  index_oid := pg_catalog.to_regclass(
    'public.qualification_private_icp_sets_one_active'
  );
  IF index_oid IS NULL THEN
    EXECUTE 'CREATE UNIQUE INDEX qualification_private_icp_sets_one_active '
      'ON public.qualification_private_icp_sets USING btree (is_active) '
      'WHERE is_active';
    index_oid := pg_catalog.to_regclass(
      'public.qualification_private_icp_sets_one_active'
    );
  END IF;
  SELECT i.indisunique AND i.indnatts=1 AND i.indnkeyatts=1
     AND pg_catalog.pg_get_expr(i.indpred,i.indrelid)='is_active'
     AND pg_catalog.pg_get_indexdef(i.indexrelid) LIKE
       '% ON public.qualification_private_icp_sets USING btree (is_active) WHERE is_active'
    INTO valid_index
  FROM pg_catalog.pg_index i
  WHERE i.indexrelid=index_oid
    AND i.indrelid='public.qualification_private_icp_sets'::REGCLASS;
  IF valid_index IS DISTINCT FROM TRUE THEN
    RAISE EXCEPTION 'daily ICP active index has an unexpected definition';
  END IF;
END;
$icp_bank_active_index$;

CREATE OR REPLACE FUNCTION public.qualification_private_icp_set_publish_v1(
  p_set_id BIGINT,
  p_icps JSONB,
  p_icp_set_hash TEXT,
  p_industry_distribution JSONB,
  p_active_from TIMESTAMPTZ,
  p_active_until TIMESTAMPTZ,
  p_generation_seed TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $publish_icp_bank$
DECLARE
  set_date DATE;
  active_set_id BIGINT;
  inserted_count BIGINT := 0;
  activated_count BIGINT;
  active_count BIGINT;
  persisted public.qualification_private_icp_sets;
BEGIN
  IF p_set_id IS NULL OR p_set_id::TEXT !~ '^[0-9]{8}$'
     OR pg_catalog.jsonb_typeof(p_icps) IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_array_length(p_icps) < 1
     OR p_icp_set_hash IS NULL OR p_icp_set_hash !~ '^[0-9a-f]{64}$'
     OR pg_catalog.jsonb_typeof(p_industry_distribution) IS DISTINCT FROM 'object'
     OR p_active_from IS NULL OR p_active_until IS NULL
     OR p_active_until <= p_active_from THEN
    RAISE EXCEPTION 'daily ICP bank input is invalid' USING ERRCODE='22023';
  END IF;
  BEGIN
    set_date := pg_catalog.to_date(p_set_id::TEXT, 'YYYYMMDD');
  EXCEPTION WHEN OTHERS THEN
    RAISE EXCEPTION 'daily ICP set id is not a calendar date' USING ERRCODE='22023';
  END;
  IF pg_catalog.to_char(set_date, 'YYYYMMDD') IS DISTINCT FROM p_set_id::TEXT
     OR p_active_from AT TIME ZONE 'UTC' <> set_date::TIMESTAMP
     OR p_active_until <> p_active_from + INTERVAL '24 hours' THEN
    RAISE EXCEPTION 'daily ICP bank date window is invalid' USING ERRCODE='22023';
  END IF;

  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('qualification_private_icp_sets.active', 0)
  );

  SELECT set_id INTO active_set_id
  FROM public.qualification_private_icp_sets
  WHERE is_active
  ORDER BY set_id DESC
  LIMIT 1
  FOR UPDATE;
  IF FOUND AND active_set_id > p_set_id THEN
    SELECT * INTO STRICT persisted
    FROM public.qualification_private_icp_sets
    WHERE set_id=active_set_id;
    RETURN pg_catalog.jsonb_build_object(
      'status','stale','set_id',persisted.set_id,'inserted',FALSE,
      'icps',persisted.icps,'icp_set_hash',persisted.icp_set_hash,
      'industry_distribution',persisted.industry_distribution
    );
  END IF;

  SELECT * INTO persisted
  FROM public.qualification_private_icp_sets
  WHERE set_id=p_set_id
  FOR UPDATE;
  IF FOUND THEN
    IF persisted.is_active THEN
      RETURN pg_catalog.jsonb_build_object(
        'status','activated','set_id',persisted.set_id,'inserted',FALSE,
        'icps',persisted.icps,'icp_set_hash',persisted.icp_set_hash,
        'industry_distribution',persisted.industry_distribution
      );
    END IF;
    IF persisted.icps IS DISTINCT FROM p_icps
       OR persisted.icp_set_hash IS DISTINCT FROM p_icp_set_hash
       OR persisted.industry_distribution IS DISTINCT FROM p_industry_distribution
       OR persisted.active_from IS DISTINCT FROM p_active_from
       OR persisted.active_until IS DISTINCT FROM p_active_until
       OR persisted.generation_seed IS DISTINCT FROM p_generation_seed THEN
      RETURN pg_catalog.jsonb_build_object(
        'status','conflict','set_id',p_set_id,'inserted',FALSE,
        'icps',persisted.icps,'icp_set_hash',persisted.icp_set_hash,
        'industry_distribution',persisted.industry_distribution,
        'active_from',persisted.active_from,'active_until',persisted.active_until,
        'generation_seed',persisted.generation_seed
      );
    END IF;
  ELSE
    INSERT INTO public.qualification_private_icp_sets (
      set_id,icps,icp_set_hash,industry_distribution,active_from,active_until,
      generation_seed,is_active
    ) VALUES (
      p_set_id,p_icps,p_icp_set_hash,p_industry_distribution,p_active_from,
      p_active_until,p_generation_seed,FALSE
    );
    GET DIAGNOSTICS inserted_count = ROW_COUNT;
    SELECT * INTO STRICT persisted
    FROM public.qualification_private_icp_sets
    WHERE set_id=p_set_id
    FOR UPDATE;
  END IF;

  UPDATE public.qualification_private_icp_sets
  SET is_active=FALSE
  WHERE is_active AND set_id<>p_set_id;
  UPDATE public.qualification_private_icp_sets
  SET is_active=TRUE
  WHERE set_id=p_set_id;
  GET DIAGNOSTICS activated_count = ROW_COUNT;
  SELECT count(*) INTO active_count
  FROM public.qualification_private_icp_sets WHERE is_active;
  IF activated_count <> 1 OR active_count <> 1 THEN
    RAISE EXCEPTION 'daily ICP activation readback failed';
  END IF;

  SELECT * INTO STRICT persisted
  FROM public.qualification_private_icp_sets
  WHERE set_id=p_set_id AND is_active;
  RETURN pg_catalog.jsonb_build_object(
    'status','activated','set_id',persisted.set_id,
    'inserted',inserted_count=1,'icps',persisted.icps,
    'icp_set_hash',persisted.icp_set_hash,
    'industry_distribution',persisted.industry_distribution
  );
END;
$publish_icp_bank$;
REVOKE ALL ON FUNCTION public.qualification_private_icp_set_publish_v1(
  BIGINT,JSONB,TEXT,JSONB,TIMESTAMPTZ,TIMESTAMPTZ,TEXT
) FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.qualification_private_icp_set_publish_v1(
  BIGINT,JSONB,TEXT,JSONB,TIMESTAMPTZ,TIMESTAMPTZ,TEXT
) TO service_role;

NOTIFY pgrst, 'reload schema';
COMMIT;
