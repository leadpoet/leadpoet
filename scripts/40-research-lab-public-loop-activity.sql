-- Research Lab public loop activity board.
--
-- Deployment policy:
--   * Apply after scripts 36, 37, 38, and 39.
--   * Adds a sanitized miner-facing projection of Research Lab loop activity.
--   * Does not change ticket, payment, auto-research, scoring, validator
--     verification, reimbursement, promotion, or weight-setting flows.
--   * No anon/authenticated grants are created.
--   * No raw OpenRouter keys, service-role keys, private repo material, hidden
--     ICP plaintext, judge prompts, private image refs, candidate patch
--     manifests, proxy credentials, credential URLs, or raw provider secrets
--     may be stored here.

BEGIN;

CREATE OR REPLACE FUNCTION public.prevent_research_lab_append_only_mutation()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
BEGIN
    RAISE EXCEPTION
        '% is append-only; write a correction or tombstone row instead',
        TG_TABLE_NAME;
END;
$$;

REVOKE ALL ON FUNCTION public.prevent_research_lab_append_only_mutation()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.prevent_research_lab_append_only_mutation()
    TO service_role;

CREATE TABLE IF NOT EXISTS public.research_lab_public_loop_cards (
    card_id                         TEXT        PRIMARY KEY
                                                    CHECK (card_id ~ '^public_loop_card:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'),
    schema_version                  TEXT        NOT NULL DEFAULT '1.0'
                                                    CHECK (schema_version = '1.0'),
    ticket_id                       UUID        NOT NULL
                                                    REFERENCES public.research_loop_tickets(ticket_id)
                                                    ON DELETE RESTRICT,
    miner_hotkey                    TEXT        NOT NULL CHECK (miner_hotkey <> ''),
    research_area                   TEXT        NOT NULL DEFAULT 'generalist' CHECK (
                                                    research_area !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|\\.dkr\\.ecr\\.|image_digest|private_model_manifest_doc|candidate_patch_manifest|proxy[_-]?url|://[^/]+:[^/@]+@)'
                                                    ),
    research_focus_summary          TEXT        NOT NULL DEFAULT '' CHECK (
                                                    research_focus_summary !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|\\.dkr\\.ecr\\.|image_digest|private_model_manifest_doc|candidate_patch_manifest|proxy[_-]?url|://[^/]+:[^/@]+@)'
                                                    ),
    topic_tags                      JSONB       NOT NULL DEFAULT '[]'::JSONB CHECK (
                                                    jsonb_typeof(topic_tags) = 'array'
                                                    AND topic_tags::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|\\.dkr\\.ecr\\.|image_digest|private_model_manifest_doc|candidate_patch_manifest|proxy[_-]?url|://[^/]+:[^/@]+@)'
                                                    ),
    topic_signature_hash            TEXT        NOT NULL CHECK (topic_signature_hash ~ '^sha256:[0-9a-f]{64}$'),
    card_doc                        JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                                                    jsonb_typeof(card_doc) = 'object'
                                                    AND card_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|\\.dkr\\.ecr\\.|image_digest|private_model_manifest_doc|candidate_patch_manifest|proxy[_-]?url|://[^/]+:[^/@]+@)'
                                                    ),
    card_hash                       TEXT        NOT NULL UNIQUE CHECK (card_hash ~ '^sha256:[0-9a-f]{64}$'),
    anchored_hash                   TEXT        NOT NULL UNIQUE CHECK (anchored_hash = card_hash),
    created_at                      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (ticket_id)
);

CREATE TABLE IF NOT EXISTS public.research_lab_public_loop_card_events (
    event_id                        UUID        PRIMARY KEY,
    event_ref                       TEXT        NOT NULL UNIQUE
                                                    CHECK (event_ref ~ '^public_loop_card_event:[0-9a-f]{64}$'),
    schema_version                  TEXT        NOT NULL DEFAULT '1.0'
                                                    CHECK (schema_version = '1.0'),
    card_id                         TEXT        NOT NULL
                                                    REFERENCES public.research_lab_public_loop_cards(card_id)
                                                    ON DELETE RESTRICT,
    ticket_id                       UUID        NOT NULL
                                                    REFERENCES public.research_loop_tickets(ticket_id)
                                                    ON DELETE RESTRICT,
    run_id                          UUID,
    receipt_id                      UUID,
    seq                             INTEGER     NOT NULL CHECK (seq >= 0),
    event_type                      TEXT        NOT NULL CHECK (
                                                    event_type IN (
                                                        'submitted',
                                                        'queued',
                                                        'running',
                                                        'candidate_generation_complete',
                                                        'scoring',
                                                        'scored',
                                                        'promotion_passed',
                                                        'promoted',
                                                        'failed',
                                                        'tombstoned'
                                                    )),
    outcome_label                   TEXT        NOT NULL CHECK (
                                                    outcome_label IN (
                                                        'submitted',
                                                        'queued',
                                                        'running',
                                                        'candidate_generation_complete',
                                                        'scoring',
                                                        'scored_no_gain',
                                                        'scored_promising',
                                                        'promotion_passed',
                                                        'promoted',
                                                        'failed'
                                                    )),
    outcome_band                    TEXT        NOT NULL CHECK (
                                                    outcome_band IN (
                                                        'pending',
                                                        'no_gain',
                                                        'small_gain',
                                                        'passed_threshold',
                                                        'promoted',
                                                        'failed'
                                                    )),
    topic_tags                      JSONB       NOT NULL DEFAULT '[]'::JSONB CHECK (
                                                    jsonb_typeof(topic_tags) = 'array'
                                                    AND topic_tags::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|\\.dkr\\.ecr\\.|image_digest|private_model_manifest_doc|candidate_patch_manifest|proxy[_-]?url|://[^/]+:[^/@]+@)'
                                                    ),
    topic_signature_hash            TEXT        NOT NULL CHECK (topic_signature_hash ~ '^sha256:[0-9a-f]{64}$'),
    candidate_count                 INTEGER     NOT NULL DEFAULT 0 CHECK (candidate_count >= 0),
    scored_candidate_count          INTEGER     NOT NULL DEFAULT 0 CHECK (scored_candidate_count >= 0),
    best_candidate_public_summary   TEXT        NOT NULL DEFAULT '' CHECK (
                                                    best_candidate_public_summary !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|\\.dkr\\.ecr\\.|image_digest|private_model_manifest_doc|candidate_patch_manifest|proxy[_-]?url|://[^/]+:[^/@]+@)'
                                                    ),
    last_activity_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_doc                       JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                                                    jsonb_typeof(event_doc) = 'object'
                                                    AND event_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|\\.dkr\\.ecr\\.|image_digest|private_model_manifest_doc|candidate_patch_manifest|proxy[_-]?url|://[^/]+:[^/@]+@)'
                                                    ),
    anchored_hash                   TEXT        NOT NULL UNIQUE CHECK (anchored_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at                      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_lab_public_loop_card_events_seq_key UNIQUE (card_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_research_lab_public_loop_cards_ticket
    ON public.research_lab_public_loop_cards(ticket_id);
CREATE INDEX IF NOT EXISTS idx_research_lab_public_loop_cards_miner
    ON public.research_lab_public_loop_cards(miner_hotkey, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_public_loop_cards_topic
    ON public.research_lab_public_loop_cards(topic_signature_hash, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_public_loop_cards_tags
    ON public.research_lab_public_loop_cards USING GIN (topic_tags jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_research_lab_public_loop_card_events_latest
    ON public.research_lab_public_loop_card_events(card_id, seq DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_public_loop_card_events_status
    ON public.research_lab_public_loop_card_events(outcome_label, outcome_band, last_activity_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_public_loop_card_events_topic
    ON public.research_lab_public_loop_card_events(topic_signature_hash, last_activity_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_public_loop_card_events_tags
    ON public.research_lab_public_loop_card_events USING GIN (topic_tags jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_research_lab_public_loop_card_events_activity
    ON public.research_lab_public_loop_card_events(last_activity_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_public_loop_card_events_run
    ON public.research_lab_public_loop_card_events(run_id, created_at DESC);

DROP TRIGGER IF EXISTS prevent_research_lab_public_loop_cards_mutation
    ON public.research_lab_public_loop_cards;
CREATE TRIGGER prevent_research_lab_public_loop_cards_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_public_loop_cards
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

DROP TRIGGER IF EXISTS prevent_research_lab_public_loop_card_events_mutation
    ON public.research_lab_public_loop_card_events;
CREATE TRIGGER prevent_research_lab_public_loop_card_events_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_public_loop_card_events
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

REVOKE ALL ON TABLE public.research_lab_public_loop_cards FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_public_loop_card_events FROM anon, authenticated;
GRANT SELECT, INSERT ON TABLE public.research_lab_public_loop_cards TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_lab_public_loop_card_events TO service_role;

ALTER TABLE public.research_lab_public_loop_cards ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_public_loop_card_events ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_read ON public.research_lab_public_loop_cards;
CREATE POLICY service_role_read ON public.research_lab_public_loop_cards
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_public_loop_cards;
CREATE POLICY service_role_insert ON public.research_lab_public_loop_cards
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_read ON public.research_lab_public_loop_card_events;
CREATE POLICY service_role_read ON public.research_lab_public_loop_card_events
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_public_loop_card_events;
CREATE POLICY service_role_insert ON public.research_lab_public_loop_card_events
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

CREATE OR REPLACE VIEW public.research_lab_public_loop_card_current
WITH (security_invoker = true) AS
SELECT
    c.*,
    e.event_ref AS current_event_ref,
    e.seq AS current_event_seq,
    e.event_type AS current_event_type,
    e.outcome_label AS current_outcome_label,
    e.outcome_band AS current_outcome_band,
    e.topic_tags AS current_topic_tags,
    e.topic_signature_hash AS current_topic_signature_hash,
    e.run_id AS current_run_id,
    e.receipt_id AS current_receipt_id,
    e.candidate_count AS current_candidate_count,
    e.scored_candidate_count AS current_scored_candidate_count,
    e.best_candidate_public_summary AS current_best_candidate_public_summary,
    e.last_activity_at AS current_last_activity_at,
    e.event_doc AS current_event_doc,
    e.anchored_hash AS current_event_hash,
    e.created_at AS current_status_at
FROM public.research_lab_public_loop_cards c
LEFT JOIN LATERAL (
    SELECT *
    FROM public.research_lab_public_loop_card_events e
    WHERE e.card_id = c.card_id
    ORDER BY e.seq DESC, e.created_at DESC
    LIMIT 1
) e ON TRUE;

REVOKE ALL ON TABLE public.research_lab_public_loop_card_current FROM anon, authenticated;
GRANT SELECT ON TABLE public.research_lab_public_loop_card_current TO service_role;

COMMENT ON TABLE public.research_lab_public_loop_cards IS
    'Append-only sanitized miner-facing Research Lab public activity cards, one per ticket.';
COMMENT ON TABLE public.research_lab_public_loop_card_events IS
    'Append-only sanitized Research Lab public activity lifecycle and outcome events.';
COMMENT ON VIEW public.research_lab_public_loop_card_current IS
    'Latest sanitized Research Lab public loop activity status by card.';

COMMIT;
