-- Expand the SOURCE_ADD catalog taxonomy for B2B/GTM evidence providers.
-- Existing values remain valid; this only widens the accepted set.
--
-- This ALTER briefly takes an ACCESS EXCLUSIVE lock. The short lock timeout
-- makes the migration fail instead of waiting behind long-running traffic.

BEGIN;

SET LOCAL lock_timeout = '5s';

ALTER TABLE public.research_lab_source_catalog
    DROP CONSTRAINT IF EXISTS research_lab_source_catalog_source_kind_check;

ALTER TABLE public.research_lab_source_catalog
    ADD CONSTRAINT research_lab_source_catalog_source_kind_check
    CHECK (
        source_kind IN (
            'web',
            'filing',
            'news',
            'registry',
            'procurement',
            'social',
            'hiring',
            'tech_stack',
            'funding',
            'firmographic',
            'people',
            'intent',
            'reviews',
            'events'
        )
    ) NOT VALID;

ALTER TABLE public.research_lab_source_catalog
    VALIDATE CONSTRAINT research_lab_source_catalog_source_kind_check;

COMMIT;
