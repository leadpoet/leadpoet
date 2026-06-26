-- Research Lab: disable new patch-only candidate artifacts.
--
-- Deployment policy:
--   * Apply after script 46.
--   * Existing legacy patch rows remain readable for historical compatibility.
--   * New candidate artifacts must be image_build rows with signed candidate
--     model manifests.

BEGIN;

ALTER TABLE public.research_lab_candidate_artifacts
    ALTER COLUMN candidate_kind DROP DEFAULT;

CREATE OR REPLACE FUNCTION public.guard_research_lab_candidate_artifact_image_build()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
BEGIN
    IF NEW.candidate_kind IS DISTINCT FROM 'image_build' THEN
        RAISE EXCEPTION
            'research_lab_candidate_artifact_kind_conflict: new candidate artifacts must be image_build, got %',
            COALESCE(NEW.candidate_kind, '<null>')
            USING ERRCODE = '23514';
    END IF;

    IF NEW.candidate_model_manifest_hash IS NULL
       OR NEW.candidate_model_manifest_doc IS NULL
       OR NEW.candidate_source_diff_hash IS NULL THEN
        RAISE EXCEPTION
            'research_lab_candidate_artifact_kind_conflict: image_build candidate missing manifest or source diff hash'
            USING ERRCODE = '23514';
    END IF;

    RETURN NEW;
END;
$$;

REVOKE ALL ON FUNCTION public.guard_research_lab_candidate_artifact_image_build()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.guard_research_lab_candidate_artifact_image_build()
    TO service_role;

DROP TRIGGER IF EXISTS guard_research_lab_candidate_artifact_image_build_insert
    ON public.research_lab_candidate_artifacts;
CREATE TRIGGER guard_research_lab_candidate_artifact_image_build_insert
    BEFORE INSERT ON public.research_lab_candidate_artifacts
    FOR EACH ROW EXECUTE FUNCTION public.guard_research_lab_candidate_artifact_image_build();

COMMENT ON FUNCTION public.guard_research_lab_candidate_artifact_image_build() IS
    'Rejects new patch-only Research Lab candidate artifacts; auto-research must produce image_build candidates.';

COMMIT;

-- Smoke checks after applying:
--
--   SELECT tgname
--   FROM pg_trigger
--   WHERE tgname = 'guard_research_lab_candidate_artifact_image_build_insert';
