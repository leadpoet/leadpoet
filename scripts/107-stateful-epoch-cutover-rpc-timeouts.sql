-- Give only the one-time stateful epoch cutover RPC family enough time to
-- audit, lock, bind, stage, and activate the complete epoch namespace. Normal
-- gateway requests retain the role-level short statement timeout.
-- Apply after migration 106 and before cutover.

BEGIN;

ALTER FUNCTION public.research_lab_stateful_subnet_epoch_cutover_preflight_v1(
    TEXT, TEXT
) SET statement_timeout = '120s';

ALTER FUNCTION public.research_lab_stateful_subnet_epoch_cutover_fence_v1(
    TEXT, INTEGER, INTEGER, INTEGER
) SET statement_timeout = '120s';

ALTER FUNCTION public.research_lab_stateful_subnet_epoch_cutover_bind_v1(
    TEXT, TEXT, TEXT, TEXT
) SET statement_timeout = '120s';

ALTER FUNCTION public.research_lab_stateful_subnet_epoch_cutover_bind_v2(
    TEXT, TEXT, TEXT, TEXT
) SET statement_timeout = '120s';

ALTER FUNCTION public.research_lab_stateful_subnet_epoch_stage_v1(
    JSONB, JSONB
) SET statement_timeout = '120s';

ALTER FUNCTION public.research_lab_stateful_subnet_epoch_stage_v2(
    JSONB, JSONB
) SET statement_timeout = '120s';

ALTER FUNCTION public.research_lab_stateful_subnet_epoch_activate_v1(
    TEXT, BOOLEAN
) SET statement_timeout = '120s';

NOTIFY pgrst, 'reload schema';

COMMIT;
