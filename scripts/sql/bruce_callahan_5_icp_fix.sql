-- Bruce Callahan 5: set required_attribute + region + size where missing.
--
-- ICP per the buyer profile:
--   Countries: United States    (company_country)
--   States:    Washington       (company_region)
--   Sizes:     500–1,000        (employee_count = ["501-1000"])
--   Required attribute: niche-vertical "mechanical contractor that procures
--                       industrial valves / flow control" + explicit
--                       exclusion list.
--
-- Targets ALL chain links by `internal_label`.  Filters to active states
-- (`open`, `continued_open`, `pending`) so closed historical rows aren't
-- mutated.  Safe to re-run — uses `jsonb_set` with COALESCE-style guards
-- to set ONLY when missing/empty.
--
-- Run order:
--   1. Inspect current state (Section A)
--   2. Apply updates (Section B) — wrapped in BEGIN/COMMIT for atomicity
--   3. Verify (Section C)

-- ============================================================
-- SECTION A — INSPECT
-- ============================================================
SELECT
    request_id,
    internal_label,
    company,
    status,
    required_attributes,
    icp_details->'company_country' AS company_country,
    icp_details->>'company_region' AS company_region,
    icp_details->'employee_count'  AS employee_count,
    created_at
FROM fulfillment_requests
WHERE internal_label = 'Bruce Callahan 5'
ORDER BY created_at DESC;

-- ============================================================
-- SECTION B — UPDATE (transactional)
-- ============================================================
BEGIN;

-- B.1 Set required_attributes on active chain link(s).
--     `required_attributes` is a dedicated column shaped as
--     {"company": [...], "contact": [...]}.
--     Existing value (if any) is OVERWRITTEN — the prior text was the
--     "or infrastructure" loophole that let electrical / civil firms
--     through.  If you want to preserve prior text, comment out this
--     statement and re-run after manual merge.
UPDATE fulfillment_requests
SET required_attributes = jsonb_build_object(
    'company', jsonb_build_array(
        'The company directly installs OR procures industrial mechanical systems — specifically HVAC, mechanical piping/plumbing, process plant equipment, water/wastewater treatment equipment, or industrial valves and flow-control components — as a core revenue-generating activity. The following do NOT qualify (answer NO): electrical-only / low-voltage-only / technology-only contractors that do NOT also self-perform mechanical, HVAC, or piping work; asphalt paving, road or surface contractors; heavy civil / excavation / site-development firms that do not self-perform mechanical work; environmental science / engineering / planning consultancies that only design or permit; HR / professional-services / staffing / PEO firms; architecture or engineering design-only firms whose clients procure the equipment, not them.'
    ),
    'contact', '[]'::jsonb
)
WHERE internal_label = 'Bruce Callahan 5'
  AND status IN ('open', 'continued_open', 'pending');

-- B.2 Set company_country = ["United States"] inside icp_details
--     IF the field is missing or an empty array.
UPDATE fulfillment_requests
SET icp_details = jsonb_set(
    icp_details,
    '{company_country}',
    '["United States"]'::jsonb,
    true   -- create_missing
)
WHERE internal_label = 'Bruce Callahan 5'
  AND status IN ('open', 'continued_open', 'pending')
  AND (
      icp_details->'company_country' IS NULL
      OR icp_details->'company_country' = '[]'::jsonb
  );

-- B.3 Set company_region = "Washington" inside icp_details
--     IF the field is missing or empty string.
UPDATE fulfillment_requests
SET icp_details = jsonb_set(
    icp_details,
    '{company_region}',
    '"Washington"'::jsonb,
    true
)
WHERE internal_label = 'Bruce Callahan 5'
  AND status IN ('open', 'continued_open', 'pending')
  AND (
      icp_details->>'company_region' IS NULL
      OR icp_details->>'company_region' = ''
  );

-- B.4 Set employee_count = ["501-1000"] inside icp_details
--     IF the field is missing or empty array.  Canonical bucket name
--     per gateway/fulfillment/models.py (range_string_to_buckets).
UPDATE fulfillment_requests
SET icp_details = jsonb_set(
    icp_details,
    '{employee_count}',
    '["501-1000"]'::jsonb,
    true
)
WHERE internal_label = 'Bruce Callahan 5'
  AND status IN ('open', 'continued_open', 'pending')
  AND (
      icp_details->'employee_count' IS NULL
      OR icp_details->'employee_count' = '[]'::jsonb
  );

COMMIT;

-- ============================================================
-- SECTION C — VERIFY POST-UPDATE STATE
-- ============================================================
SELECT
    request_id,
    internal_label,
    status,
    required_attributes,
    icp_details->'company_country'   AS company_country,
    icp_details->>'company_region'   AS company_region,
    icp_details->'employee_count'    AS employee_count
FROM fulfillment_requests
WHERE internal_label = 'Bruce Callahan 5'
ORDER BY created_at DESC;
