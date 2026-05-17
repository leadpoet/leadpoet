-- 21-backfill-required-attributes.sql
-- Backfill required_attributes for 19 active/fulfilling NULL rows across 10 distinct ICPs.
-- Attributes hand-composed against each ICP's prompt + product_service + intent_signals +
-- hard filters (industry / sub_industry / country / employee_count / target_roles).
-- Safety guard: WHERE required_attributes IS NULL on every UPDATE.

BEGIN;

-- ICP 1 — LATAM Embedded Engineers (4 rows)
UPDATE fulfillment_requests
SET required_attributes = '{
  "company": [
    "The company is private-equity backed (per the PE-backed B2B SaaS buyer requirement)",
    "The company''s primary product is B2B SaaS — not consumer software, hardware, or a services firm",
    "The company does not currently operate an established Latin American engineering office"
  ],
  "contact": []
}'::jsonb
WHERE request_id IN (
  '4308d06d-09cc-482a-b24d-1db00a7b81bd',
  'b9e496bc-8426-424b-9bd3-e922f2e411c5',
  'ebd65dfb-137f-44ef-a33f-dc1dbfeb8f09',
  'f6668f3b-3383-4c76-bf3f-60511d67889a'
) AND required_attributes IS NULL;

-- ICP 2 — FCL/LCL Ocean Freight (3 rows)
UPDATE fulfillment_requests
SET required_attributes = '{
  "company": [
    "The company is a product-based business — manufacturer, wholesaler, retailer, or distributor of physical goods — not a services-only or software firm",
    "The company maintains its own inventory or warehousing operations rather than being a pure marketplace or dropship reseller that never physically holds goods"
  ],
  "contact": []
}'::jsonb
WHERE request_id IN (
  '2ac3e129-5459-42c3-9e44-132efa990ed3',
  '6e39ce6b-3948-4153-a1e5-73fc044daeb1',
  '98dec714-8c3e-4619-ad98-c2efaaeb529d'
) AND required_attributes IS NULL;

-- ICP 3 — Industrial Valves, Washington State (3 rows)
UPDATE fulfillment_requests
SET required_attributes = '{
  "company": [
    "The company is headquartered in or has primary operations in Washington State, United States",
    "The company is either a US government agency (federal, state, or local) OR a mechanical or industrial contracting firm performing physical construction or facility work",
    "The company''s work involves industrial mechanical systems (HVAC, piping, plumbing, water treatment, process plants, or infrastructure) — i.e., it would actually consume valves and flow-control products"
  ],
  "contact": []
}'::jsonb
WHERE request_id IN (
  '45fa9dfe-7fbd-42c3-812d-950d7985b05b',
  '94e1e314-c83b-44d6-b0ab-0efdcfd58bd5',
  'c421da5b-a49f-49b7-a5b4-9bc8f8cdfbed'
) AND required_attributes IS NULL;

-- ICP 4 — Leadpoet B2B GTM Intent (2 rows)
UPDATE fulfillment_requests
SET required_attributes = '{
  "company": [
    "The company sells to other businesses (B2B), not to consumers (B2C)",
    "The company operates a high-ticket or enterprise sales motion requiring multi-touch human-led sales (presence of AE/SDR teams, not fully self-serve checkout for the primary product)"
  ],
  "contact": []
}'::jsonb
WHERE request_id IN (
  '0d2b4fc2-3c4f-46a6-b36a-d91e6c27771a',
  '19c5e580-0ec4-4236-b13d-fd171f1247d7'
) AND required_attributes IS NULL;

-- ICP 5 — Creative Asset Workspace, Mexico (2 rows)
UPDATE fulfillment_requests
SET required_attributes = '{
  "company": [
    "The company''s primary commercial output is creative or marketing assets produced for paying clients OR for its own consumer/B2B brand (not a creative-adjacent SaaS vendor, not a PR firm doing only text/press releases)",
    "The company is a potential buyer of creative-asset tooling, not a competitor — i.e., it is NOT itself a DAM, file-sharing, or creative-collaboration software vendor"
  ],
  "contact": []
}'::jsonb
WHERE request_id IN (
  '79e3dbf0-3cd8-453e-9096-90457f701f5f',
  'e0ba04c9-ff08-4ec6-ab52-bd216fd582ce'
) AND required_attributes IS NULL;

-- ICP 6 — French PropTech REST API (1 row)
UPDATE fulfillment_requests
SET required_attributes = '{
  "company": [
    "The company''s core product is software, API, or web/mobile technology serving the real-estate industry — not a traditional brokerage, REIT, property manager, or real-estate consultancy",
    "The company has technical buyers (developers, engineers, product managers) as its primary integration audience — meaning a B2B technical product, API, or SDK, not a consumer-facing end-user portal"
  ],
  "contact": []
}'::jsonb
WHERE request_id IN (
  '6496fee6-9f1b-4bf3-8d22-7d9edb2403b7'
) AND required_attributes IS NULL;

-- ICP 7 — HR Payroll Mid-Market Australia, Victoria/Tasmania (1 row)
UPDATE fulfillment_requests
SET required_attributes = '{
  "company": [
    "The company is headquartered in or has primary operations in Victoria or Tasmania, Australia",
    "The company operates multiple physical sites, locations, branches, or business units — not a single-location firm",
    "The company employs a mix of frontline/operational workers and corporate/office workers, not a purely white-collar office"
  ],
  "contact": []
}'::jsonb
WHERE request_id IN (
  '9d253630-4cb0-476e-96a9-a22d99d9701a'
) AND required_attributes IS NULL;

-- ICP 8 — Australian Transport HR Platform, Vic/Tas (1 row)
UPDATE fulfillment_requests
SET required_attributes = '{
  "company": [
    "The company''s core business is physical transport, freight, logistics, warehousing, 3PL, courier, cold-chain, or distribution operations",
    "The company employs a distributed frontline / non-desk workforce (drivers, warehouse, depot, or field workers) as a significant portion of its headcount",
    "The company is headquartered in or operates primarily out of Victoria or Tasmania, Australia"
  ],
  "contact": []
}'::jsonb
WHERE request_id IN (
  'b9ac10b7-37e6-460b-8060-d6485f0063de'
) AND required_attributes IS NULL;

-- ICP 9 — UK Suffolk Removals / "Daniel iMove 10" (7 rows: 1 historical + 6 active chain)
-- Active rows pulled 2026-05-17 18:55Z. Backfilling the chain head (f0edfd0f, continued_open,
-- window_end 19:31Z) alone would be enough now that lifecycle.py carries required_attributes
-- forward on every recycle — but we also backfill the 5 partially_fulfilled predecessors so
-- Tier 2c history is consistent across the whole live chain. Reason: every Daniel iMove 10
-- request shares the same ICP, so the same attributes apply to all rows in the chain.
UPDATE fulfillment_requests
SET required_attributes = '{
  "company": [
    "The company is headquartered in or operates primarily within Suffolk, England",
    "The company''s day-to-day business naturally generates removals or relocation demand — estate/letting agents, conveyancers, or housing associations coordinating client moves, OR corporate HR / relocation firms managing employee relocations as a recurring task"
  ],
  "contact": []
}'::jsonb
WHERE request_id IN (
  '7f6a2490-8743-4a69-b0b6-29389185c710',  -- historical (recycled, May 15)
  'f0edfd0f-c465-44f3-a0b9-e581a17612a9',  -- continued_open  (chain head — next recycle 19:31Z)
  '570e4413-e12d-4628-a880-7b5bc3b0b81b',  -- partially_fulfilled  May 17 16:51
  '8dfc0f5e-6ec1-46a3-8617-2d81975c41d8',  -- partially_fulfilled  May 17 15:22
  '05a66599-82ab-4202-b534-f148b66abcf8',  -- partially_fulfilled  May 17 13:12
  'a57418c0-063f-464d-83bc-ce36f1532378',  -- partially_fulfilled  May 17 11:44
  'd6047530-10e3-4453-aaea-5e5b8c5cf41e'   -- partially_fulfilled  May 17 09:47
) AND required_attributes IS NULL;

-- ICP 10 — Retirement Income IUL, NJ/PA (1 row, person-level ICP)
UPDATE fulfillment_requests
SET required_attributes = '{
  "company": [
    "The company is a corporate W-2 employer with a standard 401(k) plan — not a small partnership, sole proprietorship, or pass-through professional-services firm whose principals are paid as K-1 or 1099"
  ],
  "contact": [
    "The contact is based in or works in New Jersey or Pennsylvania, United States",
    "The contact is a salaried W-2 employee — not a 1099 contractor, freelancer, business owner, partner-track principal taking K-1 distributions, or self-employed practitioner"
  ]
}'::jsonb
WHERE request_id IN (
  'e02fd193-aa93-469d-9b3f-222798369758'
) AND required_attributes IS NULL;

-- Verify row counts
SELECT
  '25 expected' AS expected,
  COUNT(*) FILTER (WHERE required_attributes IS NOT NULL) AS now_populated,
  COUNT(*) FILTER (WHERE required_attributes IS NULL) AS still_null_active
FROM fulfillment_requests
WHERE request_id IN (
  '4308d06d-09cc-482a-b24d-1db00a7b81bd','b9e496bc-8426-424b-9bd3-e922f2e411c5',
  'ebd65dfb-137f-44ef-a33f-dc1dbfeb8f09','f6668f3b-3383-4c76-bf3f-60511d67889a',
  '2ac3e129-5459-42c3-9e44-132efa990ed3','6e39ce6b-3948-4153-a1e5-73fc044daeb1',
  '98dec714-8c3e-4619-ad98-c2efaaeb529d','45fa9dfe-7fbd-42c3-812d-950d7985b05b',
  '94e1e314-c83b-44d6-b0ab-0efdcfd58bd5','c421da5b-a49f-49b7-a5b4-9bc8f8cdfbed',
  '0d2b4fc2-3c4f-46a6-b36a-d91e6c27771a','19c5e580-0ec4-4236-b13d-fd171f1247d7',
  '79e3dbf0-3cd8-453e-9096-90457f701f5f','e0ba04c9-ff08-4ec6-ab52-bd216fd582ce',
  '6496fee6-9f1b-4bf3-8d22-7d9edb2403b7','9d253630-4cb0-476e-96a9-a22d99d9701a',
  'b9ac10b7-37e6-460b-8060-d6485f0063de','7f6a2490-8743-4a69-b0b6-29389185c710',
  'e02fd193-aa93-469d-9b3f-222798369758',
  -- Daniel iMove 10 active chain (added 2026-05-17)
  'f0edfd0f-c465-44f3-a0b9-e581a17612a9','570e4413-e12d-4628-a880-7b5bc3b0b81b',
  '8dfc0f5e-6ec1-46a3-8617-2d81975c41d8','05a66599-82ab-4202-b534-f148b66abcf8',
  'a57418c0-063f-464d-83bc-ce36f1532378','d6047530-10e3-4453-aaea-5e5b8c5cf41e'
);

COMMIT;
