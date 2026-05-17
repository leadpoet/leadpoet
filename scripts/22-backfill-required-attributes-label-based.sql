-- 22-backfill-required-attributes-label-based.sql
-- Re-apply required_attributes by internal_label (instead of explicit UUIDs).
--
-- Background: file 21-* targeted specific request_ids that were null on 2026-05-14/15.
-- Because lifecycle.py was dropping required_attributes on every recycle, the same
-- ICPs spawned dozens of new null successor rows between then and 2026-05-17 18:55Z.
-- After the lifecycle.py fix landed (commit 46fdca9a), this label-based backfill
-- seeded every then-active null row for the 7 known clients in one pass. From here
-- forward, the recycle code carries the attribute into every successor, so the SQL
-- below should be idempotent and rarely match anything.
--
-- The label-based WHERE is deliberate: it covers any future row that somehow appears
-- with null attrs for these labels (e.g., if a buyer creates a fresh request that
-- predates the api.py persist-on-insert fix being live, or if a manual DB tweak
-- nukes the column on a row). Always gated by `required_attributes IS NULL` so it's
-- idempotent.
--
-- Original deep-verify of each attribute block lives in file 21-*. This file just
-- broadens the WHERE clause; it doesn't change wording.

BEGIN;

-- ICP 1 — 5 Kyle Big (LATAM Embedded Engineers)
UPDATE fulfillment_requests
SET required_attributes = '{
  "company": [
    "The company is private-equity backed (per the PE-backed B2B SaaS buyer requirement)",
    "The company''s primary product is B2B SaaS — not consumer software, hardware, or a services firm",
    "The company does not currently operate an established Latin American engineering office"
  ],
  "contact": []
}'::jsonb
WHERE internal_label = '5 Kyle Big' AND required_attributes IS NULL;

-- ICP 3 — Bruce Callahan 5 (Industrial Valves, WA State)
UPDATE fulfillment_requests
SET required_attributes = '{
  "company": [
    "The company is headquartered in or has primary operations in Washington State, United States",
    "The company is either a US government agency (federal, state, or local) OR a mechanical or industrial contracting firm performing physical construction or facility work",
    "The company''s work involves industrial mechanical systems (HVAC, piping, plumbing, water treatment, process plants, or infrastructure) — i.e., it would actually consume valves and flow-control products"
  ],
  "contact": []
}'::jsonb
WHERE internal_label = 'Bruce Callahan 5' AND required_attributes IS NULL;

-- ICP 4 — Leadpoet 20 Large ICP (B2B GTM Intent)
UPDATE fulfillment_requests
SET required_attributes = '{
  "company": [
    "The company sells to other businesses (B2B), not to consumers (B2C)",
    "The company operates a high-ticket or enterprise sales motion requiring multi-touch human-led sales (presence of AE/SDR teams, not fully self-serve checkout for the primary product)"
  ],
  "contact": []
}'::jsonb
WHERE internal_label = 'Leadpoet 20 Large ICP' AND required_attributes IS NULL;

-- ICP 6 — 5 Stan (French PropTech REST API)
UPDATE fulfillment_requests
SET required_attributes = '{
  "company": [
    "The company''s core product is software, API, or web/mobile technology serving the real-estate industry — not a traditional brokerage, REIT, property manager, or real-estate consultancy",
    "The company has technical buyers (developers, engineers, product managers) as its primary integration audience — meaning a B2B technical product, API, or SDK, not a consumer-facing end-user portal"
  ],
  "contact": []
}'::jsonb
WHERE internal_label = '5 Stan' AND required_attributes IS NULL;

-- ICP 8 — 9 Vicks (Australian Transport HR Platform, Vic/Tas)
UPDATE fulfillment_requests
SET required_attributes = '{
  "company": [
    "The company''s core business is physical transport, freight, logistics, warehousing, 3PL, courier, cold-chain, or distribution operations",
    "The company employs a distributed frontline / non-desk workforce (drivers, warehouse, depot, or field workers) as a significant portion of its headcount",
    "The company is headquartered in or operates primarily out of Victoria or Tasmania, Australia"
  ],
  "contact": []
}'::jsonb
WHERE internal_label = '9 Vicks' AND required_attributes IS NULL;

-- ICP 9 — Daniel iMove 10 (UK Suffolk Removals)
UPDATE fulfillment_requests
SET required_attributes = '{
  "company": [
    "The company is headquartered in or operates primarily within Suffolk, England",
    "The company''s day-to-day business naturally generates removals or relocation demand — estate/letting agents, conveyancers, or housing associations coordinating client moves, OR corporate HR / relocation firms managing employee relocations as a recurring task"
  ],
  "contact": []
}'::jsonb
WHERE internal_label = 'Daniel iMove 10' AND required_attributes IS NULL;

-- ICP 10 — Adedeji 5 (Retirement Income IUL, NJ/PA)
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
WHERE internal_label = 'Adedeji 5' AND required_attributes IS NULL;

-- Verify: should print zero null rows across all 7 labels' active requests
SELECT
  internal_label,
  COUNT(*) FILTER (WHERE required_attributes IS NOT NULL) AS with_attrs,
  COUNT(*) FILTER (WHERE required_attributes IS NULL)     AS still_null,
  COUNT(*)                                                AS active_total
FROM fulfillment_requests
WHERE internal_label IN (
  '5 Kyle Big', 'Bruce Callahan 5', 'Leadpoet 20 Large ICP',
  '5 Stan', '9 Vicks', 'Daniel iMove 10', 'Adedeji 5'
)
AND status IN ('pending','open','continued_open','commit_closed','scoring','partially_fulfilled')
GROUP BY internal_label
ORDER BY internal_label;

COMMIT;
