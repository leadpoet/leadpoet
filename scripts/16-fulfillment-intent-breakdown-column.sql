-- Migration 16: add intent_breakdown JSONB to fulfillment_score_consensus
--
-- WHY:
--   The existing fulfillment_score_consensus.intent_details column holds a
--   single synthesized paragraph describing why a lead matches the ICP.
--   That paragraph is great for a TL;DR but customers also want a
--   per-signal breakdown — for each verified intent signal, a short
--   client-ready explanation of how THAT specific signal indicates buying
--   intent for the ICP.
--
--   We could derive the breakdown at read time from intent_signal_mapping
--   alone, but the underlying `description` field is generated upstream as
--   raw evidence text (not client-ready prose).  To get polished per-signal
--   copy we need an LLM pass, and we already make one such pass after
--   consensus closes to produce intent_details.  Extending that single
--   pass to return BOTH the overall paragraph and a per-signal array
--   costs no extra round-trip and roughly halves the per-signal cost
--   compared to a second call.
--
--   The breakdown JSONB stores ONLY the three fields the renderer needs.
--   Source / date / url / snippet / score for each signal already live in
--   intent_signal_mapping and are looked up at read time via source_index.
--   No duplication.
--
-- SHAPE:
--   {
--     "per_signal": [
--       {"source_index": 0, "icp_signal": "Hiring AI engineers",
--        "details": "Company posted 3 ML Engineer roles in the last 30 days..."},
--       {"source_index": 3, "icp_signal": "Series B funding", "details": "..."}
--     ]
--   }
--
--   "source_index" is the 0-based position in the ORIGINAL
--   intent_signal_mapping array — readers do
--   intent_signal_mapping[per_signal[i].source_index] to get raw
--   source/date/url/snippet/score for that signal.  Storing source_index
--   separately is what lets the breakdown skip failed signals while still
--   pointing at the right raw row (otherwise filtering shifts positions
--   and the zip lands on the wrong evidence).
--
--   Array order is the public contract: per_signal[] is emitted in
--   canonical (input-numbering) order, so consumers iterate sequentially.
--
-- ONLY PASSING SIGNALS:
--   per_signal contains entries ONLY for signals where after_decay_score > 0
--   (i.e. signals that actually contributed to the lead's intent score).
--   Failed signals are filtered out before the LLM ever sees them, so we
--   neither pay for LLM tokens to describe them nor show customers
--   explanations for evidence that didn't qualify.
--
-- PROVENANCE:
--   icp_signal and details are produced by the gateway's post-consensus
--   LLM pass over the verified signals.  The raw evidence backing each
--   entry (score, source, date, url, snippet) is consensus-validated and
--   lives in intent_signal_mapping; readers display both side-by-side.
--
-- FORWARD-COMPATIBLE FIELDS NOT INCLUDED (yet):
--   icp_hash, generated_at — both useful for staleness detection /
--   regeneration logic if we later want to track when each breakdown
--   was written or detect that the ICP was edited since.  Skipped now
--   to keep the schema lean; JSONB lets us add them without another
--   migration if a real consumer arrives.

ALTER TABLE fulfillment_score_consensus
    ADD COLUMN IF NOT EXISTS intent_breakdown JSONB;

COMMENT ON COLUMN fulfillment_score_consensus.intent_breakdown IS
    'Per-signal client-ready breakdown generated post-consensus alongside '
    'intent_details.  Shape: {per_signal: [{source_index, icp_signal, '
    'details}]}.  Readers zip per_signal[i] with '
    'intent_signal_mapping[per_signal[i].source_index] for raw evidence.';
