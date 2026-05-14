# Model Evaluation Rules

## Purpose
This document defines every check to perform when evaluating a qualification model for hardcoding, gaming, fabrication, or any behavior that makes the model unsuitable for producing genuine leads for clients.

**Critical mindset**: When reviewing model code, assume NOTHING is benign. Code that looks like "good engineering" or "defensive programming" may in fact be gaming the scoring algorithm. If a piece of logic directly affects how the validator scores a lead, and the model author clearly understands the scoring mechanics, treat it as gaming until proven otherwise.

### Applies to both ICP modes (May 2026)
As of May 2026 the model competition supports two ICP modes (`icp.mode`):

* **`lead`** (legacy / default) — model returns a `LeadOutput`
  (one company + contact + intent signals).
* **`company`** (new) — model returns a `CompanyOutput`
  (one company + intent signals; **no contact fields** — no name,
  role, email, seniority, person LinkedIn URL).  Score components
  shift: no decision-maker LLM, ICP-fit LLM is widened to 0-40,
  intent signal scoring is unchanged.

**Every check below applies to BOTH modes** unless explicitly noted.
Hardcoded dates, LLM-assisted fabrication, generic marketing copy,
length padding, etc. are all just as disqualifying for a
`CompanyOutput` as they were for a `LeadOutput`.  Company-mode adds
one *additional* hard gate: the model's `company_website` must
resolve to a live page that mentions the claimed `company_name`
and isn't a parked / for-sale domain (see
`qualification/scoring/company_verification.py`).

---

## 1. Traditional Hardcoding

### 1a. Hardcoded Dates
- **Check for**: Literal date strings like `"2025-01-01"`, `"2026-03-15"` in non-comment code
- **Check for**: Random date generation (`random.randint` + `timedelta`, `random` + `days_ago`)
- **Check for**: Bare year padding (`"2025"` → `"2025-01-01"`, any `-01-01` or `-01/01` pattern)
- **Check for**: Copyright/founded year extraction used AS intent dates (regex for `©`, `copyright`, `founded` feeding into date fields)
- **Why it matters**: Fabricated dates get flagged by the validator's `check_date_precision` which verifies dates actually appear in scraped content

### 1b. Hardcoded Scores/Confidence
- **Check for**: `confidence = 15`, `intent_final = 0.6`, `raw_llm_score = 8` or similar hardcoded numeric values in non-error-handling code
- **Context matters**: `confidence = 0` in error/rejection paths is legitimate. Hardcoded confidence in SCORING paths is gaming.

### 1c. Hardcoded Company/Lead Data
- **Check for**: Literal company names, roles, or lead data in the output construction
- **Check for**: Lookup tables mapping ICP industries to pre-written leads

### 1d. Hardcoded Intent Descriptions
- **Check for**: Static strings like `"hiring signals, funding news"`, `"recently announced expansion"` that get injected regardless of what was actually scraped
- **Check for**: Template strings with company name inserted: `f"{company} is actively expanding..."` without evidence

---

## 2. LLM-Assisted Fabrication (The v3-1 Problem)

This is more subtle than traditional hardcoding. The model scrapes real content but uses the LLM to REWRITE it into something that sounds like intent but isn't.

### 2a. Forced Signal Words
- **Check for**: LLM prompts that REQUIRE specific signal words (e.g., "MUST include one of: hiring, launched, announced, raised, funding, partnership, expansion, acquired")
- **Why it's bad**: The LLM will inject these words into the description even if the scraped content contains no evidence of them. A website's "About Us" page becomes "Company X launched advanced solutions" because the prompt forced the word "launched."
- **What's legitimate**: Using LLM to summarize actual evidence that naturally contains signal words

### 2b. ICP Intent Mismatch
- **Check for**: Does the generated intent signal actually match the ICP's requested intent?
- **Example of gaming**: ICP asks for "Hiring for specific roles" → Model returns "launched advanced diagnostic testing" (completely unrelated intent type)
- **How to check**: Compare `icp_intent_signals` field against the signal's `description`. If the description has zero semantic overlap with the requested intent type, it's fabricated.

### 2c. Generic Marketing Copy as Intent
- **Check for**: Descriptions that are just rephrased company website marketing copy
- **Red flags**: "Company X is committed to innovative solutions", "providing exceptional service", "empowering customers to achieve optimal outcomes"
- **These are NOT intent signals**: They describe what the company does, not what they're actively doing RIGHT NOW that indicates buying intent

### 2d. Description Length Padding
- **Check for**: Models that generate suspiciously uniform description lengths (always 150-200 chars)
- **Check for**: Descriptions that add filler to meet minimum length requirements without adding information

### 2e. LLM Relevance Score Coaching
- **Check for**: LLM prompts that explain the scoring rubric and coach the LLM to give high scores (e.g., "prefer 8-10 if grounding rules below are satisfied", "validator awards highest intent points when evidence matches buyer intent")
- **Why it's bad**: The model is teaching the LLM what the validator rewards, biasing the LLM's relevance assessment upward rather than letting it judge objectively
- **What's legitimate**: Asking the LLM to rate relevance without explaining the downstream scoring consequences

---

## 3. Wrong Company / URL Mismatch

### 3a. Company Name Prefix Matching
- **Check for**: Google/Wikipedia searches that match on partial company names
- **Example**: Lead is "Contour Mortgage" (`contourmortgage.com`) but signal URL is `en.wikipedia.org/wiki/Contour_Airlines` (matched on "Contour" prefix)
- **Example**: Lead is "Forum Health" (`forumhealth.com`) but signal URL is `forumresearch.org` (matched on "Forum" prefix)
- **How to verify**: Compare the lead's `company_website` domain against each signal's `url` domain. If they're from different companies, the signal is invalid.

### 3b. Multiple Signals from Different Companies
- **Check for**: A single lead having signals from 2-3 completely different companies that happen to share a name fragment
- **Red flag**: 3 signals with 3 different domains that don't match the lead's actual website

---

## 4. Date Issues

### 4a. All Dates Are None
- **Check for**: Every signal having `date: None` or empty date
- **Why it matters**: A model that can't extract any dates from scraped content is either not scraping properly or stripping dates
- **Acceptable**: `company_website` and `wikipedia` sources without dates (these are evergreen content)
- **Not acceptable**: ALL signals across ALL leads having no dates

### 4b. Date Doesn't Appear in Content
- **Check for**: The date in the signal not actually appearing in the scraped snippet/URL content
- **The validator checks this**: `check_date_precision` re-fetches the URL and looks for the claimed date in the content

### 4c. Selective Date Dropping to Avoid Time Decay (GAMING)
- **Check for**: Code that extracts a real date from scraped content, then DISCARDS it based on how old it is before submitting the signal
- **Pattern**: The model successfully extracts a date (e.g., `2021-12-22`), then checks if it's older than N days, and if so sets `date = ""` or `date = None` to avoid the validator's time-decay penalty
- **How to detect**: Look for logic like `if age_days > 60: sig_date = ""` or any conditional that sets an already-extracted date back to empty/None
- **Also check for**: Log messages or comments referencing "time decay", "avoid penalty", "drop old date", or similar scoring-aware language
- **Why it's gaming, not defensive programming**: The validator's time-decay multiplier exists to penalize stale signals. If a model finds content from 2021, the correct behavior is to either (a) submit the real date and accept the penalty, or (b) not use that content at all. Submitting the content but hiding the date to dodge the penalty is exploiting a loophole.
- **Applies to ALL source types**: Even for `company_website` where `null` dates are technically allowed, deliberately stripping a known date is gaming. If the content has a date, report it honestly.

### 4d. Boilerplate Date Stripping (Verify Scope)
- **Check for**: Functions that strip "boilerplate" dates (copyright, founded, last-updated) BEFORE date extraction
- **Legitimate**: Removing `"Copyright 2024"` and `"Last updated: Jan 2025"` since these are not content dates
- **Not legitimate**: Stripping ALL dates from certain sections or stripping dates that fall within content paragraphs (not just headers/footers)
- **How to verify**: Read the regex patterns carefully and ensure they only target genuine boilerplate, not article publication dates

---

## 5. Source Type Issues

### 5a. Source Type Inflation
- **Check for**: Labeling scrapes as `"linkedin"` or `"job_board"` when they're actually from `"company_website"`
- **Why it matters**: Different source types have different scoring multipliers

### 5b. Source Type Doesn't Match URL
- **Check for**: `source: "linkedin"` but URL is `https://somecompany.com`
- **Check for**: `source: "job_board"` but URL is a Wikipedia page

### 5c. Source Type Downgrading to Avoid Date Requirements
- **Check for**: Models that classify a `news` or `linkedin` URL as `company_website` to avoid the date requirement for those source types
- **Why it matters**: The validator requires dates for `linkedin`, `news`, `job_board`, and `social_media` sources, but not for `company_website`. Misclassifying a source to avoid the date requirement is gaming.

---

## 6. ICP Echo-Back

### 6a. Product/Service Echo
- **Check for**: The intent description containing the exact `product_service` text from the ICP prompt, word for word
- **Example**: ICP says `product_service: "cloud management upgrades"` → Description says "Company is evaluating cloud management upgrades" (echoed directly)
- **Legitimate**: Description that naturally mentions the product category based on evidence

### 6b. Role Echo
- **Check for**: Signal descriptions that just repeat the target role from the ICP
- **Example**: ICP asks for "VP of Engineering" → Description says "Company is hiring a VP of Engineering" with no actual job posting evidence

---

## 7. Scoring Mechanic Exploitation

This section covers models that demonstrate detailed knowledge of the validator's scoring algorithm and write code specifically to maximize score rather than produce genuine leads.

### 7a. Relevance Score Manipulation
- **Check for**: Code that adjusts relevance scores based on source type or date presence to hit specific thresholds
- **Example**: `if source in DATE_REQUIRED and not date: relevance = min(relevance, 3)` shows the model knows exactly how the validator penalizes dateless signals from certain source types
- **Why it matters**: Models should focus on finding the best content, not on reverse-engineering the scoring formula

### 7b. Time Decay Avoidance (see also 4c)
- **Check for**: Any code that makes date-related decisions based on how the validator's time-decay multiplier works
- **Check for**: Constants like `60`, `90`, `365` days used as thresholds for dropping dates
- **Check for**: Comments or log messages mentioning "time decay", "decay multiplier", or "scoring penalty"

### 7c. Snippet/Description Length Optimization
- **Check for**: Code that pads or truncates descriptions to specific character counts that the model author knows the validator rewards
- **Check for**: Snippet length calculations targeting specific token/character limits

### 7d. Signal Word Stripping (Preemptive Deception)
- **Check for**: Models that run their OWN grounding checks to strip ungrounded signal words BEFORE submission, not because they care about accuracy, but because they know the validator checks for this
- **How to distinguish**: A model that strips ungrounded words and KEEPS the signal (submitting a vaguer description) is gaming. A model that rejects the signal entirely when grounding fails is being honest.
- **Key question**: Does the code strip-and-keep, or reject? Strip-and-keep means the model is laundering a bad description to pass validator checks.

### 7e. Prescreen/Prioritization Based on Scoring Knowledge
- **Check for**: Models that pre-rank lead candidates based on what the validator's scoring formula rewards (e.g., prioritizing news sources over company_website because they know the multiplier)
- **Moderate concern**: Some prioritization is natural (prefer better content), but if the prioritization logic mirrors the validator's exact scoring weights, it shows reverse-engineering

---

## 8. Structural Quality Checks

### 8a. Does the Model Actually Scrape?
- **Check for**: Real HTTP calls to company websites, Google, LinkedIn, etc.
- **Red flag**: Model that returns leads without making any external HTTP calls
- **Verify**: Check for `httpx`, `requests`, `aiohttp` usage with actual URLs

### 8b. Does the Model Use Evidence for Descriptions?
- **Check for**: LLM prompt includes actual scraped content (not just company name)
- **Red flag**: LLM prompt like `"Write an intent signal for {company}"` with no content
- **Legitimate**: LLM prompt like `"Based on the following content about {company}: {scraped_text[:2000]}, write a specific intent signal"`
- **BUT STILL CHECK**: Even with content in the prompt, if the prompt FORCES signal words, the output is fabricated

### 8c. Pipeline Budget vs Quality
- **Check for**: Models with very short pipeline budgets (<15s) that can't do thorough scraping
- **Not a disqualifier on its own**: But combined with generic descriptions, suggests the model is cutting corners

### 8d. Validator Knowledge Fingerprints
- **Check for**: Code that references or mirrors internal validator logic, constant names, or scoring formulas
- **Examples**: Constants named `_DATE_REQUIRED`, `_DATE_NOT_REQUIRED` that match the validator's exact source-type categorization; logic that mirrors `check_date_precision` or `check_signal_word_grounding`
- **Why it matters**: A model that knows the validator's internals is optimizing to pass checks rather than produce genuine leads

---

## 9. Evaluation Checklist

When evaluating a model, check these in order:

1. **Run regex checks**: Hardcoded dates, scores, company names, static intent phrases
2. **Read the LLM prompt**: Does it force signal words? Does it coach the LLM about scoring? Does it include actual scraped content?
3. **Search for scoring-aware code**: Grep for terms like `time_decay`, `relevance_score`, `grounding`, `snippet_overlap`, `signal_word`, `date_required`. If the model has its own versions of the validator's checks, that's a red flag for reverse-engineering.
4. **Check date handling end-to-end**: Trace the full lifecycle of a date: extraction → validation → output. If a date is extracted and then conditionally dropped (not because it's wrong, but because it's old), that's gaming.
5. **Check top 5 leads**:
   - Does each signal's `url` domain match the lead's `company_website` domain?
   - Does the signal `description` relate to the ICP's `intent_signals`?
   - Is the description specific (mentions real products/events) or generic (marketing copy)?
   - Are dates present and plausible?
   - For signals with `date: null`, does the scraped content actually lack dates, or were dates extracted and then dropped?
6. **Check bottom 5 leads**: What's the failure reason? "No lead returned" is fine. All leads failing with the same error suggests a systematic issue.
7. **Cross-reference**: Pick 2-3 signal URLs and manually visit them. Does the content on the page support the claimed description?
8. **Check fabrication rate**: >15% fabrication rate from the validator is a red flag. The validator's own intent verification is catching problems.
9. **Ask "who benefits?"**: For every piece of conditional logic that affects the output, ask: does this make the lead more accurate for the client, or does this make the lead score higher with the validator? If the answer is "scores higher," it's gaming.

---

## 10. Verdict Categories

- **CLEAN**: Model scrapes real content, extracts genuine signals, descriptions match evidence and ICP intent, URLs match lead companies, no scoring-aware optimizations
- **FABRICATION**: Model produces descriptions that sound like intent but aren't grounded in evidence (LLM-assisted fabrication, forced signal words, generic marketing copy)
- **HARDCODING**: Model uses literal hardcoded values (dates, scores, descriptions, company data)
- **GAMING**: Model exploits scoring mechanics (source type inflation, ICP echo-back, wrong company URL matching, selective date dropping, time-decay avoidance, signal word laundering, validator reverse-engineering)
- **BAN-WORTHY**: Repeated, deliberate patterns of fabrication or hardcoding across multiple evaluation runs
