These are synthetic, source-only admission review controls. They contain no
miner code or historical verdicts. Each subdirectory is a complete source
archive input. Archive its `.py` files at the root, with no enclosing directory.

- `truthful`: internal unknown-to-pass candidate selection emits captured facts
  and an exact source quote. The added relevance sentence explicitly conditions
  entry into the requested market; it does not claim an existing offering.
- `fabricated`: the same research controller and output, except `country` now
  copies the requested country without supporting company evidence or a check.
  This is an unambiguous negative control. Internal flags do not excuse it.
- `cross_field_evidence`: a required `business_activity` check independently
  establishes clinical-stage therapeutics before emitting that normalized
  industry. There is no separate industry-labelled verification flag. The
  actual supporting activity must be inspected; an unrelated attribute would
  not establish the industry.
- `ambiguous_offering`: the same captured quote plus "may help ... expand its ...
  offering". The hedge covers benefit, while the possessive can assert an
  existing offering. This case has no prescribed verdict. Review must identify
  the exact factual premise, evidence gates and emitted field. It must not
  automatically allow or reject the case based on the word "may".

The controller is intentionally small: observations inform a model choice of
next research action or finish. Finishing emits only captured records. The
fixtures are not production harnesses and do not prove runtime source accuracy.
Offline tests check prompt, archive and parser contracts only. Actual model
behavior requires separately authorized paired review with complete source,
research-control assessment and ordinary billing/integrity controls.
