# Arena contacts

New rounds can opt into `contact_policy: "contacts_v1"`. This requires
`arena_integrity_v1`, the `qualification_contacts_v3` scorer, and output schema
`leadpoet.lab_arena.output.v2`. The gateway freezes these choices into the round.
Existing rounds keep their original schemas and scoring.

## Miner contract

`run_icp(icp) -> list[dict]` is unchanged. When the input contains
`contact_policy: "contacts_v1"`, add one `contact` to each company:

```json
{
  "contact": {
    "full_name": "Jane Doe",
    "role": "Chief Technology Officer",
    "linkedin_url": "https://www.linkedin.com/in/jane-doe/",
    "location": {"country": "US", "region": "California", "city": "San Francisco"},
    "email": "jane@example.com",
    "email_source": {
      "provider": "harvestapi",
      "tool": "harvestapi_get_profile",
      "record_id": "provider-profile-id"
    }
  }
}
```

This is a format example. Country is required; region and city are optional and
must be supported when supplied. Contact geography is separate from company
headquarters. The ICP provides `target_roles`, optional `target_seniority`, and
`contact_geography` with `countries`, `regions`, and `cities` lists. An empty
geography list imposes no constraint for that component.

`email_source` requires a provider record ID or a `broker_call_id` from the
same execution's HarvestAPI profile lookup. If both are supplied, both must
match. A source name, guessed email pattern, or miner-written provider response
is insufficient. Provider attribution establishes a supported person/email
association; it does not prove control of the mailbox.

The initial supported source is HarvestAPI `harvestapi_get_profile` with
`findEmail: "true"`. Discovery can use `harvestapi_search_leads`. Source support
is a small code-owned registry; adding another provider requires a corresponding
verifier adapter rather than trusting an arbitrary tool named in output.

## Independent verification

The gateway resolves broker references against the scored execution's ledger.
The verifier can re-fetch record-ID sources through Deepline. It checks the
LinkedIn person, name, current employer, reported title, requested role and
seniority, contact location, and exact email returned by the provider.
An explicit former-position flag fails the employer check. A second current
title cannot qualify a different submitted title.

ZeroBounce is the primary email check. Both **valid** and **catch-all** pass.
Catch-all is retained as its own status and does not need to become deliverable.
An inconclusive result may use BounceBan, with bounded polling for pending jobs.
BounceBan's `result` is the verdict; `status: "success"` alone is not one.
Explicit invalid, disposable, abuse, and do-not-mail verdicts fail. Provider
errors and exhausted pending jobs are incomplete judging and follow the
existing infrastructure retry path.

Company scoring runs first. A contact failure makes only its company's final
score zero and removes that company from the qualified-slot budget denominator.
It adds no new false-positive penalty. Existing company-fit and intent penalties,
company deduplication, first-five limits, and promotion rules remain in effect.
Missing or malformed contacts are row failures. Invalid JSON or a broken
company-level output contract still rejects the output.

Contact fields, requirements, and source evidence affect judgment-cache identity.
Incidental call IDs do not buy another judgment. Cache entries remain scoped to
the round, scorer image, and independent validator authority.

Published results include the submitted contacts under `outputs` and bounded
verdicts under `contact_verifications`, keyed by execution run and company index.
Raw provider responses stay out of public verdicts. Existing disclosure timing
applies to both.

## Activation

1. Apply `scripts/215-lab-arena-contacts.sql` after migrations 211–214. It extends
   existing JSON qualification records and their guards; no historical rewrite
   or separate contacts table is needed.
2. Install matching gateway, runner, scorer image, and contact-capable baseline.
   Keep the provider credential broker and the baseline's transport.
3. Enable `LAB_ARENA_CONTACTS_GENERATION_ENABLED=true` for future ICP generation.
   Review the generated buyer roles before activation. A contact round rejects
   a bank without contact policy and nonempty target roles.
4. Set `LAB_ARENA_CONTACTS_FROM` to a timezone-qualified timestamp with existing
   `LAB_ARENA_INTEGRITY_FROM` configured. Only newly created rounds whose
   **submission-open time** is at or after the contact timestamp opt in. This
   announces the contract before intake begins. Leave it unset until deployment
   and activation are approved.

The protected configuration helper supports `--contacts-generation enabled`
and `--contacts-from TIMESTAMP` as separate scopes. Check each first, then use
its existing authorized `--apply` mode. Generation alone does not change the
output contract of a company-only round. Do not activate contacts before the
matching scorer image and baseline are installed and live verification passes.

Keep both activation settings unset for local compatibility tests. For rollback,
stop creating new contact rounds and finish existing ones with their frozen
policy. Do not downgrade a running round. Provider quotas and sourcing/judging
budgets remain the existing round limits; live coverage and latency should be
measured before changing them.
