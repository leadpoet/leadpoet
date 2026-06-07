"""One-time capture of the CURRENT prompt outputs from
``_build_verification_prompt`` and ``_build_final_judge_prompt``.

Runs BEFORE any refactor changes.  Snapshots become the byte-equality
target the post-refactor dispatcher must reproduce.

Run from repo root::

    python3 scripts/capture_prompt_snapshots.py

Writes one ``<row_id>_verification.txt`` and one ``<row_id>_final_judge.txt``
file per test row into ``tests/snapshots/prompts/``.
"""
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qualification.scoring.intent_verification_three_stage import (
    _build_final_judge_prompt,
    _build_verification_prompt,
)


# 25 representative test rows covering every dispatch path + edge cases.
# Each row is a (id, row_dict, contents_for_final_judge) triple.
CASES: List[Dict[str, Any]] = [
    # ── SOCIAL_POSTING shapes ─────────────────────────────────────────────
    {
        "id": "social_li_personal_handle",
        "row": {
            "id": "sig-1",
            "company": "Courier Health",
            "website": "https://courier.health",
            "company_linkedin": "https://www.linkedin.com/company/courier-health/",
            "contact_linkedin": "https://www.linkedin.com/in/dsigurdson",
            "signal_type": "social_post",
            "claim": "dsigurdson posted on LinkedIn about Courier Health milestones",
            "signal_date": "2026-04-15",
            "_target_signal_text": "Founder or CEO is publicly posting on LinkedIn about company milestones, product launches, or fundraising in the past 60 days",
            "_evidence_type": "SOCIAL_POSTING",
            "claimed_source_urls": [
                "https://www.linkedin.com/posts/dsigurdson_proud-and-excited-to-share-that-courier-health-activity-7340755658008416256-o8Vb",
            ],
        },
        "contents": {
            "results": [
                {
                    "url": "https://www.linkedin.com/posts/dsigurdson_proud-and-excited-to-share-that-courier-health-activity-7340755658008416256-o8Vb",
                    "title": "Dsigurdson on LinkedIn",
                    "text": "Proud and excited to share that Courier Health closed a $20M Series B today. Massive thanks to the team for building the most accessible benefits navigation product on the market.",
                }
            ],
            "statuses": [],
        },
    },
    {
        "id": "social_li_company_handle",
        "row": {
            "id": "sig-2",
            "company": "Accorian",
            "website": "https://accorian.com",
            "company_linkedin": "https://www.linkedin.com/company/accorian/",
            "contact_linkedin": "",
            "signal_type": "social_post",
            "claim": "Accorian posted on LinkedIn about cybersecurity hiring",
            "signal_date": "2026-04-01",
            "_target_signal_text": "Founder or CEO is actively posting on LinkedIn about sales growth, hiring, pipeline, or go-to-market initiatives within the past 90 days",
            "_evidence_type": "SOCIAL_POSTING",
            "claimed_source_urls": [
                "https://www.linkedin.com/posts/accorian_cybersecurity-pentesting-ai-activity-7460349108223053824-RtX2",
            ],
        },
        "contents": {
            "results": [
                {
                    "url": "https://www.linkedin.com/posts/accorian_cybersecurity-pentesting-ai-activity-7460349108223053824-RtX2",
                    "title": "Accorian on LinkedIn",
                    "text": "We continue to lead in cybersecurity, pentesting, and AI risk management. Reach out to learn more about how we protect mid-market firms.",
                }
            ],
            "statuses": [],
        },
    },
    {
        "id": "social_x_tweet",
        "row": {
            "id": "sig-3",
            "company": "Tesla",
            "website": "https://www.tesla.com",
            "company_linkedin": "https://www.linkedin.com/company/tesla-motors/",
            "contact_linkedin": "https://www.linkedin.com/in/elonmusk",
            "signal_type": "social_post",
            "claim": "Elon Musk tweeted about Tesla production",
            "signal_date": "2026-04-20",
            "_target_signal_text": "Founder or CEO is publicly posting on social media about company milestones or product launches",
            "_evidence_type": "SOCIAL_POSTING",
            "claimed_source_urls": [
                "https://x.com/elonmusk/status/1815935559263572173",
            ],
        },
        "contents": {
            "results": [
                {
                    "url": "https://x.com/elonmusk/status/1815935559263572173",
                    "title": "Elon Musk on X",
                    "text": "Tesla Optimus production starting this year.",
                }
            ],
            "statuses": [],
        },
    },
    {
        "id": "social_li_pulse",
        "row": {
            "id": "sig-4",
            "company": "Salesforce",
            "website": "https://www.salesforce.com",
            "company_linkedin": "https://www.linkedin.com/company/salesforce/",
            "contact_linkedin": "https://www.linkedin.com/in/marcbenioff",
            "signal_type": "social_post",
            "claim": "Marc Benioff Pulse article about Salesforce roadmap",
            "signal_date": "2026-03-30",
            "_target_signal_text": "CEO is publishing long-form posts about company strategy on LinkedIn within the past 90 days",
            "_evidence_type": "SOCIAL_POSTING",
            "claimed_source_urls": [
                "https://www.linkedin.com/pulse/agentforce-future-marc-benioff",
            ],
        },
        "contents": {
            "results": [
                {
                    "url": "https://www.linkedin.com/pulse/agentforce-future-marc-benioff",
                    "title": "Agentforce: The future is autonomous",
                    "text": "I am excited to share our vision for Agentforce, the largest platform shift in our history.",
                }
            ],
            "statuses": [],
        },
    },
    # ── HIRING shapes ────────────────────────────────────────────────────
    {
        "id": "hiring_li_jobs_path",
        "row": {
            "id": "sig-5",
            "company": "Anthropic",
            "website": "https://www.anthropic.com",
            "company_linkedin": "https://www.linkedin.com/company/anthropicresearch/",
            "contact_linkedin": "",
            "signal_type": "hiring",
            "claim": "Anthropic posted SDR job listing",
            "signal_date": "2026-05-01",
            "_target_signal_text": "Company has active job postings for SDR/BDR roles in the past 60 days",
            "_evidence_type": "HIRING",
            "claimed_source_urls": [
                "https://www.linkedin.com/jobs/view/3812345678",
            ],
        },
        "contents": {
            "results": [
                {
                    "url": "https://www.linkedin.com/jobs/view/3812345678",
                    "title": "SDR – Anthropic",
                    "text": "We are looking for an experienced Sales Development Representative to join our team.",
                }
            ],
            "statuses": [],
        },
    },
    {
        "id": "hiring_li_company_jobs",
        "row": {
            "id": "sig-6",
            "company": "Stripe",
            "website": "https://stripe.com",
            "company_linkedin": "https://www.linkedin.com/company/stripe/",
            "contact_linkedin": "",
            "signal_type": "hiring",
            "claim": "Stripe has open AE positions",
            "signal_date": "2026-05-10",
            "_target_signal_text": "Stripe is hiring Account Executives in the past 60 days",
            "_evidence_type": "HIRING",
            "claimed_source_urls": [
                "https://www.linkedin.com/company/stripe/jobs",
            ],
        },
        "contents": {
            "results": [
                {
                    "url": "https://www.linkedin.com/company/stripe/jobs",
                    "title": "Stripe jobs",
                    "text": "Stripe – 32 open positions including Account Executive – Enterprise.",
                }
            ],
            "statuses": [],
        },
    },
    # ── FUNDING shape ─────────────────────────────────────────────────────
    {
        "id": "funding_techcrunch",
        "row": {
            "id": "sig-7",
            "company": "Mercury",
            "website": "https://mercury.com",
            "company_linkedin": "https://www.linkedin.com/company/mercury/",
            "contact_linkedin": "",
            "signal_type": "funding",
            "claim": "Mercury raised Series C",
            "signal_date": "2026-04-15",
            "_target_signal_text": "Company closed Series A or later funding round in the past 12 months",
            "_evidence_type": "FUNDING",
            "claimed_source_urls": [
                "https://techcrunch.com/2026/04/15/mercury-series-c/",
            ],
        },
        "contents": {
            "results": [
                {
                    "url": "https://techcrunch.com/2026/04/15/mercury-series-c/",
                    "title": "Mercury raises $300M Series C at $3B valuation",
                    "text": "Mercury, the SaaS bank for startups, announced today the close of a $300M Series C led by Sequoia.",
                }
            ],
            "statuses": [],
        },
    },
    # ── Generic / no evidence_type ────────────────────────────────────────
    {
        "id": "generic_no_evidence_type",
        "row": {
            "id": "sig-8",
            "company": "Plaid",
            "website": "https://plaid.com",
            "company_linkedin": "https://www.linkedin.com/company/plaid-/",
            "contact_linkedin": "",
            "signal_type": "unknown",
            "claim": "Plaid mentioned in product launch story",
            "signal_date": "2026-04-01",
            "_target_signal_text": "Company launched a new product in the past 6 months",
            # NO _evidence_type set
            "claimed_source_urls": [
                "https://plaid.com/blog/identity-verification-launch/",
            ],
        },
        "contents": {
            "results": [
                {
                    "url": "https://plaid.com/blog/identity-verification-launch/",
                    "title": "Introducing Identity Verification",
                    "text": "Today we're launching our most-requested feature: Plaid Identity Verification.",
                }
            ],
            "statuses": [],
        },
    },
    {
        "id": "generic_empty_evidence_type",
        "row": {
            "id": "sig-9",
            "company": "Vanta",
            "website": "https://vanta.com",
            "company_linkedin": "https://www.linkedin.com/company/trustvanta/",
            "contact_linkedin": "",
            "signal_type": "expansion",
            "claim": "Vanta expanded to UK",
            "signal_date": "2026-05-20",
            "_target_signal_text": "Company expanded into a new geographic market",
            "_evidence_type": "",
            "claimed_source_urls": [
                "https://vanta.com/blog/uk-launch",
            ],
        },
        "contents": {
            "results": [
                {
                    "url": "https://vanta.com/blog/uk-launch",
                    "title": "Vanta is now in the UK",
                    "text": "We're thrilled to announce Vanta is now available to customers across the UK.",
                }
            ],
            "statuses": [],
        },
    },
    # ── Edge case rows ───────────────────────────────────────────────────
    {
        "id": "edge_empty_fields",
        "row": {
            "id": "sig-10",
            "company": "",
            "website": "",
            "company_linkedin": "",
            "contact_linkedin": "",
            "signal_type": "",
            "claim": "",
            "signal_date": None,
            "_target_signal_text": "",
            "claimed_source_urls": [],
        },
        "contents": {"results": [], "statuses": []},
    },
    {
        "id": "edge_unicode_company",
        "row": {
            "id": "sig-11",
            "company": "Société Générale",
            "website": "https://societegenerale.com",
            "company_linkedin": "https://www.linkedin.com/company/societegenerale/",
            "contact_linkedin": "",
            "signal_type": "hiring",
            "claim": "Société Générale ouvre des postes",
            "signal_date": "2026-05-01",
            "_target_signal_text": "Company is hiring",
            "_evidence_type": "HIRING",
            "claimed_source_urls": [
                "https://www.linkedin.com/jobs/societegenerale",
            ],
        },
        "contents": {
            "results": [
                {
                    "url": "https://www.linkedin.com/jobs/societegenerale",
                    "title": "Société Générale jobs",
                    "text": "Société Générale – Trader d'options structurées.",
                }
            ],
            "statuses": [],
        },
    },
    {
        "id": "edge_long_claim",
        "row": {
            "id": "sig-12",
            "company": "Notion",
            "website": "https://notion.so",
            "company_linkedin": "https://www.linkedin.com/company/notionhq/",
            "contact_linkedin": "",
            "signal_type": "social_post",
            "claim": "An extremely long miner claim — " + ("the founder posted again — " * 40),
            "signal_date": "2026-04-30",
            "_target_signal_text": "Founder or CEO is publicly posting on LinkedIn about company milestones, product launches, or fundraising in the past 60 days",
            "_evidence_type": "SOCIAL_POSTING",
            "claimed_source_urls": [
                "https://www.linkedin.com/posts/notion_news-activity-7400000000000000000-aaaa",
            ],
        },
        "contents": {
            "results": [
                {
                    "url": "https://www.linkedin.com/posts/notion_news-activity-7400000000000000000-aaaa",
                    "title": "Notion update",
                    "text": "Big product launch this week!",
                }
            ],
            "statuses": [],
        },
    },
    {
        "id": "edge_multi_source_urls",
        "row": {
            "id": "sig-13",
            "company": "Linear",
            "website": "https://linear.app",
            "company_linkedin": "https://www.linkedin.com/company/linear-/",
            "contact_linkedin": "",
            "signal_type": "funding",
            "claim": "Linear raised Series B",
            "signal_date": "2026-03-15",
            "_target_signal_text": "Company closed funding in past 12 months",
            "_evidence_type": "FUNDING",
            "claimed_source_urls": [
                "https://techcrunch.com/2026/03/15/linear-series-b/",
                "https://linear.app/blog/series-b",
                "https://www.linkedin.com/posts/linear-_news-activity-7300000000000000000",
            ],
        },
        "contents": {
            "results": [
                {
                    "url": "https://techcrunch.com/2026/03/15/linear-series-b/",
                    "title": "Linear raises $35M",
                    "text": "Linear closed a $35M Series B led by Accel.",
                },
                {
                    "url": "https://linear.app/blog/series-b",
                    "title": "Our Series B",
                    "text": "We are thrilled to share that Linear has closed our Series B.",
                },
            ],
            "statuses": [],
        },
    },
    {
        "id": "edge_no_results_with_statuses",
        "row": {
            "id": "sig-14",
            "company": "Glean",
            "website": "https://glean.com",
            "company_linkedin": "https://www.linkedin.com/company/gleanwork/",
            "contact_linkedin": "",
            "signal_type": "hiring",
            "claim": "Glean is hiring",
            "signal_date": "2026-05-01",
            "_target_signal_text": "Company has open AE positions in past 60 days",
            "_evidence_type": "HIRING",
            "claimed_source_urls": [
                "https://www.linkedin.com/jobs/view/4099999",
            ],
        },
        "contents": {
            "results": [],
            "statuses": [
                {
                    "url": "https://www.linkedin.com/jobs/view/4099999",
                    "source": "scrapingdog_linkedin_job",
                    "stage": "http_404",
                    "error": "job not found",
                }
            ],
        },
    },
    {
        "id": "edge_no_results_no_statuses",
        "row": {
            "id": "sig-15",
            "company": "Retool",
            "website": "https://retool.com",
            "company_linkedin": "https://www.linkedin.com/company/retool/",
            "contact_linkedin": "",
            "signal_type": "unknown",
            "claim": "Retool announcement",
            "signal_date": "2026-04-01",
            "_target_signal_text": "Company launched something",
            "claimed_source_urls": ["https://retool.com/news"],
        },
        "contents": {},
    },
    {
        "id": "edge_url_with_query_and_anchor",
        "row": {
            "id": "sig-16",
            "company": "Webflow",
            "website": "https://webflow.com",
            "company_linkedin": "https://www.linkedin.com/company/webflow-inc-/",
            "contact_linkedin": "",
            "signal_type": "social_post",
            "claim": "Webflow CEO posted",
            "signal_date": "2026-04-15",
            "_target_signal_text": "Founder or CEO is publicly posting about company milestones",
            "_evidence_type": "SOCIAL_POSTING",
            "claimed_source_urls": [
                "https://www.linkedin.com/posts/vlad-magdalin_news-activity-7440000000000000000-aaaa?trk=public_post",
            ],
        },
        "contents": {
            "results": [
                {
                    "url": "https://www.linkedin.com/posts/vlad-magdalin_news-activity-7440000000000000000-aaaa?trk=public_post",
                    "title": "Vlad Magdalin on LinkedIn",
                    "text": "We launched Webflow Apps today.",
                }
            ],
            "statuses": [],
        },
    },
    {
        "id": "edge_quotes_in_claim",
        "row": {
            "id": "sig-17",
            "company": "Quotebox",
            "website": "https://quotebox.com",
            "company_linkedin": "https://www.linkedin.com/company/quotebox/",
            "contact_linkedin": "",
            "signal_type": "hiring",
            "claim": "Quotebox said \"we are hiring\" on LinkedIn",
            "signal_date": "2026-05-01",
            "_target_signal_text": "Company is hiring",
            "_evidence_type": "HIRING",
            "claimed_source_urls": ["https://www.linkedin.com/jobs/quotebox"],
        },
        "contents": {
            "results": [
                {
                    "url": "https://www.linkedin.com/jobs/quotebox",
                    "title": "Quotebox jobs",
                    "text": "We're hiring engineers and designers.",
                }
            ],
            "statuses": [],
        },
    },
    {
        "id": "edge_braces_in_target",
        "row": {
            "id": "sig-18",
            "company": "Braced",
            "website": "https://braced.io",
            "company_linkedin": "https://www.linkedin.com/company/braced/",
            "contact_linkedin": "",
            "signal_type": "hiring",
            "claim": "Open positions",
            "signal_date": "2026-05-01",
            "_target_signal_text": 'Company has open positions for {role} – curly braces should be preserved as literals',
            "_evidence_type": "HIRING",
            "claimed_source_urls": ["https://www.linkedin.com/jobs/braced"],
        },
        "contents": {
            "results": [
                {
                    "url": "https://www.linkedin.com/jobs/braced",
                    "title": "Braced jobs",
                    "text": "Open AE role.",
                }
            ],
            "statuses": [],
        },
    },
    {
        "id": "edge_long_text_content",
        "row": {
            "id": "sig-19",
            "company": "Cohere",
            "website": "https://cohere.com",
            "company_linkedin": "https://www.linkedin.com/company/cohere-ai/",
            "contact_linkedin": "",
            "signal_type": "funding",
            "claim": "Cohere closed funding",
            "signal_date": "2026-04-30",
            "_target_signal_text": "Company closed funding in past 12 months",
            "_evidence_type": "FUNDING",
            "claimed_source_urls": ["https://cohere.com/blog/funding"],
        },
        "contents": {
            "results": [
                {
                    "url": "https://cohere.com/blog/funding",
                    "title": "Funding update",
                    # Long text — but well within MAX_SCRAPED_CHARS (12k).
                    "text": ("Cohere closed our latest funding round. " * 200),
                }
            ],
            "statuses": [],
        },
    },
    {
        "id": "edge_text_exceeds_max",
        "row": {
            "id": "sig-20",
            "company": "Bigtext",
            "website": "https://bigtext.com",
            "company_linkedin": "https://www.linkedin.com/company/bigtext/",
            "contact_linkedin": "",
            "signal_type": "social_post",
            "claim": "Bigtext post",
            "signal_date": "2026-04-30",
            "_target_signal_text": "Founder or CEO is publicly posting on LinkedIn about company milestones in past 60 days",
            "_evidence_type": "SOCIAL_POSTING",
            "claimed_source_urls": ["https://www.linkedin.com/posts/bigtext_news-activity-1234567890"],
        },
        "contents": {
            "results": [
                {
                    "url": "https://www.linkedin.com/posts/bigtext_news-activity-1234567890",
                    "title": "Bigtext on LinkedIn",
                    # This text deliberately exceeds MAX_SCRAPED_CHARS (12000) to confirm truncation.
                    "text": "A" * 25000,
                }
            ],
            "statuses": [],
        },
    },
    {
        "id": "edge_custom_source_name",
        "row": {
            "id": "sig-21",
            "company": "Wayback",
            "website": "https://wayback.com",
            "company_linkedin": "https://www.linkedin.com/company/wayback/",
            "contact_linkedin": "",
            "signal_type": "social_post",
            "claim": "Wayback post",
            "signal_date": "2026-04-15",
            "_target_signal_text": "Founder or CEO is publicly posting on LinkedIn about company milestones in past 60 days",
            "_evidence_type": "SOCIAL_POSTING",
            "claimed_source_urls": ["https://www.linkedin.com/posts/wayback_news-activity-1111111111"],
        },
        "contents": {
            "results": [
                {
                    "url": "https://www.linkedin.com/posts/wayback_news-activity-1111111111",
                    "title": "Wayback on LinkedIn",
                    "text": "Big news from us today.",
                }
            ],
            "statuses": [],
        },
        # Note: this case uses a custom source_name in the final-judge call.
        "source_name": "Wayback Snapshot",
    },
    {
        "id": "edge_null_signal_date",
        "row": {
            "id": "sig-22",
            "company": "Nullco",
            "website": "https://nullco.com",
            "company_linkedin": "https://www.linkedin.com/company/nullco/",
            "contact_linkedin": "",
            "signal_type": "social_post",
            "claim": "Null date post",
            "signal_date": None,
            "_target_signal_text": "Founder or CEO is publicly posting on LinkedIn about company milestones",
            "_evidence_type": "SOCIAL_POSTING",
            "claimed_source_urls": ["https://www.linkedin.com/posts/nullco_news-activity-2222222222"],
        },
        "contents": {
            "results": [
                {
                    "url": "https://www.linkedin.com/posts/nullco_news-activity-2222222222",
                    "title": "Nullco on LinkedIn",
                    "text": "Generic update.",
                }
            ],
            "statuses": [],
        },
    },
    {
        "id": "edge_missing_id_field",
        # The _visible_signal helper has a fallback ``str(row.get("id") or "signal-1")``;
        # this row deliberately omits ``id`` to exercise that branch.
        "row": {
            "company": "Noid",
            "website": "https://noid.com",
            "company_linkedin": "https://www.linkedin.com/company/noid/",
            "contact_linkedin": "",
            "signal_type": "unknown",
            "claim": "no id field",
            "signal_date": "2026-05-01",
            "_target_signal_text": "Company did something",
            "claimed_source_urls": ["https://noid.com/news"],
        },
        "contents": {
            "results": [
                {
                    "url": "https://noid.com/news",
                    "title": "News",
                    "text": "A news article.",
                }
            ],
            "statuses": [],
        },
    },
    {
        "id": "edge_status_only_no_results",
        "row": {
            "id": "sig-24",
            "company": "OnlyStatus",
            "website": "https://onlystatus.com",
            "company_linkedin": "https://www.linkedin.com/company/onlystatus/",
            "contact_linkedin": "",
            "signal_type": "social_post",
            "claim": "Status-only post",
            "signal_date": "2026-04-30",
            "_target_signal_text": "Founder or CEO is publicly posting on LinkedIn about company milestones in past 60 days",
            "_evidence_type": "SOCIAL_POSTING",
            "claimed_source_urls": ["https://www.linkedin.com/posts/onlystatus_news-activity-3333333333"],
        },
        "contents": {
            "results": [],
            "statuses": [
                {
                    "url": "https://www.linkedin.com/posts/onlystatus_news-activity-3333333333",
                    "source": "scrapingdog_linkedin_post_failed",
                    "linkedin_post_stage": "linkedin_post_http_400",
                    "linkedin_post_error": "something went wrong",
                }
            ],
        },
    },
    {
        "id": "edge_unicode_text_content",
        "row": {
            "id": "sig-25",
            "company": "Globe",
            "website": "https://globe.com",
            "company_linkedin": "https://www.linkedin.com/company/globe/",
            "contact_linkedin": "",
            "signal_type": "funding",
            "claim": "Globe news",
            "signal_date": "2026-04-15",
            "_target_signal_text": "Company expanded globally in past year",
            "_evidence_type": "FUNDING",
            "claimed_source_urls": ["https://globe.com/news"],
        },
        "contents": {
            "results": [
                {
                    "url": "https://globe.com/news",
                    "title": "Globe expands",
                    "text": "Globe — нiн hao 你好 こんにちは — opens new offices.  Quotes with “smart” quotes and em-dashes — preserved.",
                }
            ],
            "statuses": [],
        },
    },
]


def main() -> None:
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tests", "snapshots", "prompts",
    )
    os.makedirs(out_dir, exist_ok=True)

    # Also save the raw test fixtures so the test file can reconstruct
    # the exact (row, contents, source_name) triples it needs to feed
    # to the post-refactor dispatcher.
    fixtures_path = os.path.join(out_dir, "_fixtures.json")
    with open(fixtures_path, "w", encoding="utf-8") as f:
        # Keep a stable JSON snapshot.  We exclude any non-JSON-able fields.
        json.dump(CASES, f, ensure_ascii=False, indent=2)

    for case in CASES:
        cid = case["id"]
        row = case["row"]
        contents = case["contents"]
        source_name = case.get("source_name", "SD/Exa Contents")

        verification = _build_verification_prompt(row)
        final = _build_final_judge_prompt(row, contents, source_name)

        with open(os.path.join(out_dir, f"{cid}_verification.txt"), "w", encoding="utf-8") as f:
            f.write(verification)
        with open(os.path.join(out_dir, f"{cid}_final_judge.txt"), "w", encoding="utf-8") as f:
            f.write(final)

        print(
            f"  {cid:35s}  verification={len(verification):6d}b  final={len(final):6d}b"
        )

    print()
    print(f"Wrote fixtures + {len(CASES) * 2} snapshot files to: {out_dir}")


if __name__ == "__main__":
    main()
