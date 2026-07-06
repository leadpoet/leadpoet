from qualification.validator.local_proxy import calculate_scrapingdog_credits


def test_scrapingdog_private_model_endpoint_credit_mapping():
    cases = [
        ("google/ai_mode", {}, (10, "google_ai_mode")),
        ("profile", {"type": "company"}, (50, "profile")),
        ("profile", {"type": "profile", "premium": "true"}, (100, "profile (premium)")),
        ("profile/post", {}, (5, "profile_post")),
        ("youtube/transcripts", {}, (1, "youtube_transcripts")),
        ("tiktok/profile", {}, (5, "tiktok_profile")),
        ("linkedinjobs", {}, (5, "linkedinjobs")),
    ]

    for endpoint, params, expected in cases:
        assert calculate_scrapingdog_credits(endpoint, params) == expected


def test_scrapingdog_existing_endpoint_credit_mapping_is_preserved():
    assert calculate_scrapingdog_credits("scrape", {"dynamic": "true"}) == (
        5,
        "scrape (JS rendering)",
    )
    assert calculate_scrapingdog_credits("scrape", {"premium": "true"}) == (
        10,
        "scrape (premium)",
    )
    assert calculate_scrapingdog_credits("x/post", {}) == (5, "x")
    assert calculate_scrapingdog_credits("youtube/video", {}) == (5, "youtube")
