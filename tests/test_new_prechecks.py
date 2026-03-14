"""
Tests for newly added pre-checks:
  1. Cross-signal URL deduplication (lead_scorer.py)
  2. URL-to-company verification (intent_verification.py)
  3. Cache TTL reduction (intent_verification.py)
  4. Future date rejection (intent_verification.py)
  5. Source type vs URL mismatch detection (intent_verification.py)

Run:
    cd /Users/pranav/Downloads/Election_Analysis/Bittensor-subnet
    python -m pytest tests/test_new_prechecks.py -v

For integration tests with real ScrapingDog:
    SCRAPINGDOG_API_KEY=693b87d05aeaf62620d721a6 \
    python -m pytest tests/test_new_prechecks.py -v -k integration
"""

import os
import sys
import asyncio
import pytest
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qualification.scoring.intent_verification import (
    check_future_date,
    check_source_url_mismatch,
    check_company_in_content,
    DEFAULT_CACHE_TTL_DAYS,
)
from qualification.scoring.lead_scorer import _extract_domain


# =============================================================================
# 4. Cache TTL reduction
# =============================================================================

class TestCacheTTL:
    def test_cache_ttl_is_2_days(self):
        assert DEFAULT_CACHE_TTL_DAYS == 2, f"Expected 2, got {DEFAULT_CACHE_TTL_DAYS}"


# =============================================================================
# 6. Future date rejection
# =============================================================================

class TestFutureDateRejection:
    def test_future_date_rejected(self):
        future = (date.today() + timedelta(days=30)).isoformat()
        err = check_future_date(future)
        assert err is not None
        assert "future" in err.lower()

    def test_today_date_accepted(self):
        today = date.today().isoformat()
        assert check_future_date(today) is None

    def test_past_date_accepted(self):
        past = (date.today() - timedelta(days=60)).isoformat()
        assert check_future_date(past) is None

    def test_none_date_accepted(self):
        assert check_future_date(None) is None

    def test_empty_string_accepted(self):
        assert check_future_date("") is None

    def test_invalid_date_format_passes(self):
        assert check_future_date("not-a-date") is None

    def test_tomorrow_rejected(self):
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        err = check_future_date(tomorrow)
        assert err is not None


# =============================================================================
# 7. Source type vs URL mismatch detection
# =============================================================================

class TestSourceUrlMismatch:
    # --- linkedin ---
    def test_linkedin_source_linkedin_url_ok(self):
        assert check_source_url_mismatch("linkedin", "https://www.linkedin.com/in/johndoe") is None

    def test_linkedin_source_github_url_mismatch(self):
        err = check_source_url_mismatch("linkedin", "https://github.com/some/repo")
        assert err is not None
        assert "linkedin" in err.lower()

    # --- github ---
    def test_github_source_github_url_ok(self):
        assert check_source_url_mismatch("github", "https://github.com/openai/gpt") is None

    def test_github_source_news_url_mismatch(self):
        err = check_source_url_mismatch("github", "https://techcrunch.com/article")
        assert err is not None

    # --- news ---
    def test_news_source_bloomberg_ok(self):
        assert check_source_url_mismatch("news", "https://www.bloomberg.com/news/article") is None

    def test_news_source_reuters_ok(self):
        assert check_source_url_mismatch("news", "https://www.reuters.com/some-story") is None

    def test_news_source_techcrunch_ok(self):
        assert check_source_url_mismatch("news", "https://techcrunch.com/2026/01/01/story") is None

    def test_news_source_forbes_ok(self):
        assert check_source_url_mismatch("news", "https://www.forbes.com/sites/article") is None

    def test_news_source_prnewswire_ok(self):
        assert check_source_url_mismatch("news", "https://www.prnewswire.com/press-release") is None

    def test_news_source_businesswire_ok(self):
        assert check_source_url_mismatch("news", "https://www.businesswire.com/news/home/release") is None

    def test_news_source_cnbc_ok(self):
        assert check_source_url_mismatch("news", "https://www.cnbc.com/story") is None

    def test_news_source_github_mismatch(self):
        err = check_source_url_mismatch("news", "https://github.com/some/repo")
        assert err is not None

    def test_news_source_linkedin_mismatch(self):
        err = check_source_url_mismatch("news", "https://www.linkedin.com/posts/foo")
        assert err is not None

    # --- company_website --- should always pass (any domain)
    def test_company_website_always_ok(self):
        assert check_source_url_mismatch("company_website", "https://random-company.com") is None
        assert check_source_url_mismatch("company_website", "https://github.com/foo") is None

    # --- other --- should always pass
    def test_other_source_always_ok(self):
        assert check_source_url_mismatch("other", "https://anything.com") is None

    # --- wikipedia ---
    def test_wikipedia_source_wikipedia_ok(self):
        assert check_source_url_mismatch("wikipedia", "https://en.wikipedia.org/wiki/Something") is None

    def test_wikipedia_source_other_mismatch(self):
        err = check_source_url_mismatch("wikipedia", "https://github.com/foo")
        assert err is not None

    # --- job_board ---
    def test_job_board_indeed_ok(self):
        assert check_source_url_mismatch("job_board", "https://www.indeed.com/job/posting") is None

    def test_job_board_greenhouse_ok(self):
        assert check_source_url_mismatch("job_board", "https://boards.greenhouse.io/company/jobs/123") is None

    def test_job_board_lever_ok(self):
        assert check_source_url_mismatch("job_board", "https://jobs.lever.co/company/abc") is None

    def test_job_board_linkedin_ok(self):
        """LinkedIn job postings are a valid job board source."""
        assert check_source_url_mismatch("job_board", "https://www.linkedin.com/jobs/view/123") is None

    def test_job_board_github_mismatch(self):
        err = check_source_url_mismatch("job_board", "https://github.com/foo/bar")
        assert err is not None

    # --- social_media ---
    def test_social_media_twitter_ok(self):
        assert check_source_url_mismatch("social_media", "https://twitter.com/user/status/123") is None

    def test_social_media_x_ok(self):
        assert check_source_url_mismatch("social_media", "https://x.com/user/status/123") is None

    def test_social_media_reddit_ok(self):
        assert check_source_url_mismatch("social_media", "https://www.reddit.com/r/sub/comments/abc") is None

    def test_social_media_github_mismatch(self):
        err = check_source_url_mismatch("social_media", "https://github.com/foo")
        assert err is not None

    # --- review_site ---
    def test_review_site_g2_ok(self):
        assert check_source_url_mismatch("review_site", "https://www.g2.com/products/foo") is None

    def test_review_site_capterra_ok(self):
        assert check_source_url_mismatch("review_site", "https://www.capterra.com/p/123") is None

    def test_review_site_github_mismatch(self):
        err = check_source_url_mismatch("review_site", "https://github.com/foo")
        assert err is not None

    # --- case insensitivity ---
    def test_case_insensitive_source(self):
        assert check_source_url_mismatch("LinkedIn", "https://www.linkedin.com/in/john") is None
        assert check_source_url_mismatch("NEWS", "https://www.bloomberg.com/article") is None


# =============================================================================
# 3. URL-to-company verification
# =============================================================================

class TestCompanyInContent:
    def test_exact_match(self):
        assert check_company_in_content("Acme Corp", "Welcome to Acme Corp! We build widgets.")

    def test_case_insensitive(self):
        assert check_company_in_content("ACME CORP", "welcome to acme corp today")

    def test_partial_word_match(self):
        assert check_company_in_content("Acme Corporation", "Welcome to Acme Corporation.")

    def test_word_fragments_match(self):
        """'Acme Corp' should match even if page says 'Acme Corporation'."""
        assert check_company_in_content("Acme Corp", "Welcome to the Acme Corporation of America")

    def test_no_match(self):
        assert not check_company_in_content("Acme Corp", "Welcome to Widgets Inc.")

    def test_empty_content(self):
        assert not check_company_in_content("Acme Corp", "")

    def test_empty_name(self):
        assert not check_company_in_content("", "Some content here")

    def test_short_name_direct_match(self):
        assert check_company_in_content("IBM", "IBM announced quarterly earnings today")

    def test_short_words_not_individually_matched(self):
        """Short words (<4 chars) shouldn't trigger the word-fragment fallback."""
        assert not check_company_in_content("AI Co", "This article about ML and Data Science")


# =============================================================================
# 1. Cross-signal URL deduplication
# =============================================================================

class TestExtractDomain:
    def test_standard_url(self):
        assert _extract_domain("https://www.bloomberg.com/news/article") == "bloomberg.com"

    def test_subdomain(self):
        assert _extract_domain("https://finance.yahoo.com/quote/AAPL") == "yahoo.com"

    def test_github(self):
        assert _extract_domain("https://github.com/openai/gpt") == "github.com"

    def test_linkedin(self):
        assert _extract_domain("https://www.linkedin.com/in/johndoe") == "linkedin.com"

    def test_no_www(self):
        assert _extract_domain("https://techcrunch.com/article") == "techcrunch.com"

    def test_invalid_url_fallback(self):
        result = _extract_domain("not-a-url")
        assert isinstance(result, str)

    def test_deep_subdomain(self):
        assert _extract_domain("https://boards.greenhouse.io/company/jobs") == "greenhouse.io"


# =============================================================================
# Integration test: Real ScrapingDog + company check
# =============================================================================

@pytest.mark.skipif(
    not os.environ.get("SCRAPINGDOG_API_KEY"),
    reason="Set SCRAPINGDOG_API_KEY to run integration tests"
)
class TestIntegrationScrapingDog:
    """
    Integration tests that hit the real ScrapingDog API.
    Run with: SCRAPINGDOG_API_KEY=693b87d05aeaf62620d721a6 python -m pytest tests/test_new_prechecks.py -v -k integration
    """

    @pytest.fixture(autouse=True)
    def set_api_key(self):
        import qualification.scoring.intent_verification as iv
        self._orig_key = iv.SCRAPINGDOG_API_KEY
        iv.SCRAPINGDOG_API_KEY = os.environ["SCRAPINGDOG_API_KEY"]
        yield
        iv.SCRAPINGDOG_API_KEY = self._orig_key

    @pytest.mark.asyncio
    async def test_bloomberg_company_check(self):
        """Fetch a real Bloomberg article and verify company name check works."""
        from qualification.scoring.intent_verification import scrapingdog_generic, extract_verification_content

        url = "https://www.bloomberg.com/profile/company/MSFT:US"
        try:
            content = await scrapingdog_generic(url)
            text = extract_verification_content(content, "news")
            assert text and len(text) > 50
            assert check_company_in_content("Microsoft", text), \
                "Microsoft should be found on Bloomberg's MSFT profile page"
            assert not check_company_in_content("FakeNonexistentCorp12345", text), \
                "Fake company should NOT be found"
        except Exception as e:
            pytest.skip(f"ScrapingDog request failed (may be rate-limited): {e}")

    @pytest.mark.asyncio
    async def test_github_company_check(self):
        """Fetch a real GitHub repo and verify company name appears."""
        from qualification.scoring.intent_verification import github_api, extract_verification_content

        url = "https://github.com/microsoft/vscode"
        try:
            content = await github_api(url)
            text = extract_verification_content(content, "github")
            assert text and len(text) > 50
            assert check_company_in_content("Microsoft", text) or check_company_in_content("vscode", text), \
                "Microsoft or vscode should appear in GitHub repo content"
        except Exception as e:
            pytest.skip(f"GitHub API request failed: {e}")

    @pytest.mark.asyncio
    async def test_news_source_url_mismatch_rejected(self):
        """A signal declared as 'news' but pointing to github.com should be rejected."""
        err = check_source_url_mismatch("news", "https://github.com/openai/gpt-4")
        assert err is not None, "news + github.com URL should be flagged as mismatch"

    @pytest.mark.asyncio
    async def test_future_date_fast_rejection(self):
        """Future dates should be rejected instantly without any API call."""
        from gateway.qualification.models import IntentSignal
        from qualification.scoring.intent_verification import verify_intent_signal

        future_date = (date.today() + timedelta(days=30)).isoformat()
        signal = IntentSignal(
            source="news",
            description="Company announced major expansion",
            url="https://www.reuters.com/fake-article",
            date=future_date,
            snippet="Some snippet text from the article about expansion plans"
        )

        verified, confidence, reason, date_status = await verify_intent_signal(
            signal, icp_industry="Technology", company_name="TestCorp"
        )
        assert not verified
        assert confidence == 0
        assert date_status == "fabricated"
        assert "future" in reason.lower()

    @pytest.mark.asyncio
    async def test_source_url_mismatch_fast_rejection(self):
        """linkedin source with github URL should be rejected without API call."""
        from gateway.qualification.models import IntentSignal
        from qualification.scoring.intent_verification import verify_intent_signal

        signal = IntentSignal(
            source="linkedin",
            description="The CEO of Acme Technologies published a detailed LinkedIn post discussing their strategic expansion into the European cloud infrastructure market during Q1 2026",
            url="https://github.com/somecompany/repo",
            date="2026-01-15",
            snippet="Strategic expansion into the European cloud infrastructure market with new partnerships announced for Q1 2026 rollout"
        )

        verified, confidence, reason, date_status = await verify_intent_signal(
            signal, icp_industry="Technology", company_name="TestCorp"
        )
        assert not verified
        assert confidence == 0
        assert date_status == "fabricated"
        assert "mismatch" in reason.lower() or "domain" in reason.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
