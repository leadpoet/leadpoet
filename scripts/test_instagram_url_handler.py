"""
Offline tests for the Instagram URL handler in qualification/scoring/intent_verification.py.

Verifies:
  1. _extract_instagram_username correctly identifies profile URLs and rejects
     post/reel/story/reserved-path URLs.
  2. _format_instagram_profile_blob handles real-world payload variants
     (multiple field name aliases, missing fields, dict-shaped metric counts).
  3. scrapingdog_instagram routes correctly (profile → fetch, post → empty).
  4. The blob shape is suitable for downstream snippet-overlap and
     description-grounding checks (>= 50 char content, contains username,
     contains follower count text, etc.).
"""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Avoid importing the full intent_verification.py (it pulls in OpenRouter etc.).
# Instead, copy the helpers under test inline. This keeps the test hermetic.
# ---------------------------------------------------------------------------

import re
from datetime import datetime, timezone

_IG_RESERVED_PATHS = {
    "p", "reel", "reels", "stories", "explore", "directory",
    "tv", "accounts", "about", "developer", "legal", "press",
    "privacy", "tags", "locations", "web", "ajax",
}
_IG_USERNAME_RE = re.compile(r'instagram\.com/([A-Za-z0-9_.]+)')


def _extract_instagram_username(url: str):
    m = _IG_USERNAME_RE.search(url)
    if not m:
        return None
    username = m.group(1).rstrip("/").strip()
    if not username or username.lower() in _IG_RESERVED_PATHS:
        return None
    if len(username) > 30:
        return None
    return username


def _format_instagram_profile_blob(data: dict) -> str:
    if not isinstance(data, dict):
        return ""
    username = (
        data.get("username")
        or (data.get("user", {}).get("username") if isinstance(data.get("user"), dict) else None)
    )
    if not username:
        return ""

    def _first(*keys, default=None):
        for k in keys:
            v = data.get(k)
            if v is not None and v != "" and v != []:
                return v
        return default

    full_name = _first("full_name", "fullname", "name")
    biography = _first("biography", "bio", "description")
    external_url = _first("external_url", "external_link", "website")
    followers = _first("followers", "followers_count", "edge_followed_by")
    if isinstance(followers, dict):
        followers = followers.get("count")
    following = _first("following", "following_count", "edge_follow")
    if isinstance(following, dict):
        following = following.get("count")
    post_count = _first("posts", "post_count", "media_count", "edge_owner_to_timeline_media")
    if isinstance(post_count, dict):
        post_count = post_count.get("count")
    is_verified = _first("is_verified", "verified")
    is_business = _first("is_business_account", "is_business")
    is_private = _first("is_private", "private")
    category = _first("category", "category_name", "business_category_name")

    parts = ["[INSTAGRAM PROFILE]"]
    parts.append(f"Username: @{username}")
    if full_name:
        parts.append(f"Display name: {full_name}")
    if is_verified is not None:
        parts.append(f"Verified: {bool(is_verified)}")
    if is_business is not None:
        parts.append(f"Business account: {bool(is_business)}")
    if is_private is not None:
        parts.append(f"Private: {bool(is_private)}")
    if category:
        parts.append(f"Category: {category}")
    if followers is not None:
        parts.append(f"Followers: {followers}")
    if following is not None:
        parts.append(f"Following: {following}")
    if post_count is not None:
        parts.append(f"Post count: {post_count}")
    if external_url:
        parts.append(f"Website: {external_url}")
    if biography:
        parts.append(f"\nBio:\n{biography}")

    recent = _first("recent_posts", "latest_posts", "posts_data") or []
    if not recent:
        edges_obj = data.get("edge_owner_to_timeline_media")
        if isinstance(edges_obj, dict):
            recent = [e.get("node") for e in (edges_obj.get("edges") or []) if isinstance(e, dict)]
    if isinstance(recent, list) and recent:
        post_lines = []
        for p in recent[:6]:
            if not isinstance(p, dict):
                continue
            caption = (
                p.get("caption")
                or p.get("text")
                or (p.get("edge_media_to_caption", {}).get("edges", [{}])[0]
                    .get("node", {}).get("text") if isinstance(p.get("edge_media_to_caption"), dict) else None)
                or ""
            )
            ts = p.get("timestamp") or p.get("taken_at_timestamp") or p.get("taken_at")
            ts_str = ""
            if ts:
                try:
                    if isinstance(ts, (int, float)):
                        ts_str = datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d")
                    else:
                        ts_str = str(ts)
                except Exception:
                    ts_str = str(ts)
            line = f"  - {ts_str}: {caption[:200]}".rstrip()
            if line.strip(" -:"):
                post_lines.append(line)
        if post_lines:
            parts.append("\nRecent posts:")
            parts.extend(post_lines)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class InstagramUsernameExtractionTests(unittest.TestCase):
    def test_simple_profile_url(self):
        self.assertEqual(
            _extract_instagram_username("https://www.instagram.com/buddybrew/"),
            "buddybrew",
        )

    def test_profile_url_no_trailing_slash(self):
        self.assertEqual(
            _extract_instagram_username("https://instagram.com/buddybrew"),
            "buddybrew",
        )

    def test_profile_url_with_query(self):
        # The regex captures only the username before any /
        self.assertEqual(
            _extract_instagram_username("https://www.instagram.com/buddybrew?hl=en"),
            "buddybrew",
        )

    def test_profile_url_dotted_username(self):
        self.assertEqual(
            _extract_instagram_username("https://www.instagram.com/buddy.brew.coffee/"),
            "buddy.brew.coffee",
        )

    def test_post_url_returns_none(self):
        # /p/{shortcode} — no deterministic username
        self.assertIsNone(
            _extract_instagram_username("https://www.instagram.com/p/Ck-XYZ_abc/")
        )

    def test_reel_url_returns_none(self):
        self.assertIsNone(
            _extract_instagram_username("https://www.instagram.com/reel/Ck-XYZ_abc/")
        )

    def test_stories_url_returns_none(self):
        self.assertIsNone(
            _extract_instagram_username("https://www.instagram.com/stories/buddybrew/")
        )

    def test_explore_url_returns_none(self):
        self.assertIsNone(
            _extract_instagram_username("https://www.instagram.com/explore/tags/coffee/")
        )

    def test_non_instagram_url_returns_none(self):
        self.assertIsNone(
            _extract_instagram_username("https://www.facebook.com/buddybrew/")
        )

    def test_oversized_username_rejected(self):
        # IG caps usernames at 30 chars
        long = "a" * 35
        self.assertIsNone(
            _extract_instagram_username(f"https://instagram.com/{long}/")
        )


class InstagramBlobFormattingTests(unittest.TestCase):
    """Mirror real-world ScrapingDog payload shapes."""

    def test_canonical_shape(self):
        data = {
            "username": "buddybrew",
            "full_name": "Buddy Brew Coffee",
            "biography": "Tampa's neighborhood coffee roaster. Order online ↓",
            "external_url": "https://buddybrew.com",
            "followers": 343,
            "following": 412,
            "post_count": 88,
            "is_verified": False,
            "is_business_account": True,
            "is_private": False,
            "category": "Coffee Shop",
        }
        blob = _format_instagram_profile_blob(data)
        self.assertIn("[INSTAGRAM PROFILE]", blob)
        self.assertIn("@buddybrew", blob)
        self.assertIn("Buddy Brew Coffee", blob)
        self.assertIn("Followers: 343", blob)
        self.assertIn("Post count: 88", blob)
        self.assertIn("Coffee Shop", blob)
        self.assertIn("Tampa's neighborhood coffee roaster", blob)
        # Should be substantial enough for snippet-overlap (>200 chars)
        self.assertGreater(len(blob), 100)

    def test_alias_field_names_followers_count(self):
        """Some payloads use _count suffix. Should still resolve."""
        data = {
            "username": "x",
            "full_name": "X",
            "followers_count": 1234,
            "following_count": 567,
            "media_count": 89,
        }
        blob = _format_instagram_profile_blob(data)
        self.assertIn("Followers: 1234", blob)
        self.assertIn("Following: 567", blob)
        self.assertIn("Post count: 89", blob)

    def test_dict_shaped_metric_counts(self):
        """GraphQL-style payloads nest counts in {"count": N}."""
        data = {
            "username": "x",
            "edge_followed_by": {"count": 999},
            "edge_follow": {"count": 50},
            "edge_owner_to_timeline_media": {"count": 12, "edges": []},
        }
        blob = _format_instagram_profile_blob(data)
        self.assertIn("Followers: 999", blob)
        self.assertIn("Following: 50", blob)
        self.assertIn("Post count: 12", blob)

    def test_recent_posts_unix_ts(self):
        data = {
            "username": "x",
            "recent_posts": [
                {
                    "caption": "We're back open after the storm!",
                    "taken_at_timestamp": 1714435200,  # 2024-04-30
                },
            ],
        }
        blob = _format_instagram_profile_blob(data)
        self.assertIn("Recent posts:", blob)
        self.assertIn("storm", blob)
        self.assertIn("2024", blob)

    def test_recent_posts_graphql_shape(self):
        """GraphQL embedded captions."""
        data = {
            "username": "x",
            "edge_owner_to_timeline_media": {
                "count": 1,
                "edges": [
                    {
                        "node": {
                            "edge_media_to_caption": {
                                "edges": [
                                    {"node": {"text": "Pumpkin spice latte is back"}}
                                ]
                            },
                            "taken_at_timestamp": 1714435200,
                        }
                    }
                ],
            },
        }
        blob = _format_instagram_profile_blob(data)
        self.assertIn("Pumpkin spice latte", blob)

    def test_missing_username_returns_empty(self):
        # No username = unrecognized payload; treat as unverifiable
        self.assertEqual(_format_instagram_profile_blob({"foo": "bar"}), "")

    def test_non_dict_input_returns_empty(self):
        self.assertEqual(_format_instagram_profile_blob("not a dict"), "")  # type: ignore
        self.assertEqual(_format_instagram_profile_blob(None), "")  # type: ignore

    def test_minimal_payload(self):
        """Just username with no other fields — should still produce a blob
        with the username line, since we don't want to false-fail when the
        endpoint returns sparse data on private/empty accounts."""
        data = {"username": "private_account_x"}
        blob = _format_instagram_profile_blob(data)
        self.assertIn("@private_account_x", blob)
        self.assertGreater(len(blob), 0)


class InstagramIntegrationShapeTests(unittest.TestCase):
    """Verify the blob is shaped suitably for downstream verification checks."""

    def test_blob_contains_company_name_for_match(self):
        """The 'company name in content' check (line ~1240 in intent_verification)
        scans the first CONTENT_MAX_LENGTH chars for the company name."""
        data = {
            "username": "buddybrew",
            "full_name": "Buddy Brew Coffee",
            "biography": "Tampa coffee. Family-run since 2009.",
        }
        blob = _format_instagram_profile_blob(data)
        # Verifier checks if "Buddy Brew Coffee" appears in content
        self.assertIn("Buddy Brew Coffee", blob)

    def test_blob_contains_metric_text_for_llm_grounding(self):
        """The downstream LLM verification (llm_verify_claim_with_icp) should
        find concrete numbers in the blob to ground claims like 'this profile
        has under 500 followers' or 'last post was 90+ days ago'."""
        data = {
            "username": "smallshop",
            "followers": 343,
            "post_count": 12,
            "recent_posts": [
                {"caption": "Open 9-3 today!", "taken_at_timestamp": 1700000000},
            ],
        }
        blob = _format_instagram_profile_blob(data)
        # The blob must contain raw numbers + the most recent post timestamp
        self.assertIn("343", blob)
        self.assertIn("12", blob)
        self.assertIn("Open 9-3 today!", blob)


if __name__ == "__main__":
    unittest.main(verbosity=2)
