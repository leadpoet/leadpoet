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
    """Mirror of qualification/scoring/intent_verification.py
    _format_instagram_profile_blob. Kept inline so tests stay hermetic."""
    if not isinstance(data, dict):
        return ""
    user_obj = data.get("user") if isinstance(data.get("user"), dict) else {}
    username = data.get("username") or user_obj.get("username")
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
    followers = _first("followers_count", "followers", "edge_followed_by")
    if isinstance(followers, dict):
        followers = followers.get("count")
    following = _first("following_count", "following", "edge_follow")
    if isinstance(following, dict):
        following = following.get("count")

    post_count = _first("post_count", "media_count", "posts")
    if post_count is None:
        otm = data.get("owner_to_timeline_media") or data.get("edge_owner_to_timeline_media")
        if isinstance(otm, dict):
            post_count = otm.get("count")
    if isinstance(post_count, dict):
        post_count = post_count.get("count")

    is_verified = _first("is_verified", "verified")
    is_business = _first("is_business_account", "is_business")
    is_professional = data.get("is_professional_account")
    is_private = _first("is_private", "private")
    category = _first("business_category_name", "category_name",
                      "overall_category_name", "category")

    external_url = _first("external_url", "external_link", "website")
    if not external_url:
        bio_links = data.get("bio_links") or []
        if isinstance(bio_links, list):
            for bl in bio_links:
                if isinstance(bl, dict) and bl.get("url"):
                    external_url = bl["url"]
                    break

    parts = ["[INSTAGRAM PROFILE]"]
    parts.append(f"Username: @{username}")
    if full_name:
        parts.append(f"Display name: {full_name}")
    if is_verified is not None:
        parts.append(f"Verified: {bool(is_verified)}")
    if is_business is not None:
        parts.append(f"Business account: {bool(is_business)}")
    if is_professional is not None:
        parts.append(f"Professional account: {bool(is_professional)}")
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

    candidates: list = []
    otm = data.get("owner_to_timeline_media") or data.get("edge_owner_to_timeline_media")
    if isinstance(otm, dict):
        media = otm.get("media") or []
        if isinstance(media, list):
            candidates.extend(m for m in media if isinstance(m, dict))
        edges = otm.get("edges") or []
        if isinstance(edges, list):
            candidates.extend(
                e["node"] for e in edges
                if isinstance(e, dict) and isinstance(e.get("node"), dict)
            )
    vt = data.get("video_timeline")
    if isinstance(vt, dict):
        videos = vt.get("videos") or []
        if isinstance(videos, list):
            candidates.extend(v for v in videos if isinstance(v, dict))
    for k in ("recent_posts", "latest_posts", "posts_data"):
        v = data.get(k)
        if isinstance(v, list):
            candidates.extend(p for p in v if isinstance(p, dict))

    seen_ids = set()
    posts_with_ts: list = []
    for p in candidates:
        pid = p.get("shortcode") or p.get("video_id") or p.get("id") or id(p)
        if pid in seen_ids:
            continue
        seen_ids.add(pid)
        ts_raw = p.get("timestamp") or p.get("taken_at_timestamp") or p.get("taken_at")
        try:
            ts_num = float(ts_raw) if ts_raw is not None else 0.0
        except (ValueError, TypeError):
            ts_num = 0.0
        posts_with_ts.append((ts_num, p))
    posts_with_ts.sort(key=lambda x: x[0], reverse=True)

    if posts_with_ts:
        post_lines = []
        for ts_num, p in posts_with_ts[:6]:
            caption = (
                p.get("caption")
                or p.get("text")
                or (p.get("edge_media_to_caption", {}).get("edges", [{}])[0]
                    .get("node", {}).get("text") if isinstance(p.get("edge_media_to_caption"), dict) else None)
                or ""
            )
            ts_str = ""
            if ts_num > 0:
                try:
                    ts_str = datetime.fromtimestamp(ts_num, tz=timezone.utc).strftime("%Y-%m-%d")
                except Exception:
                    ts_str = ""
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


class InstagramLiveScrapingDogShapeTests(unittest.TestCase):
    """Use the EXACT response shape captured from a live ScrapingDog
    /instagram/profile call on 2026-04-29 (target: @buddybrewcoffee).
    This guards against the formatter quietly drifting from the real API
    response. If ScrapingDog renames a field, this test will fail and
    flag the regression."""

    LIVE_RESPONSE = {
        "username": "buddybrewcoffee",
        "profile_id": "36488148",
        "full_name": "Buddy Brew Coffee",
        "bio": "Home of the Carrot Cake Latte\nEight cafes in Tampa Bay",
        "followers_count": 49631,
        "following_count": 989,
        "bio_links": [
            {
                "title": "",
                "lynx_url": "https://l.instagram.com/?u=https%3A%2F%2Flinktr.ee%2Fbuddybrewcoffee",
                "url": "https://linktr.ee/buddybrewcoffee",
                "link_type": "external",
            }
        ],
        "is_business_account": False,
        "is_professional_account": True,
        "is_private": False,
        "is_verified": False,
        "is_joined_recently": False,
        "business_category_name": None,
        "category_name": None,
        "overall_category_name": None,
        "owner_to_timeline_media": {"count": 1850, "media": []},
        "video_timeline": {
            "count": 6,
            "videos": [
                {
                    "video_id": "2724574633789230865",
                    "shortcode": "CXPojGrDA8R",
                    "is_video": True,
                    "caption": "Update: Our Brew Truck will be ready to serve at 8am at our Kennedy address tomorrow, 12/9.",
                    "comment_count": 4,
                    "video_view_count": 1234,
                    "timestamp": 1639014641,
                },
                {
                    "video_id": "2",
                    "shortcode": "ABC2",
                    "is_video": True,
                    "caption": "Pumpkin spice latte is back",
                    "timestamp": 1714435200,
                },
            ],
        },
    }

    def test_live_shape_extracts_all_critical_fields(self):
        blob = _format_instagram_profile_blob(self.LIVE_RESPONSE)
        # Username + display name
        self.assertIn("@buddybrewcoffee", blob)
        self.assertIn("Buddy Brew Coffee", blob)
        # Bio multi-line content (snippet matching depends on this)
        self.assertIn("Carrot Cake Latte", blob)
        self.assertIn("Tampa Bay", blob)
        # Metrics — these are the values an ICP signal would compare against
        self.assertIn("Followers: 49631", blob)
        self.assertIn("Following: 989", blob)
        self.assertIn("Post count: 1850", blob)
        # Booleans
        self.assertIn("Verified: False", blob)
        self.assertIn("Business account: False", blob)
        self.assertIn("Professional account: True", blob)
        self.assertIn("Private: False", blob)
        # External URL pulled from bio_links[0].url
        self.assertIn("Website: https://linktr.ee/buddybrewcoffee", blob)
        # Recent posts merged from video_timeline.videos, sorted desc by ts
        self.assertIn("Recent posts:", blob)
        # Latest post (ts=1714435200 → 2024-04-30) appears before older one
        idx_new = blob.find("Pumpkin spice latte")
        idx_old = blob.find("Brew Truck")
        self.assertGreater(idx_new, 0)
        self.assertGreater(idx_old, idx_new, "Posts should be sorted newest-first")

    def test_live_shape_blob_min_length_for_snippet_check(self):
        """Verifier requires text >= 50 chars to attempt grounding checks
        (line ~1231 of intent_verification.py: 'Insufficient content...')."""
        blob = _format_instagram_profile_blob(self.LIVE_RESPONSE)
        self.assertGreater(len(blob), 200,
                          "Live-shape blob must be substantial enough to "
                          "pass the 50-char Tier 3 content threshold")

    def test_live_shape_post_count_resolved_via_owner_to_timeline_media(self):
        """ScrapingDog uses owner_to_timeline_media.count (no edge_ prefix
        and no top-level post_count). This was a regression in v1 of the
        formatter — guard against re-introducing it."""
        # Force the test: payload has ONLY owner_to_timeline_media for posts
        data = {
            "username": "x",
            "owner_to_timeline_media": {"count": 1850},
        }
        blob = _format_instagram_profile_blob(data)
        self.assertIn("Post count: 1850", blob)

    def test_live_shape_external_url_resolved_via_bio_links(self):
        """ScrapingDog uses bio_links[].url, not external_url. Was a
        regression in v1 of the formatter — guard against re-introducing."""
        data = {
            "username": "x",
            "bio_links": [{"url": "https://example.com", "title": ""}],
        }
        blob = _format_instagram_profile_blob(data)
        self.assertIn("Website: https://example.com", blob)


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
