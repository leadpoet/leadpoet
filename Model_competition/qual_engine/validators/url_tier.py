"""URL specificity tier — purely structural classification (no path-keyword knowledge).

Delegates to qual_engine.utils.ai_classifiers.classify_url_tier_structural,
which classifies by path depth, numeric IDs, slug shape, and date patterns —
not by hardcoded knowledge of which words mean "category page".
"""

from qual_engine.utils.ai_classifiers import classify_url_tier_structural as classify_url_tier

__all__ = ["classify_url_tier"]
