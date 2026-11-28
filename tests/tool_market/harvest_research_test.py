"""Provider contract for the job and post operations used by Lead Search."""
from urllib.parse import parse_qs, urlsplit

import pytest

from lab_arena.tool_market.catalog import CatalogError, operation
from lead_order_vault.tools import TOOLS


@pytest.mark.parametrize("name,path,params,wire_params", [
    ("search_jobs","job-search",{"search":"account executive","postedLimit":"week","sortBy":"relevance","page":2,"industryId":"10,11"},
     {"search":"account executive","postedLimit":"week","sortBy":"relevance","page":"2","industryId":"10,11"}),
    ("get_job","job",{"jobId":"123456"},{"jobId":"123456"}),
    ("search_posts","post-search",{"companyId":"123,987","postedLimit":"month","page":1},
     {"company":"123,987","postedLimit":"month","page":"1"}),
    ("post_reactions","post-reactions",{"post":"https://www.linkedin.com/posts/example","page":2},
     {"post":"https://www.linkedin.com/posts/example","page":"2"}),
    ("post_comments","post-comments",{"post":"https://www.linkedin.com/posts/example","page":1},
     {"post":"https://www.linkedin.com/posts/example","page":"1"}),
    ("profile_posts","profile-posts",{"profileId":"example","page":2,"paginationToken":"page2"},
     {"profile":"example","page":"2","paginationToken":"page2"}),
])
def test_platform_query_and_meter(name, path, params, wire_params):
    op = operation("harvestapi", name)
    url, _, _ = op.build(params)
    u = urlsplit(url)
    assert u.hostname == "api.harvestapi.io" and u.path == "/linkedin/" + path
    assert TOOLS["harvestapi"].route(op.method, u.path, u.query)
    assert parse_qs(u.query) == {k: [v] for k, v in wire_params.items()}
    answer = {"element": {"id":"123456"}} if name == "get_job" else {"elements":[]}
    assert op.units(200, dict(answer,status=200)) == 1  # successful empty search page still costs one call
    for bad in ({"status":200}, dict(answer,status=500), dict(answer,status=200,error="unavailable")):
        assert op.units(200,bad) is None


def test_job_page_after_provider_limit_refused_before_spend():
    with pytest.raises(CatalogError):
        operation("harvestapi","search_jobs").build({"search":"recruiter","page":41})
