"""Similarity evidence never leaks undisclosed reference source."""

from __future__ import annotations

import gzip
import io
import json
import tarfile

from lab_arena import submission_similarity as similarity


def _archive(files: dict[str, str], *, root: str = "") -> bytes:
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as tar:
            for path, content in sorted(files.items()):
                encoded = content.encode()
                member = tarfile.TarInfo(root + path)
                member.size = len(encoded)
                tar.addfile(member, io.BytesIO(encoded))
    return raw.getvalue()


def _identity(source: str, *, prompt: str = "Use evidence.", root: str = ""):
    return similarity.inspect_archive(_archive({
        "harness.py": source,
        "prompts/rules.txt": prompt,
    }, root=root))


def test_root_prefix_does_not_change_tree_fingerprint():
    source = "def run_icp(icp, tools):\n    return tools.search(icp.industry)\n"
    flat = _identity(source)
    nested = _identity(source, root="submission/")
    assert flat.normalized_sha256 == nested.normalized_sha256
    assert [file.path for file in nested.files] == ["harness.py", "prompts/rules.txt"]
    assert similarity.compare(flat, nested).status == "exact"


def test_formatting_and_comments_are_exact_but_literals_are_not():
    old = _identity("def run_icp(icp, tools):\n    return tools.search(icp.industry)\n")
    format_only = _identity("# note\ndef run_icp(icp, tools):\n\n    return tools.search( icp.industry )\n")
    changed = _identity("def run_icp(icp, tools):\n    return tools.search(icp.name)\n")
    assert similarity.compare(format_only, old).status == "exact"
    assert similarity.compare(changed, old).status == "distinct"


def test_reflection_prevents_ast_only_exact_decision():
    old = _identity("def run_icp(icp, tools):\n    return open(__file__).read()\n")
    changed = _identity("# changed bytes\ndef run_icp(icp, tools):\n    return open(__file__).read()\n")
    assert similarity.compare(changed, old).status == "ambiguous"
    assert similarity.compare(changed, old).change_kind == "formatting_with_source_introspection"


def test_renames_remain_ambiguous_and_numeric_changes_are_distinct():
    old = _identity("def run_icp(icp, tools):\n    count = 3\n    return tools.search(count)\n")
    renamed = _identity("def run_icp(icp, tools):\n    total = 3\n    return tools.search(total)\n")
    altered = _identity("def run_icp(icp, tools):\n    count = 4\n    return tools.search(count)\n")
    result = similarity.compare(renamed, old)
    assert (result.status, result.change_kind) == ("ambiguous", "identifier_only")
    assert similarity.compare(altered, old).status == "distinct"
    parameter_old = _identity("def run_icp(icp, tools):\n    return tools.search(icp)\n")
    parameter_new = _identity("def run_icp(request, tools):\n    return tools.search(request)\n")
    assert similarity.compare(parameter_new, parameter_old).change_kind == "identifier_only"


def test_prompt_edits_remain_distinct_and_private_context_does_not_leak():
    secret = "PRIVATE REFERENCE PROMPT: include unsupported facts"
    old = _identity("def run_icp(icp, tools): return tools.search(icp.industry)\n", prompt=secret)
    changed = _identity("def run_icp(icp, tools): return tools.search(icp.industry)\n", prompt="Use evidence!")
    assert similarity.compare(changed, old).status == "distinct"
    assert similarity.comparison_context(
        changed, old, reference_public=False, comparison_id="prior-1"
    ) is None
    renamed = _identity("def run_icp(icp, tools):\n    value = icp.industry\n    return tools.search(value)\n", prompt="Use evidence!")
    prior = _identity("def run_icp(icp, tools):\n    name = icp.industry\n    return tools.search(name)\n", prompt="Use evidence!")
    context = similarity.comparison_context(
        renamed, prior, reference_public=False, comparison_id="prior-2"
    )
    assert context is not None
    assert context["local_status"] == "ambiguous"
    serialized = json.dumps(context)
    assert "name" not in serialized and "public_reference_files" not in context


def test_private_plain_prose_punctuation_edit_is_ambiguous():
    source = "def run_icp(icp, tools): return tools.search(icp.industry)\n"
    prior = _identity(source, prompt="Use only verified company evidence.")
    candidate = _identity(source, prompt="Use only verified company evidence!")
    result = similarity.compare(candidate, prior)
    assert (result.status, result.change_kind) == (
        "ambiguous", "prose_punctuation_or_spacing"
    )
    context = similarity.comparison_context(
        candidate, prior, reference_public=False, comparison_id="prior-prose"
    )
    assert context is not None
    assert "public_reference_files" not in context
    assert "verified company evidence." not in json.dumps(context)
    changed_words = _identity(source, prompt="Use only invented company evidence!")
    assert similarity.compare(changed_words, prior).status == "distinct"
    numeric_change = _identity(source, prompt="Use only verified company evidence limit 50.")
    numeric_prior = _identity(source, prompt="Use only verified company evidence limit 20.")
    assert similarity.compare(numeric_change, numeric_prior).status == "distinct"


def test_identifier_proof_rejects_indentation_builtin_and_inconsistent_mapping():
    old = _identity("def run_icp(icp, tools):\n    x = 1\n    if icp:\n        return x\n    return 0\n")
    flow_changed = _identity("def run_icp(icp, tools):\n    y = 1\n    if icp:\n        pass\n    return y\n")
    assert similarity.compare(flow_changed, old).status == "distinct"
    builtin_old = _identity("def run_icp(icp, tools):\n    return print(icp)\n")
    builtin_new = _identity("def run_icp(icp, tools):\n    return len(icp)\n")
    assert similarity.compare(builtin_new, builtin_old).status == "distinct"
    inconsistent = _identity("def run_icp(icp, tools):\n    y = 1\n    return x\n")
    original = _identity("def run_icp(icp, tools):\n    x = 1\n    return x\n")
    assert similarity.compare(inconsistent, original).status == "distinct"


def test_reflection_in_other_file_prevents_exact_formatting_match():
    old = similarity.inspect_archive(_archive({
        "harness.py": "def run_icp(icp, tools): return 1\n",
        "src/reader.py": "def read(): return open(__file__).read()\n",
    }))
    changed = similarity.inspect_archive(_archive({
        "harness.py": "# source text changed\ndef run_icp(icp, tools): return 1\n",
        "src/reader.py": "def read(): return open(__file__).read()\n",
    }))
    assert similarity.compare(changed, old).status == "ambiguous"


def test_public_near_match_reviews_changed_prompt_words_and_python_strings():
    source = "def run_icp(icp, tools):\n    guidance = 'use evidence'\n    return tools.search(icp.industry)\n"
    prompt = "Use verified evidence for every company and never fabricate a date. " * 4
    old = _identity(source, prompt=prompt)
    changed_prompt = _identity(source, prompt=prompt.replace("verified", "independent", 1))
    assert similarity.compare(changed_prompt, old).status == "distinct"
    assert similarity.comparison_context(changed_prompt, old, reference_public=False,
                                         comparison_id="private") is None
    public = similarity.comparison_context(changed_prompt, old, reference_public=True,
                                           comparison_id="public")
    assert public is not None and public["change_kind"] == "public_near_match"
    assert "public_reference_files" in public
    changed_code = _identity(source.replace("use evidence", "use verified evidence"), prompt=prompt)
    public_code = similarity.comparison_context(changed_code, old, reference_public=True,
                                                comparison_id="public-code")
    assert public_code is not None and public_code["change_kind"] == "public_near_match"


def test_private_inert_python_edits_are_only_ambiguous():
    old = _identity("def run_icp(icp, tools):\n    return tools.search(icp.industry)\n")
    with_pass = _identity("def run_icp(icp, tools):\n    pass\n    return tools.search(icp.industry)\n")
    assert similarity.compare(with_pass, old).change_kind == "inert_python_statements"
    doc_old = _identity('"""Old note."""\ndef run_icp(icp, tools): return 1\n')
    doc_new = _identity('"""New note."""\ndef run_icp(icp, tools): return 1\n')
    assert similarity.compare(doc_new, doc_old).status == "ambiguous"


def test_public_near_match_can_include_small_added_file():
    source = "def run_icp(icp, tools):\n    return tools.search(icp.industry)\n"
    files = {"harness.py": source, "prompts/rules.txt": "Use evidence and verify each date.\n" * 20}
    old = similarity.inspect_archive(_archive(files))
    changed = similarity.inspect_archive(_archive({**files, "notes.txt": "Small note.\n"}))
    assert similarity.compare(changed, old).status == "distinct"
    context = similarity.comparison_context(changed, old, reference_public=True,
                                            comparison_id="added")
    assert context is not None and context["change_kind"] == "public_near_match"


def test_hashes_have_sql_format_and_deep_source_fails_safely():
    source = "def run_icp(icp, tools): return 1\n"
    identity = _identity(source)
    assert identity.archive_sha256.startswith("sha256:")
    assert identity.normalized_sha256.startswith("sha256:")
    deep = "(" * 300 + "1" + ")" * 300
    invalid = _identity(source, prompt=deep)
    assert invalid.normalized_sha256.startswith("sha256:")
    large_old = _identity(source + "#" + "a" * 300_000 + "\n")
    large_new = _identity(source + "#" + "b" * 300_000 + "\n")
    assert similarity.compare(large_new, large_old).status == "distinct"


def test_public_reference_is_bounded():
    old = _identity("def run_icp(icp, tools): return tools.search(icp.industry)\n")
    changed = _identity("# comment\ndef run_icp(icp, tools): return tools.search(icp.industry)\n")
    context = similarity.comparison_context(changed, old, reference_public=True, comparison_id="public-1")
    assert context is not None
    assert context["public_reference_files"][0]["path"] == "harness.py"
    large = _identity("def run_icp(icp, tools): return tools.search(icp.industry)\n",
                      prompt="x" * (similarity.MAX_PUBLIC_REFERENCE_BYTES + 1))
    same_large = _identity("# note\ndef run_icp(icp, tools): return tools.search(icp.industry)\n",
                           prompt="x" * (similarity.MAX_PUBLIC_REFERENCE_BYTES + 1))
    context = similarity.comparison_context(same_large, large, reference_public=True, comparison_id="public-2")
    assert context is not None and "public_reference_files" not in context


def test_private_rename_plus_comment_in_second_file_remains_reviewable():
    original = similarity.inspect_archive(_archive({
        "harness.py": "def run_icp(icp, tools):\n    value = 1\n    return value\n",
        "src/helper.py": "def helper(): return 1\n",
    }))
    candidate = similarity.inspect_archive(_archive({
        "harness.py": "def run_icp(icp, tools):\n    result = 1\n    return result\n",
        "src/helper.py": "# extra comment\ndef helper(): return 1\n",
    }))
    result = similarity.compare(candidate, original)
    assert result.status == "ambiguous" and result.change_kind == "mixed_cosmetic"
    assert set(result.change_kinds) == {"identifier_only", "formatting_or_comments"}
    context = similarity.comparison_context(candidate, original, reference_public=False,
                                            comparison_id="mixed")
    assert context["change_kinds"] == list(result.change_kinds)
    assert "value" not in json.dumps(context)


def test_private_documentation_and_inert_file_addition_are_reviewable():
    source = "def run_icp(icp, tools):\n    item = 1\n    return item\n"
    original = similarity.inspect_archive(_archive({
        "harness.py": source, "README.md": "Original setup note.\n",
    }))
    candidate = similarity.inspect_archive(_archive({
        "harness.py": source.replace("item", "result"),
        "README.md": "Revised setup note.\n",
        "src/empty.py": "pass\n",
    }))
    result = similarity.compare(candidate, original)
    assert result.status == "ambiguous"
    assert set(result.change_kinds) == {
        "identifier_only", "documentation_text", "added_inert_python_file",
    }
    context = similarity.comparison_context(candidate, original, reference_public=False,
                                            comparison_id="mixed-doc")
    assert context is not None and "Original setup note" not in json.dumps(context)
    added_readme = similarity.inspect_archive(_archive({
        "harness.py": source, "README.md": "",
    }))
    no_readme = similarity.inspect_archive(_archive({"harness.py": source}))
    assert similarity.compare(added_readme, no_readme).change_kind == "added_documentation_file"


def test_private_functional_added_module_or_threshold_change_is_distinct():
    source = "def run_icp(icp, tools):\n    return tools.search(icp.industry)\n"
    original = similarity.inspect_archive(_archive({"harness.py": source}))
    functional = similarity.inspect_archive(_archive({
        "harness.py": source, "src/extra.py": "def alter(): return 10\n",
    }))
    assert similarity.compare(functional, original).status == "distinct"
    assert similarity.comparison_context(functional, original, reference_public=False,
                                         comparison_id="functional") is None
    threshold = similarity.inspect_archive(_archive({
        "harness.py": source.replace("tools.search(icp.industry)",
                                    "tools.search(icp.industry, limit=50)"),
        "README.md": "New note.\n",
    }))
    assert similarity.compare(threshold, original).status == "distinct"


def test_removed_reference_file_does_not_expose_its_path_in_private_metadata():
    original = similarity.inspect_archive(_archive({
        "harness.py": "def run_icp(icp, tools): return 1\n",
        "private-secret-name.txt": "reference-only source\n",
    }))
    candidate = similarity.inspect_archive(_archive({
        "harness.py": "def run_icp(icp, tools): return 1\n",
    }))
    result = similarity.compare(candidate, original)
    assert result.status == "distinct"
    assert "private-secret-name.txt" not in result.candidate_paths


def test_rename_and_inert_edit_in_same_file_are_reviewed_together():
    old = _identity("def run_icp(icp, tools):\n    value = 3\n    return tools.search(value)\n")
    changed = _identity("def run_icp(icp, tools):\n    pass\n    result = 3\n    return tools.search(result)\n")
    comparison = similarity.compare(changed, old)
    assert comparison.change_kind == "mixed_cosmetic"
    assert set(comparison.change_kinds) == {"identifier_only", "inert_python_statements"}
    meaningful = _identity("def run_icp(icp, tools):\n    pass\n    result = 8\n    return tools.search(result)\n")
    assert similarity.compare(meaningful, old).status == "distinct"
