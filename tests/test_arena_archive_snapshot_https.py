"""Only source-bound Wayback snapshot locators gain HTTPS transport."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from lab_arena import shim
from qualification.scoring import intent_verification_three_stage as intent
from tests.reviewer_frozen_sources import frozen_source

ORIGINAL = "https://publisher.example/article?edition=one"
TIMESTAMP = "20261008185541"
PREFIX = "http://web.archive.org/web/" + TIMESTAMP + "/"
SNAPSHOT = PREFIX + ORIGINAL


def run_locator(monkeypatch, *, snapshot=SNAPSHOT, original=ORIGINAL, **fields):
    closest = {"available": True, "status": "200", "timestamp": TIMESTAMP, "url": snapshot, **fields}
    calls = []

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url):
            calls.append(url)
            if len(calls) == 1:
                return httpx.Response(200, json={"archived_snapshots": {"closest": closest}},
                                      request=httpx.Request("GET", url))
            # Discovery alone is not evidence that an archived page is usable.
            raise RuntimeError("synthetic snapshot not fetched")

    monkeypatch.setattr(intent, "httpx", SimpleNamespace(AsyncClient=Client))
    result = asyncio.run(intent._try_wayback(original))
    assert result["ok"] is False and result["stage"] == "wayback_exception"
    return calls[-1]


def test_matching_owned_http_snapshot_uses_https_without_changing_timestamp_path_or_query(monkeypatch):
    assert run_locator(monkeypatch) == "https" + SNAPSHOT[4:]


@pytest.mark.parametrize("snapshot,fields", [
    (SNAPSHOT.replace("web.archive.org", "foreign.example"), {}),
    (SNAPSHOT.replace("web.archive.org", "web.archive.org.foreign.example"), {}),
    (SNAPSHOT.replace("web.archive.org", "user@web.archive.org"), {}),
    (SNAPSHOT.replace("web.archive.org", "web.archive.org:80"), {}),
    (SNAPSHOT + "#fragment", {}), (SNAPSHOT + "#", {}),
    (SNAPSHOT.replace(TIMESTAMP, "2026100818554"), {}),
    (SNAPSHOT.replace(TIMESTAMP, "２０２６１００８１８５５４１"), {}),
    (SNAPSHOT.replace("https://publisher", "https%3A%2F%2Fpublisher"), {}),
    (SNAPSHOT.replace("edition=one", "edition=two"), {}),
    (PREFIX + "https://foreign.example/article?edition=one", {}),
    (PREFIX + "https://web.archive.org/web/" + TIMESTAMP + "/" + ORIGINAL, {}),
    (PREFIX + "https://127.0.0.1/private", {}),
    (SNAPSHOT, {"available": False}), (SNAPSHOT, {"available": "true"}),
    (SNAPSHOT, {"status": "404"}), (SNAPSHOT, {"status": 200}),
    (SNAPSHOT, {"timestamp": "20261008185542"}), (SNAPSHOT, {"timestamp": None}),
])
def test_unproved_locator_does_not_gain_https_or_bypass_http_shim(monkeypatch, snapshot, fields):
    assert run_locator(monkeypatch, snapshot=snapshot, **fields) == snapshot
    assert shim._trusted_page_fetch("GET", snapshot, b"") is None


def test_existing_https_snapshot_is_unchanged(monkeypatch):
    snapshot = "https" + SNAPSHOT[4:]
    assert run_locator(monkeypatch, snapshot=snapshot) == snapshot


@pytest.mark.parametrize("original", [
    "https://127.0.0.1/private", "https://169.254.169.254/private",
    "https://metadata.google.internal/private", "https://private.local/private",
    "https://8.8.8.8/public-ip", "https://[2001:4860:4860::8888]/public-ip",
])
def test_matching_private_or_ip_original_cannot_gain_archive_https_transport(monkeypatch, original):
    snapshot = PREFIX + original
    followed = run_locator(monkeypatch, snapshot=snapshot, original=original)
    assert followed != "https" + snapshot[4:]


def original_source():
    from scripts import arena_reviewer_archive_https_seam as seam
    reviewed = frozen_source(
        "archive_https", "qualification/scoring/intent_verification_three_stage.py",
    )
    assert reviewed.count(seam.REPLACEMENT.encode()) == 1
    return reviewed.replace(seam.REPLACEMENT.encode(), seam.ANCHOR.encode(), 1)


def test_frozen_overlay_changes_only_the_complete_reviewed_function():
    from scripts import arena_reviewer_archive_https_seam as seam
    original = original_source()
    repaired = seam.convert_archive_https_seam(original)
    assert repaired == frozen_source(
        "archive_https", "qualification/scoring/intent_verification_three_stage.py",
    )
    assert seam.convert_archive_https_seam(repaired) == repaired
    assert repaired.replace(seam.REPLACEMENT.encode(), seam.ANCHOR.encode(), 1) == original


def test_native_body_guard_does_not_expand_the_historical_overlay():
    from pathlib import Path
    from scripts import arena_reviewer_archive_https_seam as seam

    with pytest.raises(seam.ArchiveHttpsSeamError):
        seam.convert_archive_https_seam(Path(intent.__file__).read_bytes())


@pytest.mark.parametrize("fault", ["unknown", "weakened", "shadowed", "duplicate", "decorated", "decoy", "crlf"])
def test_frozen_overlay_rejects_unreviewed_or_ambiguous_code(fault):
    from scripts import arena_reviewer_archive_https_seam as seam
    source = original_source()
    if fault == "unknown":
        source = source.replace(b'"wayback_no_snapshot"', b'"changed"', 1)
    elif fault == "weakened":
        source = seam.convert_archive_https_seam(source).replace(
            b'                and snapshot_parts.netloc == "web.archive.org"\n', b'')
    elif fault == "shadowed":
        source += b'\n_try_wayback = lambda *args: None\n'
    elif fault == "duplicate":
        source += b'\nasync def _try_wayback(url):\n    return None\n'
    elif fault == "decorated":
        source = source.replace(b"async def _try_wayback", b"@unreviewed\nasync def _try_wayback", 1)
    elif fault == "decoy":
        source = b'decoy = ' + repr(source.decode()).encode() + b'\n'
    else:
        source = source.replace(b'\n', b'\r\n')
    with pytest.raises(seam.ArchiveHttpsSeamError):
        seam.convert_archive_https_seam(source)


def test_composer_requires_explicit_archive_source_and_retains_every_other_byte():
    from scripts import build_arena_reviewer_runtime as builder
    from scripts import arena_reviewer_archive_https_seam as seam
    sources = {
        name: (builder.ROOT / ('lab_arena/scorer_entrypoint.py'
                              if name == 'scorer_entrypoint.py' else name)).read_bytes()
        for name in (builder.GRADER_FILE, *builder.ENTRYPOINT_FILES)
    }
    sources[builder.GRADER_FILE] = frozen_source("native_industry_budget", builder.GRADER_FILE)
    prior = builder.expected_overlay(sources)
    assert builder.ARCHIVE_FILE not in prior
    with pytest.raises(builder.CompositionError):
        builder.expected_overlay(sources, archive_snapshot_https=True)
    sources[builder.ARCHIVE_FILE] = original_source()
    repaired = builder.expected_overlay(sources, archive_snapshot_https=True)
    assert set(repaired) == set(prior) | {builder.ARCHIVE_FILE}
    assert repaired[builder.ARCHIVE_FILE] == seam.convert_archive_https_seam(sources[builder.ARCHIVE_FILE])
    assert all(repaired[name] == value for name, value in prior.items())


def test_archive_composer_flag_extracts_and_reports_exact_overlay(monkeypatch, capsys):
    import hashlib
    import json
    from copy import deepcopy
    from scripts import build_arena_reviewer_runtime as builder
    from scripts import arena_reviewer_archive_https_seam as seam
    sources = {
        name: (builder.ROOT / ('lab_arena/scorer_entrypoint.py'
                              if name == 'scorer_entrypoint.py' else name)).read_bytes()
        for name in (builder.GRADER_FILE, *builder.ENTRYPOINT_FILES)
    }
    sources[builder.GRADER_FILE] = frozen_source("native_industry_budget", builder.GRADER_FILE)
    sources[builder.ARCHIVE_FILE] = original_source()
    base = {'Architecture': 'amd64', 'Os': 'linux', 'Config': {}, 'RootFS': {'Layers': ['sha256:base']}}
    candidate = deepcopy(base)
    candidate['RootFS']['Layers'].append('sha256:overlay')
    grading = 'registry.example/model@sha256:' + 'a' * 64
    seen = {}
    monkeypatch.setattr('sys.argv', ['build', '--grading-image', grading, '--verify-image',
                                    'reviewed:runtime', '--archive-snapshot-https'])
    monkeypatch.setattr(builder, 'checked_output', lambda *a, **k: 'b' * 40)
    monkeypatch.setattr(builder, 'image_inspect', lambda ref: base if ref == grading else candidate)
    monkeypatch.setattr(builder, 'extract_frozen_sources', lambda *a, **k: seen.update(k) or sources)
    monkeypatch.setattr(builder, 'verify_saved_overlay', lambda *a: seen.update(overlay=a[-1]))
    assert builder.main() == 0
    receipt = json.loads(capsys.readouterr().out)
    assert seen['archive_snapshot_https'] is True
    assert receipt['grading_corrections'] == [seam.CORRECTION_ID]
    assert receipt['base_layers_unchanged'] and receipt['process_configuration_unchanged']
    assert receipt['original_files'][builder.ARCHIVE_FILE] == hashlib.sha256(sources[builder.ARCHIVE_FILE]).hexdigest()
    assert seen['overlay'][builder.ARCHIVE_FILE] == seam.convert_archive_https_seam(sources[builder.ARCHIVE_FILE])
