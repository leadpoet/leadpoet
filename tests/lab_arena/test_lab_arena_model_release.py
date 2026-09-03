"""The daily king's model is committed to the sales-agent repository.

A fake GitHub keeps blobs, trees, commits, refs, and contents in memory and
enforces the same rules the real API does for these calls: Git Data writes
fail on an empty repository, a tree created without a base holds exactly its
entries, a ref update without force refuses a non-fast-forward, and contents
are resolved by walking the tree at a ref.
"""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any, Dict, List, Optional

import httpx
import pytest

from lab_arena import contracts
from lab_arena import model_release as mr

TOKEN = "ghp_" + "t" * 36
REPO = "leadpoet/leadpoet-sales-agent"


def _sha(kind: str, payload: str) -> str:
    return hashlib.sha1((kind + ":" + payload).encode()).hexdigest()


class FakeGitHub:
    """Enough of api.github.com for the release sequence, with request logging."""

    def __init__(self):
        self.blobs: Dict[str, bytes] = {}
        self.trees: Dict[str, Dict[str, tuple]] = {}  # sha -> {name: (mode, type, sha)}
        self.commits: Dict[str, Dict[str, Any]] = {}
        self.refs: Dict[str, str] = {}
        self.requests: List[httpx.Request] = []
        self.move_head_once: Optional[str] = None  # branch whose head moves right before the first ref update
        self.fail_after_commit = False

    # -- helpers -----------------------------------------------------------
    def _put_tree(self, entries: Dict[str, tuple]) -> str:
        sha = _sha("tree", json.dumps(sorted(entries.items())))
        self.trees[sha] = dict(entries)
        return sha

    def _put_blob(self, content: bytes) -> str:
        sha = _sha("blob", content.hex())
        self.blobs[sha] = content
        return sha

    def _put_commit(self, message: str, tree: str, parents: List[str]) -> str:
        sha = _sha("commit", json.dumps([message, tree, parents, len(self.commits)]))
        self.commits[sha] = {"message": message, "tree": tree, "parents": parents}
        return sha

    def _tree_with_path(self, base: Optional[str], path: str, entry: tuple) -> str:
        """Return the sha of a copy of ``base`` with ``path`` set (nested paths create subtrees)."""

        entries = dict(self.trees.get(base, {})) if base else {}
        head, _, rest = path.partition("/")
        if rest:
            sub = entries.get(head)
            sub_sha = sub[2] if sub and sub[1] == "tree" else None
            entries[head] = ("040000", "tree", self._tree_with_path(sub_sha, rest, entry))
        else:
            entries[head] = entry
        return self._put_tree(entries)

    def files_at(self, commit: str, prefix: str = "") -> Dict[str, bytes]:
        out: Dict[str, bytes] = {}

        def walk(tree_sha: str, path: str):
            for name, (mode, kind, sha) in self.trees[tree_sha].items():
                full = path + name
                if kind == "tree":
                    walk(sha, full + "/")
                else:
                    out[full] = self.blobs[sha]

        walk(self.commits[commit]["tree"], "")
        return {k: v for k, v in out.items() if k.startswith(prefix)}

    def lookup(self, commit: str, path: str):
        tree = self.commits[commit]["tree"]
        parts = path.split("/")
        for part in parts[:-1]:
            entry = self.trees[tree].get(part)
            if not entry or entry[1] != "tree":
                return None
            tree = entry[2]
        return self.trees[tree].get(parts[-1])

    def seed(self, branch: str, files: Dict[str, bytes], message: str = "seed") -> str:
        tree = None
        for path, content in files.items():
            tree = self._tree_with_path(tree, path, ("100644", "blob", self._put_blob(content)))
        commit = self._put_commit(message, tree, [self.refs[branch]] if branch in self.refs else [])
        self.refs[branch] = commit
        return commit

    # -- transport ---------------------------------------------------------
    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert request.headers.get("Authorization") == "Bearer " + TOKEN
        assert request.headers.get("X-GitHub-Api-Version") == mr.API_VERSION
        path = request.url.path
        prefix = "/repos/%s/" % REPO
        assert path.startswith(prefix), path
        route = path[len(prefix):]
        body = json.loads(request.content.decode()) if request.content else {}
        empty = not self.refs
        if route.startswith("git/ref/heads/"):
            branch = route[len("git/ref/heads/"):]
            if empty:
                return httpx.Response(409, json={"message": "Git Repository is empty."})
            if branch not in self.refs:
                return httpx.Response(404, json={"message": "Not Found"})
            return httpx.Response(200, json={"object": {"sha": self.refs[branch]}})
        if route.startswith("git/commits/") and request.method == "GET":
            sha = route[len("git/commits/"):]
            return httpx.Response(200, json={"sha": sha, "tree": {"sha": self.commits[sha]["tree"]}})
        if route.startswith("contents/") and request.method == "GET":
            file_path = route[len("contents/"):]
            ref = request.url.params.get("ref")
            if empty or ref not in self.commits:
                return httpx.Response(404, json={"message": "Not Found"})
            entry = self.lookup(ref, file_path)
            if entry is None or entry[1] != "blob":
                return httpx.Response(404, json={"message": "Not Found"})
            return httpx.Response(200, json={"encoding": "base64", "content": base64.b64encode(self.blobs[entry[2]]).decode(), "sha": entry[2]})
        if route.startswith("contents/") and request.method == "PUT":
            file_path = route[len("contents/"):]
            branch = body["branch"]
            content = base64.b64decode(body["content"])
            parent = self.refs.get(branch)
            base_tree = self.commits[parent]["tree"] if parent else None
            tree = self._tree_with_path(base_tree, file_path, ("100644", "blob", self._put_blob(content)))
            commit = self._put_commit(body["message"], tree, [parent] if parent else [])
            self.refs[branch] = commit
            return httpx.Response(201, json={"commit": {"sha": commit}})
        if empty:
            return httpx.Response(409, json={"message": "Git Repository is empty."})
        if route == "git/blobs":
            assert body["encoding"] == "base64"
            return httpx.Response(201, json={"sha": self._put_blob(base64.b64decode(body["content"]))})
        if route == "git/trees":
            base = body.get("base_tree")
            tree_sha = base
            entries = dict(self.trees[base]) if base else {}
            tree_sha = self._put_tree(entries)
            for entry in body["tree"]:
                assert entry["mode"] in ("100644", "040000"), entry
                tree_sha = self._tree_with_path(tree_sha, entry["path"], (entry["mode"], entry["type"], entry["sha"]))
            return httpx.Response(201, json={"sha": tree_sha})
        if route == "git/commits" and request.method == "POST":
            if self.fail_after_commit:
                return httpx.Response(500, json={"message": "boom"})
            return httpx.Response(201, json={"sha": self._put_commit(body["message"], body["tree"], list(body["parents"]))})
        if route.startswith("git/refs/heads/") and request.method == "PATCH":
            branch = route[len("git/refs/heads/"):]
            if self.move_head_once == branch:
                self.move_head_once = None
                self.seed(branch, {"docs/other.md": b"someone else pushed\n"}, "concurrent push")
            new = body["sha"]
            current = self.refs[branch]
            if body.get("force") or self.commits[new]["parents"][:1] == [current]:
                self.refs[branch] = new
                return httpx.Response(200, json={"object": {"sha": new}})
            return httpx.Response(422, json={"message": "Update is not a fast forward"})
        return httpx.Response(404, json={"message": "unhandled %s %s" % (request.method, route)})


def client_for(fake: FakeGitHub) -> mr.GitHubClient:
    return mr.GitHubClient(REPO, TOKEN, http_client=httpx.Client(transport=httpx.MockTransport(fake.handler)))


def manifest_for(round_id: str, source_files: Dict[str, bytes], *, outcome: str = "crowned", hotkey: str = "5" + "k" * 47) -> Dict[str, Any]:
    tree_hash = contracts.document_hash({path: contracts.hash_bytes(content) for path, content in sorted(source_files.items())})
    return mr.release_manifest(
        repository=REPO, branch="main", round_id=round_id, king_hotkey=hotkey, king_outcome=outcome, submission_id="sub-" + "a" * 32,
        package_hash=contracts.document_hash(round_id + "pkg"), source_tree_hash=tree_hash, image_digest="sha256:" + "1" * 64, entry_point="main.py",
        dependency_lock=["h11==0.16.0"], base_image_digest="sha256:" + "2" * 64, file_count=len(source_files), configuration_hash=contracts.document_hash("cfg"),
        result_bundle_hash=contracts.document_hash("bundle"), publication_hash=contracts.document_hash("pub"), reward_basis_hash=contracts.document_hash("basis"),
        signing_public_key_hash=contracts.document_hash("key"), released_at="2026-09-03T00:10:00+00:00",
    )


FILES_A = {"main.py": b"print('a')\n", "model/__init__.py": b"", "model/agent.py": b"AGENT = 'a'\n", "requirements.lock": b"h11==0.16.0\n"}
FILES_B = {"main.py": b"print('b')\n", "pkg/core.py": b"CORE = 'b'\n", "requirements.lock": b"h11==0.16.0\n"}


def test_first_release_bootstraps_the_empty_repository_and_writes_model_and_manifest():
    fake = FakeGitHub()
    manifest = manifest_for("arena-2026-09-03", FILES_A)
    receipt = mr.release_king_model(client_for(fake), branch="main", manifest=manifest, files=FILES_A)
    assert receipt.changed and receipt.commit_sha == fake.refs["main"] and receipt.repository == REPO
    files = fake.files_at(receipt.commit_sha)
    assert {k: v for k, v in files.items() if k.startswith("model/")} == {"model/" + path: content for path, content in FILES_A.items()}
    assert json.loads(files["arena/current.json"]) == manifest and json.loads(files["arena/history/arena-2026-09-03.json"]) == manifest
    assert files["arena/README.md"] == mr.README_TEXT.encode()
    # The bootstrap commit is the parent; every blob is a plain 0644 file.
    assert fake.commits[receipt.commit_sha]["parents"] == [receipt.parent_sha] and fake.commits[receipt.parent_sha]["message"].startswith("Initialize")
    modes = {entry[0] for tree in fake.trees.values() for entry in tree.values() if entry[1] == "blob"}
    assert modes == {"100644"}
    assert "crowned king" in fake.commits[receipt.commit_sha]["message"]


def test_a_defended_king_makes_no_commit_and_a_retry_is_idempotent():
    fake = FakeGitHub()
    first = mr.release_king_model(client_for(fake), branch="main", manifest=manifest_for("arena-2026-09-03", FILES_A), files=FILES_A)
    count = len(fake.commits)
    again = mr.release_king_model(client_for(fake), branch="main", manifest=manifest_for("arena-2026-09-03", FILES_A), files=FILES_A)
    defended = mr.release_king_model(client_for(fake), branch="main", manifest=manifest_for("arena-2026-09-04", FILES_A, outcome="defended"), files=FILES_A)
    assert not again.changed and not defended.changed and again.commit_sha == defended.commit_sha == first.commit_sha
    assert len(fake.commits) == count and fake.refs["main"] == first.commit_sha


def test_a_new_king_replaces_the_model_tree_wholesale_and_keeps_other_paths():
    fake = FakeGitHub()
    fake.seed("main", {"README.md": b"# hand-written\n", "arena/README.md": b"old readme\n", "model/stale.py": b"should disappear\n"})
    a = mr.release_king_model(client_for(fake), branch="main", manifest=manifest_for("arena-2026-09-03", FILES_A), files=FILES_A)
    b = mr.release_king_model(client_for(fake), branch="main", manifest=manifest_for("arena-2026-09-04", FILES_B, hotkey="5" + "z" * 47), files=FILES_B)
    assert a.changed and b.changed and b.parent_sha == a.commit_sha and fake.refs["main"] == b.commit_sha
    files = fake.files_at(b.commit_sha)
    assert {k for k in files if k.startswith("model/")} == {"model/" + path for path in FILES_B}
    assert files["README.md"] == b"# hand-written\n"  # preserved through the base tree
    assert set(k for k in files if k.startswith("arena/history/")) == {"arena/history/arena-2026-09-03.json", "arena/history/arena-2026-09-04.json"}
    assert json.loads(files["arena/current.json"])["round_id"] == "arena-2026-09-04"
    # No bootstrap happened on a non-empty repository.
    assert not any(request.method == "PUT" for request in fake.requests)


def test_a_concurrent_push_is_retried_onto_the_new_head():
    fake = FakeGitHub()
    fake.seed("main", {"README.md": b"# repo\n"})
    fake.move_head_once = "main"
    receipt = mr.release_king_model(client_for(fake), branch="main", manifest=manifest_for("arena-2026-09-03", FILES_A), files=FILES_A)
    assert receipt.changed and fake.refs["main"] == receipt.commit_sha
    parent = fake.commits[receipt.commit_sha]["parents"][0]
    assert fake.commits[parent]["message"] == "concurrent push"
    files = fake.files_at(receipt.commit_sha)
    assert files["docs/other.md"] == b"someone else pushed\n" and files["model/main.py"] == FILES_A["main.py"]
    # The model tree was built once; only the root tree and commit were redone.
    assert sum(1 for r in fake.requests if r.method == "POST" and r.url.path.endswith("git/blobs")) == len(FILES_A) + 1


def test_manifest_and_files_are_validated_and_errors_never_carry_the_token():
    fake = FakeGitHub()
    client = client_for(fake)
    manifest = manifest_for("arena-2026-09-03", FILES_A)
    altered = dict(manifest, king_hotkey="5" + "q" * 47)
    with pytest.raises(mr.ModelReleaseError, match="unsigned or altered"):
        mr.release_king_model(client, branch="main", manifest=altered, files=FILES_A)
    with pytest.raises(mr.ModelReleaseError, match="different repository"):
        mr.release_king_model(client, branch="release", manifest=manifest, files=FILES_A)
    with pytest.raises(mr.ModelReleaseError, match="file count"):
        mr.release_king_model(client, branch="main", manifest=manifest, files=dict(FILES_A, extra=b"x"))
    with pytest.raises(mr.ModelReleaseError, match="path is invalid"):
        mr.validate_model_files({"../escape.py": b""})
    with pytest.raises(mr.ModelReleaseError, match="no files"):
        mr.validate_model_files({})
    with pytest.raises(mr.ModelReleaseError):
        mr.GitHubClient(REPO, "bad token with spaces")
    with pytest.raises(mr.ModelReleaseError):
        mr.GitHubClient("not-a-repo", TOKEN)
    fake.fail_after_commit = True
    with pytest.raises(mr.ModelReleaseError) as excinfo:
        mr.release_king_model(client, branch="main", manifest=manifest, files=FILES_A)
    assert TOKEN not in str(excinfo.value) and "HTTP 500" in str(excinfo.value)
    assert TOKEN not in repr(client)
    with pytest.raises(contracts.ArenaContractError):
        manifest_for("arena-2026-09-03", FILES_A, outcome="retained")


def test_receipt_document_is_plain_and_hash_bound():
    fake = FakeGitHub()
    manifest = manifest_for("arena-2026-09-03", FILES_A)
    receipt = mr.release_king_model(client_for(fake), branch="main", manifest=manifest, files=FILES_A)
    document = receipt.to_document()
    assert document["schema_version"] == mr.RELEASE_RECEIPT_SCHEMA_VERSION and document["release_hash"] == manifest["release_hash"]
    assert document["commit_sha"] == fake.refs["main"] and document["changed"] is True and document["repository"] == REPO
    contracts.check_strict_document(document)
