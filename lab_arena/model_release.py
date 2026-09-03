"""Commit the daily king's model to the public sales-agent repository.

After a round publishes with a crowned or defended king, the king's frozen
source archive, which already passed the secret scan at acceptance and again
at publication, is committed to ``leadpoet/leadpoet-sales-agent`` as that
repository's ``model/`` tree, together with a signed release manifest that
binds the round, the signed publication, and the artifact hashes.

Properties:

- One atomic commit per model change: one tree, one commit, one fast-forward
  ref update. The ``model/`` subtree is replaced wholesale so files removed by
  the new model disappear; every other path in the repository is preserved
  through the previous commit's tree.
- Idempotent: when ``arena/current.json`` at the branch head already names
  this ``source_tree_hash`` nothing is written, so a defended king costs no
  commit and a retry after a crash never duplicates one.
- Bounded optimistic concurrency: the ref update is a compare-and-swap against
  the head that was read; a concurrent push is retried a bounded number of
  times onto the new head. This is GitHub's non-force ref update, not a
  transactional guarantee across the blob, tree, and commit writes, which are
  content-addressed and harmless when orphaned.
- Readback: the release is reported only after the branch head reads back as
  the new commit.
- The token is used only in the request header; errors carry status codes and
  short, redacted messages.

An empty repository has no ref and cannot take Git Data API writes, so the
first release bootstraps it with one Contents API commit that adds a README.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import httpx

from lab_arena import contracts
from lab_arena.contracts import ArenaContractError, F, hashed_document, validate_document

DEFAULT_REPOSITORY = "leadpoet/leadpoet-sales-agent"
DEFAULT_BRANCH = "main"
DEFAULT_API_BASE_URL = "https://api.github.com"
API_VERSION = "2022-11-28"
MODEL_ROOT = "model"
MANIFEST_PATH = "arena/current.json"
HISTORY_PREFIX = "arena/history"
README_PATH = "arena/README.md"
RELEASE_SCHEMA_VERSION = "lab_arena.model_release.v1"
RELEASE_RECEIPT_SCHEMA_VERSION = "lab_arena.model_release_receipt.v1"
MAX_REF_RETRIES = 3
MAX_FILES = 4000
BLOB_MODE = "100644"  # never executable: the Arena build writes every source file 0644

_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}$")
_BRANCH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,200}$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_PATH_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,120}(/[A-Za-z0-9_][A-Za-z0-9._-]{0,120}){0,15}$")

README_TEXT = """# Leadpoet sales agent

This repository holds the model that currently leads the Leadpoet Lab Arena.
The `model/` tree is the exact frozen source of the reigning king's package
and is replaced whenever a new king is crowned. `arena/current.json` is the
signed release manifest that binds the model to the round that produced it,
and `arena/history/` keeps one manifest per released round.

Releases are written by the Arena service only; edit nothing here by hand.
"""


class ModelReleaseError(RuntimeError):
    """A release step failed. Messages never carry credentials or file contents."""


class RefConflict(ModelReleaseError):
    """The branch moved between the head read and the ref update."""


MODEL_RELEASE_FIELDS = (
    F("schema_version", "str", choices=(RELEASE_SCHEMA_VERSION,)),
    F("repository", "str", minimum=3, maximum=201),
    F("branch", "str", minimum=1, maximum=201),
    F("round_id", "str", minimum=6, maximum=64),
    F("king_hotkey", "str", minimum=40, maximum=64),
    F("king_outcome", "str", choices=("crowned", "defended")),
    F("submission_id", "str", minimum=6, maximum=64),
    F("package_hash", "sha256"),
    F("source_tree_hash", "sha256"),
    F("image_digest", "str", minimum=71, maximum=280),
    F("entry_point", "str", minimum=1, maximum=200),
    F("dependency_lock", "list[str]", minimum=0, maximum=200),
    F("base_image_digest", "str", minimum=71, maximum=280),
    F("file_count", "int", minimum=1, maximum=MAX_FILES),
    F("configuration_hash", "sha256"),
    F("result_bundle_hash", "sha256"),
    F("publication_hash", "sha256"),
    F("reward_basis_hash", "sha256"),
    F("signing_public_key_hash", "sha256"),
    F("released_at", "str", minimum=20, maximum=40),
)


def release_manifest(
    *,
    repository: str,
    branch: str,
    round_id: str,
    king_hotkey: str,
    king_outcome: str,
    submission_id: str,
    package_hash: str,
    source_tree_hash: str,
    image_digest: str,
    entry_point: str,
    dependency_lock: Sequence[str],
    base_image_digest: str,
    file_count: int,
    configuration_hash: str,
    result_bundle_hash: str,
    publication_hash: str,
    reward_basis_hash: str,
    signing_public_key_hash: str,
    released_at: str,
) -> Dict[str, Any]:
    """The unsigned release manifest; the service signs it under ``release_hash``."""

    document = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "repository": repository,
        "branch": branch,
        "round_id": round_id,
        "king_hotkey": king_hotkey,
        "king_outcome": king_outcome,
        "submission_id": submission_id,
        "package_hash": package_hash,
        "source_tree_hash": source_tree_hash,
        "image_digest": image_digest,
        "entry_point": entry_point,
        "dependency_lock": list(dependency_lock),
        "base_image_digest": base_image_digest,
        "file_count": int(file_count),
        "configuration_hash": configuration_hash,
        "result_bundle_hash": result_bundle_hash,
        "publication_hash": publication_hash,
        "reward_basis_hash": reward_basis_hash,
        "signing_public_key_hash": signing_public_key_hash,
        "released_at": released_at,
    }
    validate_document(document, MODEL_RELEASE_FIELDS)
    if not _REPOSITORY_RE.match(repository) or not _BRANCH_RE.match(branch):
        raise ArenaContractError("model release repository or branch is invalid")
    return hashed_document(document, "release_hash")


def validate_model_files(files: Mapping[str, bytes]) -> Dict[str, bytes]:
    """The exact file set that becomes ``model/``: bounded, plain paths, bytes only."""

    if not isinstance(files, Mapping) or not files:
        raise ModelReleaseError("model release has no files")
    if len(files) > MAX_FILES:
        raise ModelReleaseError("model release has too many files")
    ordered: Dict[str, bytes] = {}
    for path in sorted(files):
        if not isinstance(path, str) or not _PATH_RE.match(path) or ".." in path.split("/"):
            raise ModelReleaseError("model release file path is invalid")
        content = files[path]
        if not isinstance(content, (bytes, bytearray)):
            raise ModelReleaseError("model release file content must be bytes")
        ordered[path] = bytes(content)
    return ordered


@dataclass(frozen=True)
class ReleaseReceipt:
    changed: bool
    commit_sha: str
    parent_sha: Optional[str]
    tree_sha: Optional[str]
    branch: str
    repository: str
    manifest_hash: str

    def to_document(self) -> Dict[str, Any]:
        return {
            "schema_version": RELEASE_RECEIPT_SCHEMA_VERSION,
            "changed": bool(self.changed),
            "commit_sha": self.commit_sha,
            "parent_sha": self.parent_sha,
            "tree_sha": self.tree_sha,
            "branch": self.branch,
            "repository": self.repository,
            "release_hash": self.manifest_hash,
        }


class GitHubClient:
    """The few GitHub REST calls the release needs, over HTTP/1.1 with no redirects."""

    def __init__(
        self,
        repository: str,
        token: str,
        *,
        base_url: str = DEFAULT_API_BASE_URL,
        http_client: Optional[httpx.Client] = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not _REPOSITORY_RE.match(repository or ""):
            raise ModelReleaseError("repository must be owner/name")
        if not isinstance(token, str) or not token.strip() or any(ch.isspace() for ch in token.strip()):
            raise ModelReleaseError("GitHub token is missing or malformed")
        self._repository = repository
        self._base_url = base_url.rstrip("/")
        self._client = http_client or httpx.Client(http1=True, http2=False, timeout=httpx.Timeout(timeout_seconds), follow_redirects=False)
        self._headers = {
            "Authorization": "Bearer " + token.strip(),
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": "leadpoet-lab-arena",
        }

    def __repr__(self) -> str:
        return "GitHubClient(%r)" % self._repository

    @property
    def repository(self) -> str:
        return self._repository

    def _request(self, method: str, path: str, *, json_body: Any = None, params: Optional[Mapping[str, str]] = None, expected: Tuple[int, ...] = (200,)) -> Tuple[int, Any]:
        url = "%s/repos/%s/%s" % (self._base_url, self._repository, path.lstrip("/"))
        try:
            response = self._client.request(method, url, headers=self._headers, json=json_body, params=dict(params or {}))
        except httpx.HTTPError as exc:
            raise ModelReleaseError("GitHub %s %s transport failure: %s" % (method, path, type(exc).__name__)) from exc
        if response.status_code not in expected:
            raise ModelReleaseError("GitHub %s %s failed: HTTP %d" % (method, path, response.status_code))
        if response.status_code == 204 or not response.content:
            return response.status_code, None
        try:
            return response.status_code, response.json()
        except ValueError:
            raise ModelReleaseError("GitHub %s %s returned non-JSON" % (method, path)) from None

    def branch_head(self, branch: str) -> Optional[str]:
        """The branch's commit sha, or None when the branch or repository is empty."""

        status, payload = self._request("GET", "git/ref/heads/%s" % branch, expected=(200, 404, 409))
        if status in (404, 409):
            return None
        sha = (payload or {}).get("object", {}).get("sha") if isinstance(payload, dict) else None
        if not isinstance(sha, str) or not _SHA_RE.match(sha):
            raise ModelReleaseError("GitHub returned an invalid branch head")
        return sha

    def commit_tree(self, commit_sha: str) -> str:
        _status, payload = self._request("GET", "git/commits/%s" % commit_sha)
        sha = (payload or {}).get("tree", {}).get("sha") if isinstance(payload, dict) else None
        if not isinstance(sha, str) or not _SHA_RE.match(sha):
            raise ModelReleaseError("GitHub returned an invalid commit tree")
        return sha

    def file_at(self, path: str, ref: str) -> Optional[bytes]:
        status, payload = self._request("GET", "contents/%s" % path, params={"ref": ref}, expected=(200, 404))
        if status == 404:
            return None
        if not isinstance(payload, dict) or payload.get("encoding") != "base64" or not isinstance(payload.get("content"), str):
            raise ModelReleaseError("GitHub returned an unexpected contents shape")
        try:
            return base64.b64decode(payload["content"].encode("ascii"), validate=False)
        except (ValueError, UnicodeEncodeError):
            raise ModelReleaseError("GitHub returned undecodable contents") from None

    def create_file(self, path: str, content: bytes, message: str, branch: str) -> str:
        """Contents API write: the only call that works on an empty repository."""

        body = {"message": message, "content": base64.b64encode(content).decode("ascii"), "branch": branch}
        _status, payload = self._request("PUT", "contents/%s" % path, json_body=body, expected=(200, 201))
        sha = (payload or {}).get("commit", {}).get("sha") if isinstance(payload, dict) else None
        if not isinstance(sha, str) or not _SHA_RE.match(sha):
            raise ModelReleaseError("GitHub returned an invalid bootstrap commit")
        return sha

    def create_blob(self, content: bytes) -> str:
        _status, payload = self._request("POST", "git/blobs", json_body={"content": base64.b64encode(content).decode("ascii"), "encoding": "base64"}, expected=(201,))
        return self._sha_of(payload, "blob")

    def create_tree(self, entries: Sequence[Mapping[str, Any]], *, base_tree: Optional[str] = None) -> str:
        body: Dict[str, Any] = {"tree": [dict(entry) for entry in entries]}
        if base_tree:
            body["base_tree"] = base_tree
        _status, payload = self._request("POST", "git/trees", json_body=body, expected=(201,))
        return self._sha_of(payload, "tree")

    def create_commit(self, message: str, tree_sha: str, parents: Sequence[str]) -> str:
        _status, payload = self._request("POST", "git/commits", json_body={"message": message, "tree": tree_sha, "parents": list(parents)}, expected=(201,))
        return self._sha_of(payload, "commit")

    def update_ref(self, branch: str, commit_sha: str) -> None:
        """Fast-forward only: a moved head is reported as ``RefConflict``."""

        status, _payload = self._request("PATCH", "git/refs/heads/%s" % branch, json_body={"sha": commit_sha, "force": False}, expected=(200, 409, 422))
        if status in (409, 422):
            raise RefConflict("branch %s moved during the release" % branch)

    @staticmethod
    def _sha_of(payload: Any, kind: str) -> str:
        sha = payload.get("sha") if isinstance(payload, dict) else None
        if not isinstance(sha, str) or not _SHA_RE.match(sha):
            raise ModelReleaseError("GitHub returned an invalid %s sha" % kind)
        return sha


def current_manifest(client: GitHubClient, branch: str, head: str) -> Optional[Dict[str, Any]]:
    raw = client.file_at(MANIFEST_PATH, head)
    if raw is None:
        return None
    try:
        document = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise ModelReleaseError("the repository manifest is not valid JSON") from None
    if not isinstance(document, dict):
        raise ModelReleaseError("the repository manifest is not an object")
    return document


def release_king_model(
    client: GitHubClient,
    *,
    branch: str,
    manifest: Mapping[str, Any],
    files: Mapping[str, bytes],
    log: Optional[Callable[[str], None]] = None,
) -> ReleaseReceipt:
    """Make the repository's ``model/`` tree equal the king's files, atomically and idempotently."""

    if not _BRANCH_RE.match(branch):
        raise ModelReleaseError("branch name is invalid")
    try:
        intact = manifest.get("schema_version") == RELEASE_SCHEMA_VERSION and contracts.verify_hashed_document(manifest, "release_hash") == manifest.get("release_hash")
    except ArenaContractError:
        intact = False
    if not intact:
        raise ModelReleaseError("release manifest is unsigned or altered")
    if manifest.get("repository") != client.repository or manifest.get("branch") != branch:
        raise ModelReleaseError("release manifest names a different repository or branch")
    ordered = validate_model_files(files)
    if len(ordered) != int(manifest["file_count"]):
        raise ModelReleaseError("release manifest file count differs from the file set")
    manifest_bytes = contracts.canonical_json(dict(manifest)).encode("utf-8")
    emit = log or (lambda message: None)

    head = client.branch_head(branch)
    if head is None:
        emit("bootstrapping empty repository")
        client.create_file(README_PATH, README_TEXT.encode("utf-8"), "Initialize Lab Arena model releases", branch)
        head = client.branch_head(branch)
        if head is None:
            raise ModelReleaseError("repository bootstrap produced no branch head")

    blob_cache: Dict[str, str] = {}

    def blob(content: bytes) -> str:
        key = contracts.hash_bytes(content)
        if key not in blob_cache:
            blob_cache[key] = client.create_blob(content)
        return blob_cache[key]

    model_tree: Optional[str] = None
    for attempt in range(MAX_REF_RETRIES):
        existing = current_manifest(client, branch, head)
        if existing is not None and existing.get("source_tree_hash") == manifest["source_tree_hash"] and existing.get("release_hash") == manifest["release_hash"]:
            emit("repository already holds this model")
            return ReleaseReceipt(False, head, None, None, branch, client.repository, str(manifest["release_hash"]))
        if existing is not None and existing.get("source_tree_hash") == manifest["source_tree_hash"]:
            # Same model, different round (a defended king): the tree is unchanged and no commit is made.
            emit("repository already holds this model from an earlier round")
            return ReleaseReceipt(False, head, None, None, branch, client.repository, str(manifest["release_hash"]))
        if model_tree is None:
            entries = [{"path": path, "mode": BLOB_MODE, "type": "blob", "sha": blob(content)} for path, content in ordered.items()]
            model_tree = client.create_tree(entries)
        root_entries = [
            {"path": MODEL_ROOT, "mode": "040000", "type": "tree", "sha": model_tree},
            {"path": MANIFEST_PATH, "mode": BLOB_MODE, "type": "blob", "sha": blob(manifest_bytes)},
            {"path": "%s/%s.json" % (HISTORY_PREFIX, manifest["round_id"]), "mode": BLOB_MODE, "type": "blob", "sha": blob(manifest_bytes)},
        ]
        root_tree = client.create_tree(root_entries, base_tree=client.commit_tree(head))
        message = "Lab Arena %s: %s king %s (%s)" % (manifest["round_id"], manifest["king_outcome"], manifest["king_hotkey"], manifest["source_tree_hash"][:19])
        commit = client.create_commit(message, root_tree, [head])
        try:
            client.update_ref(branch, commit)
        except RefConflict:
            moved = client.branch_head(branch)
            emit("branch moved during release attempt %d" % (attempt + 1))
            if moved is None or moved == head:
                raise
            head = moved
            continue
        readback = client.branch_head(branch)
        if readback != commit:
            raise ModelReleaseError("branch head did not read back as the release commit")
        return ReleaseReceipt(True, commit, head, root_tree, branch, client.repository, str(manifest["release_hash"]))
    raise ModelReleaseError("branch kept moving; release not applied after %d attempts" % MAX_REF_RETRIES)
