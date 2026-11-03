"""Assessment-owned worker snapshots; standalone verification stays stateless.

Only ordinary JSON containers are owned. These private containers reject normal
mutation APIs and share immutable strings; trusted validators must not bypass
them with explicit dict/list base-class mutation calls.
"""

from dataclasses import dataclass
import hashlib
import json

from lab_arena import contracts


MAX_EXACT_PRIVACY_PROOFS = 1024


def _exact_privacy_digest(value, response):
    """Hash current plain JSON in its original order, including the scan mode."""
    def validate(item):
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str:
                    raise _NotPlainJSON()
                validate(child)
        elif type(item) is list:
            for child in item:
                validate(child)
        elif item is not None and type(item) not in (str, bool, int, float):
            raise _NotPlainJSON()

    if type(response) is not bool:
        raise _NotPlainJSON()
    validate(value)
    encoded = json.dumps([response, value], ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).digest()


def _immutable(*args, **kwargs):
    raise TypeError("assessment worker payload is read-only")


class _ReadOnlyDict(dict):
    __setattr__ = __delattr__ = _immutable
    __init__ = __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self


class _ReadOnlyList(list):
    __setattr__ = __delattr__ = _immutable
    __init__ = __setitem__ = __delitem__ = append = clear = extend = insert = pop = remove = reverse = sort = __iadd__ = __imul__ = _immutable

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self


class _NotPlainJSON(TypeError):
    pass


def _freeze(value):
    if type(value) is dict:
        result = dict.__new__(_ReadOnlyDict)
        dict.update(result, ((key, _freeze(child)) for key, child in value.items()))
        return result
    if type(value) is list:
        result = list.__new__(_ReadOnlyList)
        list.extend(result, (_freeze(child) for child in value))
        return result
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise _NotPlainJSON()


@dataclass(frozen=True)
class _PayloadProof:
    payload: _ReadOnlyDict
    digest: str


class WorkerProofs:
    """One selected example's owned payloads, never shared across assessments."""

    def __init__(self):
        self._proofs = {}
        self._exact_privacy = {}

    def exact_privacy(self, value, *, response, check):
        # Requests remain mutable. Never reuse object identity or retain input
        # bytes; recompute content plus mode even for a previous cache hit.
        try:
            digest = _exact_privacy_digest(value, response)
        except Exception:
            # Unsupported values or hashing failures keep the original scan,
            # including its exceptions. Scanner exceptions are never cached.
            return check(value, response=response)
        key = (digest, response)
        if key in self._exact_privacy:
            return self._exact_privacy[key]
        result = check(value, response=response)
        if type(result) is bool:
            if len(self._exact_privacy) >= MAX_EXACT_PRIVACY_PROOFS:
                del self._exact_privacy[next(iter(self._exact_privacy))]
            self._exact_privacy[key] = result
        return result

    def prepare(self, example):
        view = dict(example)
        artifacts = example.get("worker_observation_artifacts")
        if not isinstance(artifacts, dict):
            return view
        view["worker_observation_artifacts"] = dict(artifacts)
        for run_id, artifact in artifacts.items():
            if not isinstance(artifact, dict) or type(artifact.get("payload")) is not dict:
                continue
            try:
                payload = _freeze(artifact["payload"])
            except _NotPlainJSON:
                # Unknown Python values retain the original stateless checks.
                continue
            proof = _PayloadProof(payload, contracts.document_hash(payload))
            self._proofs[id(payload)] = proof
            view["worker_observation_artifacts"][run_id] = dict(artifact, payload=payload)
        return view

    def payload_hash(self, payload):
        proof = self._proofs.get(id(payload))
        if proof is not None and proof.payload is payload:
            return proof.digest
        return contracts.document_hash(payload)


def payload_hash(payload, worker_proofs=None):
    if type(worker_proofs) is WorkerProofs:
        return worker_proofs.payload_hash(payload)
    return contracts.document_hash(payload)
