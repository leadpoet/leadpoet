"""One normal Arena validator: score leases and publish independently derived weights.

It depends only on the Arena public API, finalized chain reads, and the
validator's local Bittensor hotkey. Scoring availability never gates weights.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import stat
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence

from lab_arena.contracts import document_hash

MAX_ARENA_WEIGHT_ATTEMPTS = 3


class ArenaValidatorError(RuntimeError):
    """The Arena validator cannot safely continue its current operation."""


class ArenaSignerClient(Protocol):
    def prepare_arena_weight_extrinsic_v1(self, request: Dict[str, Any]) -> Dict[str, Any]: ...
    def confirm_arena_weight_extrinsic_v1(self, request: Dict[str, Any]) -> Dict[str, Any]: ...
    def recover_arena_weight_extrinsic_v1(self, request: Dict[str, Any]) -> Dict[str, Any]: ...
    def sign_arena_chain_outcome_v1(self, outcome_document: Dict[str, Any]) -> Dict[str, Any]: ...


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, sort_keys=True, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        directory = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArenaValidatorError("Arena weight journal is unreadable") from exc
    if not isinstance(value, dict):
        raise ArenaValidatorError("Arena weight journal is invalid")
    return value


def _read_hashed_json(path: Path) -> Optional[Dict[str, Any]]:
    value = _read_json(path)
    if value is None:
        return None
    claimed = value.get("record_hash")
    body = {key: item for key, item in value.items() if key != "record_hash"}
    if claimed != document_hash(body):
        raise ArenaValidatorError("Arena weight journal hash is invalid")
    return value


class ArenaPublicApi:
    def __init__(self, base_url: str, *, timeout_seconds: int = 30) -> None:
        self.base_url = str(base_url).rstrip("/")
        if not self.base_url.startswith(("https://", "http://")):
            raise ArenaValidatorError("Arena API URL must be explicit HTTP(S)")
        self.timeout_seconds = int(timeout_seconds)

    def _get(self, path: str) -> Dict[str, Any]:
        request = urllib.request.Request(self.base_url + path, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read(2 * 1024 * 1024 + 1)
        except (OSError, urllib.error.HTTPError) as exc:
            raise ArenaValidatorError("Arena public API read failed") from exc
        if len(body) > 2 * 1024 * 1024:
            raise ArenaValidatorError("Arena public API response is too large")
        try:
            value = json.loads(body.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ArenaValidatorError("Arena public API returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise ArenaValidatorError("Arena public API response must be an object")
        return value

    def signing_key(self) -> Dict[str, Any]:
        return self._get("/arena/v1/signing-key")

    def accepted_weight_state(self, epoch: int) -> Optional[Dict[str, Any]]:
        value = self._get("/arena/v1/weight-state?" + urllib.parse.urlencode({"epoch": int(epoch)}))
        if value.get("lookup_ok") is not True or set(value) != {"lookup_ok", "state"}:
            raise ArenaValidatorError("Arena accepted-state response is invalid")
        state = value["state"]
        if state is None:
            return None
        if not isinstance(state, dict):
            raise ArenaValidatorError("Arena accepted state must be an object")
        return state

    def submit_chain_outcome(self, document: Mapping[str, Any]) -> Dict[str, Any]:
        encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/arena/v1/chain-outcomes", data=encoded, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                value = json.loads(response.read(1024 * 1024 + 1).decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, urllib.error.HTTPError) as exc:
            raise ArenaValidatorError("Arena chain outcome submission failed") from exc
        if not isinstance(value, dict):
            raise ArenaValidatorError("Arena chain outcome acknowledgment is invalid")
        if value.get("status") != "recorded":
            raise ArenaValidatorError("Arena chain outcome was not recorded")
        return value


@dataclass(frozen=True)
class ArenaWeightPaths:
    root: Path

    def signed(self, epoch: int) -> Path:
        return self.root / ("epoch-%d-signed.json" % int(epoch))

    def outcome(self, epoch: int) -> Path:
        return self.root / ("epoch-%d-outcome.json" % int(epoch))

    def archived_attempt(self, epoch: int, sequence: int) -> Path:
        return self.root / ("epoch-%d-attempt-%d-signed.json" % (int(epoch), int(sequence)))


class ArenaWeightOrchestrator:
    """Crash-safe host half of the Arena weight protocol."""

    def __init__(
        self,
        *,
        api: ArenaPublicApi,
        chain: Any,
        signer: ArenaSignerClient,
        validator_hotkey: str,
        expected_signing_key_hash: str,
        paths: ArenaWeightPaths,
        extrinsic_period: int,
    ) -> None:
        self.api = api
        self.chain = chain
        self.signer = signer
        self.validator_hotkey = str(validator_hotkey)
        self.expected_signing_key_hash = str(expected_signing_key_hash)
        self.paths = paths
        self.extrinsic_period = int(extrinsic_period)
        self._last_confirmation = None
        if self.extrinsic_period <= 0:
            raise ArenaValidatorError("protected Arena extrinsic period is invalid")

    def _verified_state(self, epoch: int, signing_key: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        from leadpoet_canonical.arena_weights import (
            validate_accepted_weight_state,
            verify_accepted_weight_state_signature,
        )
        from lab_arena.signing import load_public_key_from_document

        public_key = load_public_key_from_document(signing_key)
        if signing_key.get("public_key_hash") != self.expected_signing_key_hash:
            raise ArenaValidatorError("Arena signing key differs from the pinned key")
        state = self.api.accepted_weight_state(epoch)
        if state is None:
            return None
        verified = validate_accepted_weight_state(state)
        verify_accepted_weight_state_signature(
            verified,
            public_key_der=public_key,
            expected_public_key_hash=self.expected_signing_key_hash,
        )
        genesis = str(self.chain.client.get_block_hash(block_id=0)).lower().removeprefix("0x")
        if (
            int(verified["epoch"]) != int(epoch)
            or int(verified["netuid"]) != int(self.chain.config.netuid)
            or str(verified["network"]) != str(self.chain.config.network_name)
            or str(verified["genesis_hash"]) != genesis
        ):
            raise ArenaValidatorError("Arena accepted state differs from the finalized chain")
        return verified

    def _host_derivation(self, state: Mapping[str, Any], hotkeys: Sequence[str]) -> Dict[str, Any]:
        from leadpoet_canonical.arena_weights import derive_arena_weights

        return dict(derive_arena_weights(state, hotkeys))

    def _confirm(self, signed: Mapping[str, Any]) -> Optional[str]:
        outcome = self.signer.confirm_arena_weight_extrinsic_v1(
            {
                "state_hash": signed["state_hash"],
                "extrinsic_hash": signed["extrinsic_hash"],
            }
        )
        self._last_confirmation = dict(outcome)
        status = outcome.get("status")
        if status in {"pending", "included_pending_reveal"} and outcome.get("finalized") is False:
            if status == "included_pending_reveal":
                return status
            return None
        if status == "not_included_expired" and outcome.get("finalized") is False:
            return status
        if status not in {"finalized", "not_included_expired"}:
            raise ArenaValidatorError("protected Arena finalization status is invalid")
        if status == "finalized" and not {
            "finalized_block_hash", "finalized_block", "commit_included_block"
        }.issubset(outcome):
            raise ArenaValidatorError("protected Arena finalization result is incomplete")
        if status == "finalized" and (
            outcome.get("weights_hash") != signed.get("weights_hash")
            or outcome.get("revealed_weights")
            != [list(item) for item in zip(signed["sparse_uids"], signed["sparse_weights_u16"])]
            or isinstance(outcome.get("validator_uid"), bool)
            or not isinstance(outcome.get("validator_uid"), int)
            or int(outcome.get("last_update", -1)) < int(outcome["commit_included_block"])
        ):
            raise ArenaValidatorError("protected Arena finalized readback differs from signed weights")
        report_document = None
        if status == "finalized":
            core = {
                "schema_version": "leadpoet.arena.chain_outcome.v1",
                "network": self.chain.config.network_name,
                "netuid": int(signed["netuid"]), "epoch": int(signed["epoch"]),
                "validator_hotkey": self.validator_hotkey,
                "state_hash": signed["state_hash"], "weights_hash": signed["weights_hash"],
                "extrinsic_hash": signed["extrinsic_hash"],
                "finalized_block_hash": outcome["finalized_block_hash"],
                "finalized_block_number": int(outcome["finalized_block"]),
                "observed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            request_id = document_hash(core)
            unsigned = {**core, "request_id": request_id}
            signature_result = self.signer.sign_arena_chain_outcome_v1(unsigned)
            signature = str(signature_result.get("signature") or "")
            if not signature:
                raise ArenaValidatorError("protected Arena outcome signature is missing")
            report_document = {**unsigned, "signature": signature}
        record = {
            "schema_version": "leadpoet.arena.validator_chain_outcome.v1",
            "state_hash": signed["state_hash"],
            "epoch": int(signed["epoch"]),
            "extrinsic_hash": signed["extrinsic_hash"],
            "outcome": outcome,
            "report_document": report_document,
            "reported": report_document is None,
        }
        record["record_hash"] = document_hash(record)
        outcome_path = self.paths.outcome(int(signed["epoch"]))
        _atomic_json(outcome_path, record)
        self._report_outcome(record)
        return status

    def _record_expired(self, signed: Mapping[str, Any], outcome: Mapping[str, Any]) -> None:
        record = {
            "schema_version": "leadpoet.arena.validator_chain_outcome.v1",
            "state_hash": signed["state_hash"], "epoch": int(signed["epoch"]),
            "extrinsic_hash": signed["extrinsic_hash"], "outcome": dict(outcome),
            "report_document": None, "reported": True,
        }
        record["record_hash"] = document_hash(record)
        _atomic_json(self.paths.outcome(int(signed["epoch"])), record)

    def _fresh_era_fits(self, state: Mapping[str, Any], current_block: int) -> bool:
        from leadpoet_canonical.hotkey_authority_v2 import mortal_era_bounds

        _birth, death = mortal_era_bounds(
            period=self.extrinsic_period, current=int(current_block)
        )
        return (
            int(state["valid_from_block"]) <= int(current_block)
            and int(death) - 1 <= int(state["valid_until_block"])
        )

    def _recover_protected_state(self, signed: Mapping[str, Any]) -> None:
        accepted_state = signed.get("accepted_state")
        recovery_record = signed.get("recovery_record")
        if not isinstance(accepted_state, Mapping) or not isinstance(recovery_record, Mapping):
            raise ArenaValidatorError("Arena journal lacks protected recovery state")
        recovered = self.signer.recover_arena_weight_extrinsic_v1(
            {"accepted_state": dict(accepted_state), "recovery_record": dict(recovery_record)}
        )
        for field in ("state_hash", "authorization_hash", "extrinsic_hash", "extrinsic_hex"):
            if recovered.get(field) != signed.get(field):
                raise ArenaValidatorError("protected Arena recovery differs from durable journal")

    def _report_outcome(self, record: Mapping[str, Any]) -> None:
        if record.get("reported") is True:
            return
        document = record.get("report_document")
        if not isinstance(document, Mapping):
            raise ArenaValidatorError("Arena outcome report journal is invalid")
        self.api.submit_chain_outcome(document)
        updated = {key: value for key, value in record.items() if key != "record_hash"}
        updated["reported"] = True
        updated["record_hash"] = document_hash(updated)
        _atomic_json(self.paths.outcome(int(updated["epoch"])), updated)

    def _broadcast(self, extrinsic_hex: str) -> None:
        value = str(extrinsic_hex)
        if value.startswith("0x"):
            value = value[2:]
        if not value or any(character not in "0123456789abcdef" for character in value.lower()):
            raise ArenaValidatorError("protected signer returned invalid extrinsic bytes")
        self.chain.client.rpc_request("author_submitExtrinsic", ["0x" + value])

    def run_once(self, epoch: int) -> str:
        signed_path = self.paths.signed(epoch)
        outcome_path = self.paths.outcome(epoch)
        existing = _read_hashed_json(signed_path)
        retry_state = None
        submission_context = None
        retry_to_archive = None
        if existing is not None:
            outcome = _read_hashed_json(outcome_path)
            if outcome is not None and outcome.get("extrinsic_hash") == existing.get("extrinsic_hash"):
                self._report_outcome(outcome)
                return str(outcome["outcome"]["status"])
            self._recover_protected_state(existing)
            confirmation = self._confirm(existing)
            if confirmation in {"finalized", "included_pending_reveal"}:
                return confirmation
            if confirmation == "not_included_expired":
                accepted = existing.get("accepted_state")
                if not isinstance(accepted, Mapping):
                    raise ArenaValidatorError("Arena retry lacks accepted state")
                prior_sequence = int(existing.get("attempt_sequence", 0))
                if prior_sequence <= 0:
                    raise ArenaValidatorError("Arena retry lacks attempt sequence")
                if prior_sequence >= MAX_ARENA_WEIGHT_ATTEMPTS:
                    self._record_expired(
                        existing,
                        self._last_confirmation or {},
                    )
                    return "not_included_expired"
                submission_context = self.chain.finalized_weight_submission_context(
                    self.validator_hotkey
                )
                head, _metagraph, rate_limit_ready = submission_context
                if not self._fresh_era_fits(accepted, head.number):
                    self._record_expired(
                        existing,
                        self._last_confirmation or {},
                    )
                    return "not_included_expired"
                if not rate_limit_ready:
                    return "rate_limited"
                retry_state = dict(accepted)
                retry_to_archive = (prior_sequence, existing)
            head = self.chain.finalized_head()
            if confirmation != "not_included_expired" and int(existing["valid_from_block"]) <= head.number <= int(existing["valid_until_block"]):
                self._broadcast(str(existing["extrinsic_hex"]))
                return "rebroadcast"
            if confirmation != "not_included_expired":
                raise ArenaValidatorError("signed Arena weight extrinsic expired without finalization")

        signing_key = self.api.signing_key()
        state = self._verified_state(epoch, signing_key)
        if state is None:
            return "state_unavailable"
        if retry_state is not None and state != retry_state:
            raise ArenaValidatorError("Arena retry state differs from durable accepted state")
        if submission_context is None:
            submission_context = self.chain.finalized_weight_submission_context(
                self.validator_hotkey
            )
        head, metagraph, rate_limit_ready = submission_context
        if not int(state["valid_from_block"]) <= head.number <= int(state["valid_until_block"]):
            return "outside_submission_window"
        if not rate_limit_ready:
            return "rate_limited"
        host_result = self._host_derivation(state, metagraph.hotkeys)
        substrate = self.chain.client
        nonce = substrate.get_account_nonce(self.validator_hotkey)
        if nonce is None or isinstance(nonce, bool):
            raise ArenaValidatorError("finalized validator nonce is unavailable")
        nonce = int(nonce)
        if nonce < 0:
            raise ArenaValidatorError("finalized validator nonce is invalid")
        era_current = head.number
        era = substrate.runtime_config.create_scale_object("Era")
        era.encode({"period": self.extrinsic_period, "current": era_current})
        birth_hash = str(substrate.get_block_hash(block_id=era.birth(era_current))).removeprefix("0x")
        protected = self.signer.prepare_arena_weight_extrinsic_v1(
            {
                "accepted_state": state,
                "nonce": int(nonce),
                "era_current": era_current,
                "runtime_block_hash": head.hash,
                "block_hash": birth_hash,
            }
        )
        protected_fields = {
            "schema_version", "state_hash", "epoch", "netuid", "finalized_block",
            "finalized_block_hash", "weights_hash", "sparse_uids",
            "sparse_weights_u16", "authorization_hash", "extrinsic_hash",
            "extrinsic_hex",
            "attempt_sequence", "recovery_record",
        }
        if not protected_fields.issubset(protected):
            raise ArenaValidatorError("protected Arena result fields are invalid")
        if (
            int(protected.get("epoch", -1)) != int(epoch)
            or int(protected.get("netuid", -1)) != int(self.chain.config.netuid)
        ):
            raise ArenaValidatorError("protected Arena result targets the wrong epoch or subnet")
        if retry_state is not None and int(protected.get("attempt_sequence", -1)) != int(existing["attempt_sequence"]) + 1:
            raise ArenaValidatorError("protected Arena retry sequence is invalid")
        for field in ("state_hash", "weights_hash", "sparse_uids", "sparse_weights_u16"):
            if protected.get(field) != host_result.get(field):
                raise ArenaValidatorError("protected Arena derivation differs from host derivation")
        protected_body = dict(protected)
        protected_schema = protected_body.pop("schema_version")
        record = {
            **protected_body,
            "schema_version": "leadpoet.arena.validator_signed_weight.v1",
            "protected_schema_version": protected_schema,
            "accepted_state": dict(state),
            "valid_from_block": int(state["valid_from_block"]),
            "valid_until_block": int(state["valid_until_block"]),
        }
        record["record_hash"] = document_hash(record)
        if retry_to_archive is not None:
            prior_sequence, prior_record = retry_to_archive
            # Preserve the old exact bytes only after the signer produced a
            # complete, validated replacement and before replacing the active
            # journal.
            _atomic_json(
                self.paths.archived_attempt(epoch, prior_sequence), prior_record
            )
        _atomic_json(signed_path, record)
        self._broadcast(str(record["extrinsic_hex"]))
        return "broadcast"

    def poll_prior_outcomes(self, current_epoch: int) -> None:
        """Advance every local unfinished journal without blocking a new epoch."""

        candidates = []
        for path in self.paths.root.glob("epoch-*-signed.json"):
            try:
                epoch = int(path.name.removeprefix("epoch-").removesuffix("-signed.json"))
            except ValueError:
                print("Arena validator ignored an invalid journal filename", file=sys.stderr, flush=True)
                continue
            if epoch < int(current_epoch):
                try:
                    outcome = _read_hashed_json(self.paths.outcome(epoch))
                    if outcome is not None:
                        self._report_outcome(outcome)
                    else:
                        candidates.append(epoch)
                except Exception as exc:
                    print(
                        "Arena validator prior outcome failed: epoch=%d type=%s"
                        % (epoch, type(exc).__name__), file=sys.stderr, flush=True,
                    )
        if len(candidates) > 128:
            candidates = sorted(candidates)[-128:]
            print("Arena validator prior outcome polling limited to 128 epochs", file=sys.stderr, flush=True)
        for epoch in sorted(candidates):
            try:
                signed = _read_hashed_json(self.paths.signed(epoch))
                if signed is None:
                    continue
                self._recover_protected_state(signed)
                self._confirm(signed)
            except Exception as exc:
                print(
                    "Arena validator prior recovery failed: epoch=%d type=%s"
                    % (epoch, type(exc).__name__), file=sys.stderr, flush=True,
                )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one normal Leadpoet Arena validator", add_help=True)
    parser.add_argument("--netuid", type=int, default=int(os.environ.get("LAB_ARENA_NETUID", "71")))
    parser.add_argument("--subtensor.network", "--subtensor_network", dest="subtensor_network", default=os.environ.get("LAB_ARENA_NETWORK", "finney"))
    parser.add_argument("--wallet.name", dest="wallet_name", default=os.environ.get("LAB_ARENA_WALLET_NAME", "default"))
    parser.add_argument("--wallet.hotkey", dest="hotkey_name", default=os.environ.get("LAB_ARENA_HOTKEY", "default"))
    parser.add_argument("--wallet.path", dest="wallet_path", default=os.environ.get("LAB_ARENA_WALLET_PATH", "~/.bittensor/wallets"))
    parser.add_argument("--arena-api-base-url", dest="api_base_url", default=os.environ.get("LAB_ARENA_API_BASE_URL", ""))
    parser.add_argument("--arena-work-dir", dest="work_dir", default=os.environ.get("LAB_ARENA_RUNNER_WORK_DIR", "/var/lib/lab-arena/runner"))
    parser.add_argument("--arena-runsc-path", dest="runsc_path", default=os.environ.get("LAB_ARENA_RUNSC_PATH", "/usr/local/bin/runsc"))
    parser.add_argument("--arena-poll-seconds", dest="poll_seconds", type=int, default=int(os.environ.get("LAB_ARENA_VALIDATOR_POLL_SECONDS", "30")))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ArenaValidatorError("%s is required" % name)
    return value


def load_local_hotkey(args):
    """Load an existing private wallet; never create or change an identity."""
    from bittensor_wallet import Wallet

    for name in (args.wallet_name, args.hotkey_name):
        if not name or name in (".", "..") or "/" in name or "\\" in name:
            raise ArenaValidatorError("wallet and hotkey names must be simple file names")
    wallet_path = Path(args.wallet_path).expanduser()
    hotkey_path = wallet_path / args.wallet_name / "hotkeys" / args.hotkey_name
    try:
        metadata = hotkey_path.lstat()
    except OSError as exc:
        raise ArenaValidatorError("configured local validator hotkey file is unavailable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_mode & 0o077
        or (os.geteuid() != 0 and metadata.st_uid != os.geteuid())
    ):
        raise ArenaValidatorError("local validator hotkey must be an owned private regular file")
    try:
        wallet = Wallet(name=args.wallet_name, hotkey=args.hotkey_name, path=str(wallet_path))
        keypair = wallet.hotkey
        # Prove access to signing material, not just a public hotkey file.
        challenge = b"leadpoet.arena.local-wallet-readiness.v1"
        if not keypair.verify(challenge, keypair.sign(challenge)):
            raise ArenaValidatorError("local validator hotkey failed its signing check")
    except Exception:
        raise ArenaValidatorError("local validator hotkey cannot sign; check its private wallet file") from None
    expected = os.environ.get("LAB_ARENA_EXPECTED_HOTKEY", "").strip()
    if expected and keypair.ss58_address != expected:
        raise ArenaValidatorError("local wallet identity differs from LAB_ARENA_EXPECTED_HOTKEY")
    return keypair


def run_validator_loops(*, orchestrator, runner_factory, epoch_supplier, stop,
                        poll_seconds=30, once=False) -> None:
    """Start weights first; retry scoring setup and cycles without stopping them."""
    interval = max(5, int(poll_seconds))
    first_weight_cycle = threading.Event()

    def weight_loop() -> None:
        while not stop.is_set():
            try:
                epoch = int(epoch_supplier())
                try:
                    orchestrator.poll_prior_outcomes(epoch)
                except Exception as exc:
                    print("Arena validator prior polling failed: type=%s" % type(exc).__name__,
                          file=sys.stderr, flush=True)
                status = orchestrator.run_once(epoch)
                print("Arena validator weight status: %s" % status, flush=True)
            except Exception as exc:
                print("Arena validator weight cycle failed: %s" % type(exc).__name__,
                      file=sys.stderr, flush=True)
            finally:
                first_weight_cycle.set()
            if once:
                break
            stop.wait(interval)

    weights = threading.Thread(target=weight_loop, name="arena-weight-loop", daemon=False)
    weights.start()
    runner = None
    try:
        while not stop.is_set():
            try:
                if runner is None:
                    runner = runner_factory()
                taken = runner.run_once(stop_event=stop)
            except Exception as exc:
                # No exception text: provider errors can contain credential-bearing URLs.
                print("Arena validator scoring cycle failed: type=%s; weights continue"
                      % type(exc).__name__, file=sys.stderr, flush=True)
                if runner is not None:
                    try:
                        runner.close()
                    except Exception as close_exc:
                        print("Arena validator scoring cleanup failed: type=%s"
                              % type(close_exc).__name__, file=sys.stderr, flush=True)
                    runner = None
                taken = 0
            if once:
                # A fast empty scoring queue must not cancel the first weight cycle.
                first_weight_cycle.wait()
                break
            if taken == 0:
                stop.wait(interval)
    finally:
        stop.set()
        if runner is not None:
            try:
                runner.close()
            except Exception as exc:
                print("Arena validator scoring cleanup failed: type=%s"
                      % type(exc).__name__, file=sys.stderr, flush=True)
        # Do not close the chain underneath an in-flight journal/finalization check.
        weights.join()


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    endpoint = _required_environment("LAB_ARENA_CHAIN_ENDPOINT")
    key_hash = _required_environment("LAB_ARENA_SIGNING_KEY_HASH")
    if not args.api_base_url:
        raise ArenaValidatorError("Arena API URL is required")
    from lab_arena import chain as chain_module
    from lab_arena.local_weight_signer import build_local_weight_signer
    from leadpoet_canonical.lab_arena_rewards import signing_key_from_document

    keypair = load_local_hotkey(args)
    config = chain_module.ArenaChainConfig(
        endpoint=endpoint, netuid=int(args.netuid),
        network_name=str(args.subtensor_network),
        request_timeout_seconds=int(os.environ.get("LAB_ARENA_CHAIN_TIMEOUT_SECONDS", "30")),
    )
    chain = chain_module.ArenaChain(config, chain_module.connect_substrate(config))
    signer = None
    try:
        public_api = ArenaPublicApi(args.api_base_url)
        signing_key = public_api.signing_key()
        signing_key_from_document(signing_key, key_hash)
        cutover = chain_module.load_arena_cutover()
        burn_hotkey = os.environ.get("LAB_ARENA_BURN_HOTKEY", "").strip()
        if not burn_hotkey and config.network_name == "finney":
            burn_hotkey = "5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"
        if not burn_hotkey:
            raise ArenaValidatorError("LAB_ARENA_BURN_HOTKEY is required on this network")
        signer = build_local_weight_signer(
            keypair=keypair, chain_config=config, signing_key_document=signing_key,
            expected_signing_key_hash=key_hash, cutover=cutover, burn_hotkey=burn_hotkey,
        )
        snapshot = chain_module.finalized_epoch_snapshot(chain)
        snapshot.settlement_epoch_id(cutover)  # Proves configured genesis and subnet.
        metagraph = chain.refresh_metagraph()
        if keypair.ss58_address not in metagraph.hotkeys:
            raise ArenaValidatorError("validator hotkey is not in the finalized metagraph")
        if args.check_only:
            signer.readiness(int(snapshot.settlement_epoch_id(cutover)))
            print("Arena local-wallet validator readiness is valid: hotkey=%s"
                  % keypair.ss58_address, flush=True)
            return 0

        from lab_arena.wiring import build_runner_from_environment

        orchestrator = ArenaWeightOrchestrator(
            api=public_api, chain=chain, signer=signer,
            validator_hotkey=keypair.ss58_address,
            expected_signing_key_hash=key_hash,
            paths=ArenaWeightPaths(Path(os.environ.get(
                "LAB_ARENA_VALIDATOR_STATE_DIR", "/var/lib/leadpoet/arena-validator"))),
            extrinsic_period=signer.extrinsic_period,
        )
        stop = threading.Event()
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda _signum, _frame: stop.set())
        run_validator_loops(
            orchestrator=orchestrator,
            runner_factory=lambda: build_runner_from_environment(args, keypair=keypair),
            epoch_supplier=lambda: chain_module.current_settlement_epoch(chain, cutover),
            stop=stop, poll_seconds=args.poll_seconds, once=args.once,
        )
        return 0
    finally:
        if signer is not None:
            signer.close()
        chain.close()


if __name__ == "__main__":
    raise SystemExit(main())
