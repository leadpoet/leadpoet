"""Protected signer for the small normal-validator Arena weight path."""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Mapping

from leadpoet_canonical.arena_weights import (
    ArenaWeightError,
    derive_arena_weights,
    verify_accepted_weight_state_signature,
)
from leadpoet_canonical.attested_v2 import sha256_bytes, sha256_json
from leadpoet_canonical.hotkey_authority_v2 import (
    encode_commit_timelocked_call,
    encode_signed_extrinsic_v2,
    encode_weight_signature_payload,
    mortal_era_bounds,
    select_chain_signing_profile,
    signed_extrinsic_hash_v2,
)

ARENA_RECOVERY_PREFIX = "LEADPOET_ARENA_WEIGHT_RECOVERY_V1|"


class ArenaWeightSignerError(RuntimeError):
    """A requested Arena chain signature is not authorized."""


def _without_hex_prefix(value: Any) -> str:
    text = str(value or "").lower()
    return text[2:] if text.startswith("0x") else text


class ArenaWeightSigner:
    """Verify accepted state and finalized ownership before signing one call."""

    def __init__(
        self, *, validator_hotkey: str, hotkey_public_key_hex: str,
        chain_profile: Mapping[str, Any], chain_source: Any, drand_backend: Any,
        sign_sr25519: Callable[[bytes], bytes], arena_public_key_der: bytes,
        verify_sr25519: Callable[[bytes, bytes], bool], arena_public_key_hash: str,
        network: str, netuid: int, burn_hotkey: str,
        state_source: Any = None,
    ) -> None:
        self.validator_hotkey = str(validator_hotkey)
        self.hotkey_public_key_hex = str(hotkey_public_key_hex)
        self.chain_profile = dict(chain_profile)
        self._chain_source = chain_source
        self._drand = drand_backend
        self._sign = sign_sr25519
        self._verify = verify_sr25519
        self._arena_public_key_der = bytes(arena_public_key_der)
        self._arena_public_key_hash = str(arena_public_key_hash)
        self._network = str(network)
        self._netuid = int(netuid)
        self._burn_hotkey = str(burn_hotkey)
        self._state_source = state_source
        self._epoch_states = {}  # type: Dict[str, str]
        self._signed = {}  # type: Dict[str, Dict[str, Any]]
        self._records = {}  # type: Dict[str, Dict[str, Any]]
        self._attempts = {}  # type: Dict[str, Dict[str, Any]]
        self._confirmed = {}  # type: Dict[str, Dict[str, Any]]
        self._lock = threading.Lock()
        self._prepare_lock = threading.Lock()

    def prepare(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        with self._prepare_lock:
            return self._prepare_locked(request)

    def _prepare_locked(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        expected = {"accepted_state", "nonce", "era_current", "runtime_block_hash", "block_hash"}
        if not isinstance(request, Mapping) or set(request) != expected:
            raise ArenaWeightSignerError("Arena weight signature request fields are invalid")
        state = request["accepted_state"]
        try:
            state_hash = verify_accepted_weight_state_signature(
                state, public_key_der=self._arena_public_key_der,
                expected_public_key_hash=self._arena_public_key_hash,
            )
        except ArenaWeightError as exc:
            raise ArenaWeightSignerError("accepted Arena state is unauthorized") from exc
        if self._state_source is not None:
            authoritative = self._state_source.read(epoch=int(state["epoch"]))
            try:
                authoritative_hash = verify_accepted_weight_state_signature(
                    authoritative, public_key_der=self._arena_public_key_der,
                    expected_public_key_hash=self._arena_public_key_hash,
                )
            except ArenaWeightError as exc:
                raise ArenaWeightSignerError("authoritative Arena state is unauthorized") from exc
            if authoritative_hash != state_hash or dict(authoritative) != dict(state):
                raise ArenaWeightSignerError("host Arena state differs from authenticated gateway state")
        if state["network"] != self._network or int(state["netuid"]) != self._netuid:
            raise ArenaWeightSignerError("accepted Arena state targets another configured subnet")
        if state["burn_hotkey"] != self._burn_hotkey:
            raise ArenaWeightSignerError("accepted Arena state burn hotkey differs from sealed policy")
        epoch_key = "%s:%s:%s" % (state["network"], state["netuid"], state["epoch"])
        with self._lock:
            prior = self._epoch_states.get(epoch_key)
            if prior is not None and prior != state_hash:
                raise ArenaWeightSignerError("conflicting accepted Arena state for epoch")
            self._epoch_states[epoch_key] = state_hash
            existing_record = self._records.get(state_hash)
            if existing_record is not None:
                return dict(existing_record["result"])
            prior_sequences = [
                int(item["result"]["attempt_sequence"])
                for item in self._attempts.values()
                if item["result"]["state_hash"] == state_hash
            ]
            attempt_sequence = 1 + max(prior_sequences or [0])
        try:
            snapshot = self._chain_source.read_finalized_snapshot(
                netuid=int(state["netuid"]), epoch_id=int(state["epoch"])
            )
            block = int(snapshot["header"]["block"])
            if not int(state["valid_from_block"]) <= block <= int(state["valid_until_block"]):
                raise ArenaWeightSignerError("accepted Arena state is stale at finalized head")
            if str(state["genesis_hash"]).lower() != str(self.chain_profile["genesis_hash"]).lower():
                raise ArenaWeightSignerError("accepted Arena state targets another genesis")
            epoch_authority = snapshot["epoch_authority"]
            settlement_epoch = int(epoch_authority.get("settlement_epoch_id", epoch_authority.get("epoch_id", -1)))
            if settlement_epoch != int(state["epoch"]):
                raise ArenaWeightSignerError("finalized chain epoch differs from accepted state")
            derived = derive_arena_weights(state, snapshot["metagraph"]["hotkeys"])
            runtime = self._chain_source.read_chain_signing_runtime(
                runtime_block_hash=str(request["runtime_block_hash"]),
                max_block_drift=int(self.chain_profile["max_snapshot_block_drift"]),
            )
            if not int(state["valid_from_block"]) <= int(runtime["finalized_block"]) <= int(state["valid_until_block"]):
                raise ArenaWeightSignerError("accepted Arena state is stale at signing finalized head")
            if int(request["era_current"]) != int(runtime["runtime_block"]):
                raise ArenaWeightSignerError("era current differs from authenticated runtime block")
            birth, death = mortal_era_bounds(
                period=int(self.chain_profile["extrinsic_period"]),
                current=int(request["era_current"]),
            )
            if int(death) - 1 > int(state["valid_until_block"]):
                raise ArenaWeightSignerError(
                    "mortal era extends beyond accepted Arena state validity"
                )
            authenticated_next_epoch = int(epoch_authority.get(
                "next_epoch_block",
                int(epoch_authority["last_epoch_block"]) + int(epoch_authority["tempo"]),
            ))
            if int(runtime["finalized_block"]) >= authenticated_next_epoch:
                raise ArenaWeightSignerError(
                    "signing finalized head is outside authenticated subnet epoch"
                )
            if not birth <= int(runtime["finalized_block"]) < death:
                raise ArenaWeightSignerError("mortal era is not live at finalized head")
            mortality_hash = self._chain_source.read_canonical_block_hash(block=birth)
            if _without_hex_prefix(request["block_hash"]) != mortality_hash:
                raise ArenaWeightSignerError("mortality block hash differs from canonical era birth")
            finalized_nonce = self._chain_source.read_finalized_account_nonce(
                account_public_key_hex=self.hotkey_public_key_hex,
                finalized_block_hash=runtime["finalized_block_hash"],
            )
            if int(request["nonce"]) != finalized_nonce:
                raise ArenaWeightSignerError("nonce differs from finalized account state")
            profile = select_chain_signing_profile(
                self.chain_profile,
                runtime_version={"specVersion": runtime["spec_version"], "transactionVersion": runtime["transaction_version"]},
                genesis_hash=runtime["genesis_hash"],
            )
            commitment, reveal_round = self._drand.generate_commit(
                uids=derived["sparse_uids"], weights_u16=derived["sparse_weights_u16"],
                version_key=int(profile["version_key"]),
                last_epoch_block=int(epoch_authority["last_epoch_block"]),
                pending_epoch_at=int(epoch_authority["pending_epoch_at"]),
                subnet_epoch_index=int(epoch_authority["subnet_epoch_index"]),
                tempo=int(epoch_authority["tempo"]),
                blocks_since_last_step=int(epoch_authority["blocks_since_last_step"]),
                current_block=int(epoch_authority["current_block"]),
                subnet_reveal_period_epochs=int(profile["subnet_reveal_period_epochs"]),
                block_time=float(profile["block_time_millis"]) / 1000.0,
                hotkey_public_key=bytes.fromhex(self.hotkey_public_key_hex),
            )
            call = encode_commit_timelocked_call(
                profile=profile, netuid=int(state["netuid"]),
                commitment=commitment, reveal_round=int(reveal_round),
            )
            preimage, signed_message = encode_weight_signature_payload(
                profile=profile, call_bytes=call,
                era_current=int(request["era_current"]), nonce=int(request["nonce"]),
                block_hash=str(request["block_hash"]),
            )
            authorization_body = {
                "schema_version": "leadpoet.arena.weight_extrinsic_authorization.v1",
                "state_hash": state_hash, "validator_hotkey": self.validator_hotkey,
                "attempt_sequence": attempt_sequence,
                "netuid": int(state["netuid"]), "epoch": int(state["epoch"]),
                "finalized_block": block,
                "finalized_block_hash": str(snapshot["finalized_block_hash"]),
                "weights_hash": derived["weights_hash"],
                "sparse_uids": derived["sparse_uids"],
                "sparse_weights_u16": derived["sparse_weights_u16"],
                "recipient_uid_hotkeys": [
                    {"uid": int(uid), "hotkey": str(snapshot["metagraph"]["hotkeys"][uid])}
                    for uid in derived["sparse_uids"]
                ],
                "commitment_hash": sha256_bytes(bytes(commitment)),
                "reveal_round": int(reveal_round), "call_data_hash": sha256_bytes(call),
                "runtime_spec_version": int(profile["spec_version"]),
                "era_current": int(request["era_current"]),
                "era_period": int(profile["extrinsic_period"]), "nonce": int(request["nonce"]),
                "block_hash": _without_hex_prefix(request["block_hash"]),
                "signature_payload_preimage_hash": sha256_bytes(preimage),
                "signed_message_hash": sha256_bytes(signed_message),
            }
            authorization_hash = sha256_json(authorization_body)
            request_key = sha256_json({"authorization_hash": authorization_hash})
            with self._lock:
                cached = self._signed.get(request_key)
                if cached is not None:
                    return dict(cached)
            signature = bytes(self._sign(signed_message))
            if len(signature) != 64:
                raise ArenaWeightSignerError("protected sr25519 signer returned invalid signature")
            extrinsic = encode_signed_extrinsic_v2(
                hotkey_public_key_hex=self.hotkey_public_key_hex,
                signature_hex=signature.hex(), era_period=int(profile["extrinsic_period"]),
                era_current=int(request["era_current"]), nonce=int(request["nonce"]),
                call_data_hex=call.hex(),
            )
        except ArenaWeightSignerError:
            raise
        except Exception as exc:
            raise ArenaWeightSignerError("Arena weight transaction authorization failed") from exc
        result = dict(authorization_body)
        result.update({
            "schema_version": "leadpoet.arena.weight_extrinsic.v1",
            "authorization_hash": authorization_hash,
            "extrinsic_hash": signed_extrinsic_hash_v2(extrinsic),
            "extrinsic_hex": extrinsic.hex(),
        })
        recovery_body = {
            "authorization": authorization_body,
            "authorization_hash": authorization_hash,
            "signature_hex": signature.hex(), "call_data_hex": call.hex(),
            "commitment_hex": bytes(commitment).hex(),
            "extrinsic_hex": extrinsic.hex(), "extrinsic_hash": result["extrinsic_hash"],
            "minimum_block": int(request["era_current"]),
            "maximum_block": min(int(state["valid_until_block"]), int(death) - 1),
            "subnet_epoch_index": int(epoch_authority["subnet_epoch_index"]),
            "reveal_deadline_block": int(epoch_authority.get(
                "next_epoch_block",
                int(epoch_authority["last_epoch_block"]) + int(epoch_authority["tempo"]),
            ))
            + int(profile["subnet_reveal_period_epochs"]) * int(epoch_authority["tempo"]),
        }
        recovery_signature = bytes(self._sign(
            (ARENA_RECOVERY_PREFIX + sha256_json(recovery_body)).encode("utf-8")
        ))
        if len(recovery_signature) != 64:
            raise ArenaWeightSignerError("protected recovery signer returned invalid signature")
        result["recovery_record"] = {
            **recovery_body, "record_signature_hex": recovery_signature.hex()
        }
        with self._lock:
            existing = self._signed.get(request_key)
            if existing is not None:
                return dict(existing)
            self._signed[request_key] = dict(result)
            self._records[state_hash] = {
                "result": dict(result), "extrinsic_hex": extrinsic.hex(),
                "commitment": bytes(commitment), "reveal_round": int(reveal_round),
                "subnet_epoch_index": int(epoch_authority["subnet_epoch_index"]),
                "nonce": int(request["nonce"]), "minimum_block": int(request["era_current"]),
                "maximum_block": min(int(state["valid_until_block"]), int(death) - 1),
                "reveal_deadline_block": recovery_body["reveal_deadline_block"],
            }
            self._attempts[result["extrinsic_hash"]] = self._records[state_hash]
        return result

    def recover(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(request, Mapping) or set(request) != {"accepted_state", "recovery_record"}:
            raise ArenaWeightSignerError("Arena recovery request fields are invalid")
        try:
            state_hash = verify_accepted_weight_state_signature(
                request["accepted_state"], public_key_der=self._arena_public_key_der,
                expected_public_key_hash=self._arena_public_key_hash,
            )
            recovery = request["recovery_record"]
            expected = {"authorization", "authorization_hash", "signature_hex", "call_data_hex", "commitment_hex", "extrinsic_hex", "extrinsic_hash", "minimum_block", "maximum_block", "subnet_epoch_index", "reveal_deadline_block", "record_signature_hex"}
            if not isinstance(recovery, Mapping) or set(recovery) != expected:
                raise ArenaWeightSignerError("Arena recovery record fields are invalid")
            recovery_body = {key: recovery[key] for key in recovery if key != "record_signature_hex"}
            record_signature = bytes.fromhex(str(recovery["record_signature_hex"]))
            if not self._verify(
                record_signature,
                (ARENA_RECOVERY_PREFIX + sha256_json(recovery_body)).encode("utf-8"),
            ):
                raise ArenaWeightSignerError("Arena recovery record signature is invalid")
            authorization = recovery["authorization"]
            if not isinstance(authorization, Mapping) or sha256_json(dict(authorization)) != recovery["authorization_hash"]:
                raise ArenaWeightSignerError("Arena recovery authorization hash differs")
            if authorization.get("state_hash") != state_hash or authorization.get("netuid") != self._netuid:
                raise ArenaWeightSignerError("Arena recovery authorization targets another state")
            profile = select_chain_signing_profile(
                self.chain_profile,
                runtime_version={"specVersion": authorization["runtime_spec_version"], "transactionVersion": self.chain_profile["transaction_version"]},
                genesis_hash=self.chain_profile["genesis_hash"],
            )
            call = bytes.fromhex(str(recovery["call_data_hex"]))
            preimage, signed_message = encode_weight_signature_payload(
                profile=profile, call_bytes=call,
                era_current=int(authorization["era_current"]), nonce=int(authorization["nonce"]),
                block_hash=str(authorization["block_hash"]),
            )
            if sha256_bytes(preimage) != authorization["signature_payload_preimage_hash"] or sha256_bytes(signed_message) != authorization["signed_message_hash"]:
                raise ArenaWeightSignerError("Arena recovery signature payload differs")
            signature = bytes.fromhex(str(recovery["signature_hex"]))
            if not self._verify(signature, signed_message):
                raise ArenaWeightSignerError("Arena recovery signature is invalid")
            extrinsic = encode_signed_extrinsic_v2(
                hotkey_public_key_hex=self.hotkey_public_key_hex,
                signature_hex=signature.hex(), era_period=int(authorization["era_period"]),
                era_current=int(authorization["era_current"]), nonce=int(authorization["nonce"]),
                call_data_hex=call.hex(),
            )
            if extrinsic.hex() != recovery["extrinsic_hex"] or signed_extrinsic_hash_v2(extrinsic) != recovery["extrinsic_hash"]:
                raise ArenaWeightSignerError("Arena recovery extrinsic differs")
        except ArenaWeightSignerError:
            raise
        except Exception as exc:
            raise ArenaWeightSignerError("Arena signed transaction recovery failed") from exc
        result = dict(authorization)
        result.update({
            "schema_version": "leadpoet.arena.weight_extrinsic.v1",
            "authorization_hash": recovery["authorization_hash"],
            "extrinsic_hash": recovery["extrinsic_hash"], "extrinsic_hex": recovery["extrinsic_hex"],
            "recovery_record": dict(recovery),
        })
        with self._lock:
            epoch_key = "%s:%s:%s" % (
                self._network, self._netuid, authorization["epoch"]
            )
            prior = self._epoch_states.get(epoch_key)
            if prior is not None and prior != state_hash:
                raise ArenaWeightSignerError(
                    "conflicting recovered Arena state for epoch"
                )
            self._records[state_hash] = {
                "result": result, "extrinsic_hex": recovery["extrinsic_hex"],
                "commitment": bytes.fromhex(str(recovery["commitment_hex"])),
                "reveal_round": int(authorization["reveal_round"]),
                "subnet_epoch_index": int(recovery["subnet_epoch_index"]),
                "nonce": int(authorization["nonce"]),
                "minimum_block": int(recovery["minimum_block"]),
                "maximum_block": int(recovery["maximum_block"]),
                "reveal_deadline_block": int(recovery["reveal_deadline_block"]),
            }
            self._attempts[result["extrinsic_hash"]] = self._records[state_hash]
            self._epoch_states[epoch_key] = state_hash
        return result

    def confirm(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        expected = {"state_hash", "extrinsic_hash"}
        if not isinstance(request, Mapping) or set(request) != expected:
            raise ArenaWeightSignerError("Arena confirmation request fields are invalid")
        with self._lock:
            record = self._attempts.get(str(request.get("extrinsic_hash") or ""))
            if record is None or record["result"]["state_hash"] != request.get("state_hash"):
                raise ArenaWeightSignerError("Arena signed transaction was not found")
            record = dict(record)
        result = record["result"]
        try:
            found = self._chain_source.find_finalized_extrinsic_inclusion(
                expected_extrinsics={result["extrinsic_hash"]: record["extrinsic_hex"]},
                expected_commitments={result["extrinsic_hash"]: {
                    "netuid": result["netuid"],
                    "subnet_epoch_index": record["subnet_epoch_index"],
                    "hotkey_public_key": self.hotkey_public_key_hex,
                    "commitment_hex": record["commitment"].hex(),
                    "reveal_round": record["reveal_round"],
                }},
                minimum_block=record["minimum_block"], maximum_block=record["maximum_block"],
                epoch_id=result["epoch"],
                finalization_scan_id=sha256_json({"state_hash": result["state_hash"], "extrinsic_hash": result["extrinsic_hash"]}),
            )
        except Exception as scan_error:
            from validator_tee.enclave.chain_source_v2 import ValidatorChainSourceV2Error
            if not isinstance(scan_error, ValidatorChainSourceV2Error):
                raise
            if str(scan_error) == "authorized extrinsic range is not finalized":
                head = self._chain_source.read_finalized_head()
                return {
                    "schema_version": "leadpoet.arena.chain_outcome.v1", "status": "pending",
                    "finalized": False, "state_hash": result["state_hash"],
                    "extrinsic_hash": result["extrinsic_hash"], "finalized_head": head,
                }
            if str(scan_error) != "no authorized weight extrinsic was found in finalized blocks":
                raise
            head = self._chain_source.read_finalized_head()
            if int(head["block"]) <= int(record["maximum_block"]):
                return {
                    "schema_version": "leadpoet.arena.chain_outcome.v1", "status": "pending",
                    "finalized": False, "state_hash": result["state_hash"],
                    "extrinsic_hash": result["extrinsic_hash"], "finalized_head": head,
                }
            nonce = self._chain_source.read_finalized_account_nonce(
                account_public_key_hex=self.hotkey_public_key_hex,
                finalized_block_hash=head["block_hash"],
            )
            if nonce != int(record["nonce"]):
                raise ArenaWeightSignerError(
                    "expired Arena transaction has ambiguous replacement or nonce state"
                ) from scan_error
            with self._lock:
                active = self._records.get(result["state_hash"])
                if (active is not None and active["result"]["extrinsic_hash"]
                        == result["extrinsic_hash"]):
                    self._records.pop(result["state_hash"], None)
            return {
                "schema_version": "leadpoet.arena.chain_outcome.v1",
                "status": "not_included_expired", "finalized": False,
                "state_hash": result["state_hash"], "extrinsic_hash": result["extrinsic_hash"],
                "finalized_head": head, "finalized_nonce": nonce,
                "attempt_sequence": result["attempt_sequence"],
            }
        expected_weights = list(zip(result["sparse_uids"], result["sparse_weights_u16"]))
        reveal = self._chain_source.prove_timelocked_reveal_transition(
            netuid=int(result["netuid"]), validator_hotkey=self.validator_hotkey,
            hotkey_public_key_hex=self.hotkey_public_key_hex,
            subnet_epoch_index=int(record["subnet_epoch_index"]),
            commitment_hex=record["commitment"].hex(),
            reveal_round=int(record["reveal_round"]),
            inclusion_block=int(found["finalized_block"]),
            reveal_deadline_block=int(record["reveal_deadline_block"]),
            expected_weights=expected_weights,
            expected_recipient_uid_hotkeys=result["recipient_uid_hotkeys"],
            chain_profile=self.chain_profile,
        )
        if reveal is None:
            reveal_head = self._chain_source.read_finalized_head()
            if int(reveal_head["block"]) > int(record["reveal_deadline_block"]):
                raise ArenaWeightSignerError("Arena reveal was not finalized before its deadline")
            return {
                "schema_version": "leadpoet.arena.chain_outcome.v1", "status": "included_pending_reveal",
                "finalized": False,
                "state_hash": result["state_hash"], "extrinsic_hash": result["extrinsic_hash"],
                "included_block": found["finalized_block"],
                "included_block_hash": found["finalized_block_hash"],
                "finalized_head": reveal_head,
            }
        outcome = {
            "schema_version": "leadpoet.arena.chain_outcome.v1", "status": "finalized",
            "finalized": True, "state_hash": result["state_hash"],
            "extrinsic_hash": result["extrinsic_hash"],
            "finalized_block": reveal["reveal_block"],
            "finalized_block_hash": reveal["reveal_block_hash"],
            "state_transition_hash": reveal["transition_hash"],
            "commit_included_block": found["finalized_block"],
            "commit_included_block_hash": found["finalized_block_hash"],
            "commit_state_transition_hash": found["state_transition_hash"],
            "weights_hash": result["weights_hash"],
            "validator_uid": reveal["validator_uid"],
            "last_update": reveal["last_update"],
            "revealed_weights": [list(item) for item in reveal["weights"]],
            "reveal_block": reveal["reveal_block"],
            "reveal_block_hash": reveal["reveal_block_hash"],
            "reveal_transition_hash": reveal["transition_hash"],
        }
        with self._lock:
            self._confirmed[result["state_hash"]] = dict(outcome)
        return outcome

    def sign_chain_outcome(self, document: Mapping[str, Any]) -> Dict[str, Any]:
        fields = {
            "schema_version", "network", "netuid", "epoch", "validator_hotkey",
            "state_hash", "weights_hash", "extrinsic_hash", "finalized_block_hash",
            "finalized_block_number", "observed_at", "request_id",
        }
        if not isinstance(document, Mapping) or set(document) != fields:
            raise ArenaWeightSignerError("Arena chain outcome document fields are invalid")
        body = {key: document[key] for key in document if key != "request_id"}
        if document.get("schema_version") != "leadpoet.arena.chain_outcome.v1" or document.get("request_id") != sha256_json(body):
            raise ArenaWeightSignerError("Arena chain outcome request hash is invalid")
        if document.get("validator_hotkey") != self.validator_hotkey or document.get("network") != self._network or int(document.get("netuid", -1)) != self._netuid:
            raise ArenaWeightSignerError("Arena chain outcome identity is invalid")
        with self._lock:
            record = self._records.get(str(document.get("state_hash") or ""))
            confirmed = self._confirmed.get(str(document.get("state_hash") or ""))
        if record is None or confirmed is None:
            raise ArenaWeightSignerError("Arena chain outcome has no protected finalization")
        result = record["result"]
        checks = (
            document.get("epoch") == result["epoch"],
            document.get("weights_hash") == result["weights_hash"],
            document.get("extrinsic_hash") == result["extrinsic_hash"],
            document.get("finalized_block_hash") == confirmed["finalized_block_hash"],
            document.get("finalized_block_number") == confirmed["finalized_block"],
        )
        if not all(checks):
            raise ArenaWeightSignerError("Arena chain outcome differs from protected finalization")
        message = ("arena_chain_outcome:" + str(document["request_id"])).encode("utf-8")
        signature = bytes(self._sign(message))
        if len(signature) != 64:
            raise ArenaWeightSignerError("protected sr25519 signer returned invalid outcome signature")
        return {"signature": "0x" + signature.hex()}
