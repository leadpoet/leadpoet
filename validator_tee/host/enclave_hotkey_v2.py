"""Public-only Bittensor wallet backed by the validator V2 enclave.

The normal application-signature path is purpose constrained by the enclave.
The chain path is narrower: while the existing ``Subtensor.set_weights`` call
runs, this module replaces only the SDK's stateful timelocked-commit generation
and extrinsic assembly. The enclave independently reconstructs the exact SCALE
payload before releasing one sr25519 signature.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
import importlib
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional

from bittensor_wallet import Keypair, Wallet

from leadpoet_canonical.hotkey_authority_v2 import signed_extrinsic_hash_v2
from validator_tee.host.vsock_client import ValidatorEnclaveClient


class EnclaveHotkeyV2Error(RuntimeError):
    """The public-only wallet or set-weights authorization is invalid."""


_CHAIN_SIGNING_PATCH_LOCK = threading.RLock()
_ACTIVE_CHAIN_SIGNING_CONTEXT = None


def _weight_extrinsic_module() -> Any:
    """Resolve the module whose stateful drand helper Bittensor 10 calls."""

    errors = []
    for module_name in (
        "bittensor.core.extrinsics.weights",
        "bittensor.core.extrinsics.mechanism",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(exc)
            continue
        if hasattr(module, "get_encrypted_commit_v2"):
            return module
    raise EnclaveHotkeyV2Error(
        "installed Bittensor SDK has no stateful timelocked-weight helper; "
        "Bittensor 10.5.0 or newer is required"
    ) from (errors[-1] if errors else None)


def _message_bytes(value: Any) -> bytes:
    if hasattr(value, "data"):
        value = value.data
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        if value.startswith("0x"):
            try:
                return bytes.fromhex(value[2:])
            except ValueError as exc:
                raise EnclaveHotkeyV2Error("signature input is invalid hex") from exc
        return value.encode("utf-8")
    try:
        return bytes(value)
    except Exception as exc:
        raise EnclaveHotkeyV2Error("signature input is not byte-compatible") from exc


class EnclaveBackedKeypairV2(Keypair):
    """A public sr25519 keypair whose only signing operation is an enclave RPC."""

    def __new__(
        cls,
        *,
        ss58_address: str,
        client: Optional[ValidatorEnclaveClient] = None,
    ):
        return super().__new__(cls, ss58_address=str(ss58_address))

    def __init__(
        self,
        *,
        ss58_address: str,
        client: Optional[ValidatorEnclaveClient] = None,
    ) -> None:
        self._enclave_client_v2 = client or ValidatorEnclaveClient()
        self._last_application_receipt_v2 = None

    @property
    def enclave_client_v2(self) -> ValidatorEnclaveClient:
        return self._enclave_client_v2

    @property
    def last_application_receipt_v2(self) -> Optional[Dict[str, Any]]:
        value = self._last_application_receipt_v2
        return dict(value) if isinstance(value, Mapping) else None

    def sign(self, data: Any) -> bytes:
        """Sign only an enclave-recognized application message.

        SCALE weight payloads never use this method. They must pass through
        :class:`AuthoritativeSetWeightsContextV2`, which supplies the receipt
        and authoritative gateway publication bindings the enclave requires.
        """

        message = _message_bytes(data)
        result = self._enclave_client_v2.sign_application_message_v2(message)
        try:
            signature = bytes.fromhex(str(result["signature"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise EnclaveHotkeyV2Error(
                "enclave application signature response is invalid"
            ) from exc
        if len(signature) != 64:
            raise EnclaveHotkeyV2Error(
                "enclave application signature has the wrong length"
            )
        receipt = result.get("receipt")
        if not isinstance(receipt, Mapping):
            raise EnclaveHotkeyV2Error(
                "enclave application signature receipt is missing"
            )
        self._last_application_receipt_v2 = dict(receipt)
        return signature


class EnclaveBackedWalletV2(Wallet):
    """A Bittensor ``Wallet`` subtype with no usable host hotkey secret."""

    def __new__(
        cls,
        *,
        name: str,
        hotkey: str,
        path: str,
        keypair: EnclaveBackedKeypairV2,
    ):
        return super().__new__(cls, name=name, hotkey=hotkey, path=path)

    def __init__(
        self,
        *,
        name: str,
        hotkey: str,
        path: str,
        keypair: EnclaveBackedKeypairV2,
    ) -> None:
        self._enclave_hotkey_v2 = keypair

    @property
    def hotkey(self) -> EnclaveBackedKeypairV2:
        return self._enclave_hotkey_v2

    @property
    def hotkeypub(self) -> EnclaveBackedKeypairV2:
        return self._enclave_hotkey_v2

    def get_hotkey(self) -> EnclaveBackedKeypairV2:
        return self._enclave_hotkey_v2

    def get_hotkeypub(self) -> EnclaveBackedKeypairV2:
        return self._enclave_hotkey_v2

    def unlock_hotkey(self) -> None:
        # The key was unsealed in Nitro during startup. There is no host file to
        # decrypt and no secret-bearing object to return.
        return None

    def set_hotkey(self, *args: Any, **kwargs: Any) -> None:
        raise EnclaveHotkeyV2Error("host hotkey mutation is disabled in V2")

    def set_hotkeypub(self, *args: Any, **kwargs: Any) -> None:
        raise EnclaveHotkeyV2Error("host hotkey mutation is disabled in V2")


def build_enclave_backed_wallet_v2(
    *,
    name: str,
    hotkey_name: str,
    path: str,
    client: Optional[ValidatorEnclaveClient] = None,
) -> EnclaveBackedWalletV2:
    enclave_client = client or ValidatorEnclaveClient()
    state = enclave_client.get_hotkey_state_v2()
    if state.get("provisioned") is not True:
        raise EnclaveHotkeyV2Error("validator hotkey is not provisioned in Nitro")
    address = str(state.get("validator_hotkey") or "")
    public_key = str(state.get("hotkey_public_key") or "").lower()
    keypair = EnclaveBackedKeypairV2(
        ss58_address=address,
        client=enclave_client,
    )
    if keypair.public_key.hex() != public_key:
        raise EnclaveHotkeyV2Error(
            "validator hotkey address and measured public key differ"
        )
    return EnclaveBackedWalletV2(
        name=str(name),
        hotkey=str(hotkey_name),
        path=str(path),
        keypair=keypair,
    )


class AuthoritativeServeAxonContextV2(AbstractContextManager):
    """Constrain startup signing to one exact ``serve_axon`` call shape."""

    period = 8

    def __init__(self, *, substrate: Any, wallet: EnclaveBackedWalletV2) -> None:
        if not isinstance(wallet, EnclaveBackedWalletV2):
            raise EnclaveHotkeyV2Error("serve_axon requires a public-only V2 wallet")
        self.substrate = substrate
        self.wallet = wallet
        self.client = wallet.hotkey.enclave_client_v2
        self._original_create_signed_extrinsic = None
        self.extrinsic_signature_results = []  # type: List[Dict[str, Any]]

    def __enter__(self) -> "AuthoritativeServeAxonContextV2":
        global _ACTIVE_CHAIN_SIGNING_CONTEXT
        _CHAIN_SIGNING_PATCH_LOCK.acquire()
        try:
            if _ACTIVE_CHAIN_SIGNING_CONTEXT is not None:
                raise EnclaveHotkeyV2Error(
                    "another authoritative chain-signing context is active"
                )
            self._original_create_signed_extrinsic = (
                self.substrate.create_signed_extrinsic
            )
            self.substrate.create_signed_extrinsic = self._create_signed_extrinsic
            _ACTIVE_CHAIN_SIGNING_CONTEXT = self
            return self
        except Exception:
            _CHAIN_SIGNING_PATCH_LOCK.release()
            raise

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        global _ACTIVE_CHAIN_SIGNING_CONTEXT
        try:
            if self._original_create_signed_extrinsic is not None:
                self.substrate.create_signed_extrinsic = (
                    self._original_create_signed_extrinsic
                )
            _ACTIVE_CHAIN_SIGNING_CONTEXT = None
        finally:
            _CHAIN_SIGNING_PATCH_LOCK.release()

    def _create_signed_extrinsic(
        self,
        call: Any,
        keypair: Any,
        era: Any = None,
        nonce: Optional[int] = None,
        tip: int = 0,
        tip_asset_id: Optional[int] = None,
        signature: Any = None,
    ) -> Any:
        if keypair is not self.wallet.hotkey:
            raise EnclaveHotkeyV2Error("SDK requested an unexpected signing key")
        if signature is not None:
            raise EnclaveHotkeyV2Error("host-supplied chain signatures are forbidden")
        if tip != 0 or tip_asset_id is not None:
            raise EnclaveHotkeyV2Error("serve_axon tips are not authorized")
        value = getattr(call, "value", None)
        if not isinstance(value, Mapping) or set(value) != {
            "call_module",
            "call_function",
            "call_args",
        }:
            raise EnclaveHotkeyV2Error("serve_axon call metadata is invalid")
        args = value.get("call_args")
        required_args = {
            "version",
            "ip",
            "port",
            "ip_type",
            "netuid",
            "hotkey",
            "coldkey",
            "protocol",
            "placeholder1",
            "placeholder2",
        }
        if (
            value.get("call_module") != "SubtensorModule"
            or value.get("call_function") != "serve_axon"
            or not isinstance(args, Mapping)
            or set(args) != required_args
            or args.get("hotkey") != self.wallet.hotkey.ss58_address
        ):
            raise EnclaveHotkeyV2Error("SDK call is not the authorized serve_axon domain")
        self.substrate.init_runtime()
        actual_nonce = (
            self.substrate.get_account_nonce(keypair.ss58_address) or 0
            if nonce is None
            else int(nonce)
        )
        if not isinstance(era, dict):
            raise EnclaveHotkeyV2Error("serve_axon must use a mortal era")
        actual_era = dict(era)
        if "phase" in actual_era:
            raise EnclaveHotkeyV2Error("phase-only serve_axon eras are not authorized")
        if "current" not in actual_era:
            finalised_head = self.substrate.get_chain_finalised_head()
            actual_era["current"] = self.substrate.get_block_number(
                finalised_head
            )
        era_current = int(actual_era["current"])
        era_object = self.substrate.runtime_config.create_scale_object("Era")
        era_object.encode(actual_era)
        birth_block = era_object.birth(era_current)
        block_hash = str(self.substrate.get_block_hash(block_id=birth_block))
        if block_hash.startswith("0x"):
            block_hash = block_hash[2:]
        payload = _message_bytes(
            self.substrate.generate_signature_payload(
                call=call,
                era=actual_era,
                nonce=actual_nonce,
                tip=tip,
                tip_asset_id=tip_asset_id,
            )
        )
        result = self.client.sign_serve_axon_extrinsic_v2(
            {
                "netuid": int(args["netuid"]),
                "version": int(args["version"]),
                "ip": int(args["ip"]),
                "port": int(args["port"]),
                "ip_type": int(args["ip_type"]),
                "protocol": int(args["protocol"]),
                "placeholder1": int(args["placeholder1"]),
                "placeholder2": int(args["placeholder2"]),
                "era_current": era_current,
                "nonce": actual_nonce,
                "block_hash": block_hash,
                "signature_payload_hex": payload.hex(),
            }
        )
        try:
            enclave_signature = bytes.fromhex(str(result["signature"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise EnclaveHotkeyV2Error(
                "enclave serve_axon signature response is invalid"
            ) from exc
        authorization = result.get("authorization")
        if (
            len(enclave_signature) != 64
            or not isinstance(result.get("receipt"), Mapping)
            or not isinstance(authorization, Mapping)
            or authorization.get("signed_message_hex") != payload.hex()
            or authorization.get("call_data_hex")
            != _message_bytes(call.data).hex()
        ):
            raise EnclaveHotkeyV2Error(
                "enclave serve_axon signature response is incomplete"
            )
        self.extrinsic_signature_results.append(dict(result))
        return self._original_create_signed_extrinsic(
            call=call,
            keypair=keypair,
            era=actual_era,
            nonce=actual_nonce,
            tip=tip,
            tip_asset_id=tip_asset_id,
            signature=enclave_signature,
        )


class AuthoritativeSetWeightsContextV2(AbstractContextManager):
    """Constrain one unchanged ``Subtensor.set_weights`` invocation."""

    def __init__(
        self,
        *,
        substrate: Any,
        wallet: EnclaveBackedWalletV2,
        weight_authorization_id: str,
        weight_submission_event_hash: str,
        on_signed_extrinsic: Optional[
            Callable[[Mapping[str, Any]], None]
        ] = None,
    ) -> None:
        if not isinstance(wallet, EnclaveBackedWalletV2):
            raise EnclaveHotkeyV2Error("set_weights requires a public-only V2 wallet")
        self.substrate = substrate
        self.wallet = wallet
        self.weight_authorization_id = str(weight_authorization_id)
        self.weight_submission_event_hash = str(weight_submission_event_hash)
        self._on_signed_extrinsic = on_signed_extrinsic
        self.client = wallet.hotkey.enclave_client_v2
        self._weight_module = None
        self._original_get_encrypted_commit_v2 = None
        self._original_create_signed_extrinsic = None
        self._commit_queue = []  # type: List[Dict[str, Any]]
        self.extrinsic_signature_results = []  # type: List[Dict[str, Any]]

    def __enter__(self) -> "AuthoritativeSetWeightsContextV2":
        global _ACTIVE_CHAIN_SIGNING_CONTEXT
        _CHAIN_SIGNING_PATCH_LOCK.acquire()
        try:
            if _ACTIVE_CHAIN_SIGNING_CONTEXT is not None:
                raise EnclaveHotkeyV2Error(
                    "another authoritative chain-signing context is active"
                )
            self._weight_module = _weight_extrinsic_module()
            self._original_get_encrypted_commit_v2 = (
                self._weight_module.get_encrypted_commit_v2
            )
            self._original_create_signed_extrinsic = (
                self.substrate.create_signed_extrinsic
            )
            self._weight_module.get_encrypted_commit_v2 = (
                self._get_encrypted_commit_v2
            )
            self.substrate.create_signed_extrinsic = self._create_signed_extrinsic
            _ACTIVE_CHAIN_SIGNING_CONTEXT = self
            return self
        except Exception:
            _CHAIN_SIGNING_PATCH_LOCK.release()
            raise

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        global _ACTIVE_CHAIN_SIGNING_CONTEXT
        try:
            if (
                self._weight_module is not None
                and self._original_get_encrypted_commit_v2 is not None
            ):
                self._weight_module.get_encrypted_commit_v2 = (
                    self._original_get_encrypted_commit_v2
                )
            if self._original_create_signed_extrinsic is not None:
                self.substrate.create_signed_extrinsic = (
                    self._original_create_signed_extrinsic
                )
            _ACTIVE_CHAIN_SIGNING_CONTEXT = None
        finally:
            _CHAIN_SIGNING_PATCH_LOCK.release()

    def _get_encrypted_commit_v2(
        self,
        *,
        uids: List[int],
        weights: List[int],
        version_key: int,
        last_epoch_block: int,
        pending_epoch_at: int,
        subnet_epoch_index: int,
        tempo: int,
        blocks_since_last_step: int,
        current_block: int,
        subnet_reveal_period_epochs: int,
        block_time: float,
        hotkey: bytes,
    ):
        result = self.client.prepare_weight_commit_v2(
            {
                "weight_authorization_id": self.weight_authorization_id,
                "weight_submission_event_hash": self.weight_submission_event_hash,
                "uids": [int(item) for item in uids],
                "weights_u16": [int(item) for item in weights],
                "version_key": int(version_key),
                "last_epoch_block": int(last_epoch_block),
                "pending_epoch_at": int(pending_epoch_at),
                "subnet_epoch_index": int(subnet_epoch_index),
                "tempo": int(tempo),
                "blocks_since_last_step": int(blocks_since_last_step),
                "current_block": int(current_block),
                "subnet_reveal_period_epochs": int(
                    subnet_reveal_period_epochs
                ),
                "block_time": float(block_time),
                "hotkey_public_key_hex": bytes(hotkey).hex(),
            }
        )
        try:
            commitment = bytes.fromhex(str(result["commitment_hex"]))
            reveal_round = int(result["reveal_round"])
            commit_authorization_id = str(result["commit_authorization_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EnclaveHotkeyV2Error(
                "enclave commitment response is invalid"
            ) from exc
        if not commitment or not commit_authorization_id.startswith("sha256:"):
            raise EnclaveHotkeyV2Error("enclave commitment response is invalid")
        self._commit_queue.append(
            {"commit_authorization_id": commit_authorization_id}
        )
        return commitment, reveal_round

    def _create_signed_extrinsic(
        self,
        call: Any,
        keypair: Any,
        era: Any = None,
        nonce: Optional[int] = None,
        tip: int = 0,
        tip_asset_id: Optional[int] = None,
        signature: Any = None,
    ) -> Any:
        if keypair is not self.wallet.hotkey:
            raise EnclaveHotkeyV2Error("SDK requested an unexpected signing key")
        if signature is not None:
            raise EnclaveHotkeyV2Error("host-supplied chain signatures are forbidden")
        if tip != 0 or tip_asset_id is not None:
            raise EnclaveHotkeyV2Error("weight extrinsic tip settings are not authorized")
        if not self._commit_queue:
            raise EnclaveHotkeyV2Error(
                "weight extrinsic has no enclave commitment authorization"
            )
        self.substrate.init_runtime()
        actual_nonce = (
            self.substrate.get_account_nonce(keypair.ss58_address) or 0
            if nonce is None
            else int(nonce)
        )
        if not isinstance(era, dict):
            raise EnclaveHotkeyV2Error("weight extrinsic must use a mortal era")
        actual_era = dict(era)
        if "phase" in actual_era:
            raise EnclaveHotkeyV2Error("phase-only weight eras are not authorized")
        if "current" not in actual_era:
            finalised_head = self.substrate.get_chain_finalised_head()
            actual_era["current"] = self.substrate.get_block_number(
                finalised_head
            )
        era_current = int(actual_era["current"])
        era_object = self.substrate.runtime_config.create_scale_object("Era")
        era_object.encode(actual_era)
        birth_block = era_object.birth(era_current)
        block_hash = str(self.substrate.get_block_hash(block_id=birth_block))
        if block_hash.startswith("0x"):
            block_hash = block_hash[2:]
        payload = _message_bytes(
            self.substrate.generate_signature_payload(
                call=call,
                era=actual_era,
                nonce=actual_nonce,
                tip=tip,
                tip_asset_id=tip_asset_id,
            )
        )
        commit_record = self._commit_queue.pop(0)
        result = self.client.sign_weight_extrinsic_v2(
            {
                "commit_authorization_id": commit_record[
                    "commit_authorization_id"
                ],
                "era_current": era_current,
                "nonce": actual_nonce,
                "block_hash": block_hash,
                "signature_payload_hex": payload.hex(),
            }
        )
        try:
            enclave_signature = bytes.fromhex(str(result["signature"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise EnclaveHotkeyV2Error(
                "enclave extrinsic signature response is invalid"
            ) from exc
        if len(enclave_signature) != 64 or not isinstance(
            result.get("receipt"), Mapping
        ):
            raise EnclaveHotkeyV2Error(
                "enclave extrinsic signature response is incomplete"
            )
        extrinsic = self._original_create_signed_extrinsic(
            call=call,
            keypair=keypair,
            era=actual_era,
            nonce=actual_nonce,
            tip=tip,
            tip_asset_id=tip_asset_id,
            signature=enclave_signature,
        )
        observed_hash = signed_extrinsic_hash_v2(_message_bytes(extrinsic.data))
        if observed_hash != result.get("extrinsic_hash"):
            raise EnclaveHotkeyV2Error(
                "SDK extrinsic bytes differ from enclave reconstruction"
            )
        if self._on_signed_extrinsic is not None:
            # This callback must durably fsync before the signed extrinsic is
            # returned to the SDK and therefore before it can be broadcast.
            self._on_signed_extrinsic(dict(result))
        self.extrinsic_signature_results.append(dict(result))
        return extrinsic
