"""Authenticated replacement for qualification's shared block-file read."""

from __future__ import annotations

import base64
from typing import Any, Mapping

from Leadpoet.utils.subnet_epoch import (
    LEGACY_EPOCH_MODE,
    STATEFUL_EPOCH_MODE,
    SubnetEpochCutover,
    SubnetEpochError,
)
from gateway.tee.provider_client_v2 import BrokeredProviderTransportV2
from leadpoet_canonical.chain_source_v2 import (
    CHAIN_ARCHIVE_ENDPOINT_HOST,
    CHAIN_ENDPOINT_HOST,
    CHAIN_ENDPOINT_PATH,
    CHAIN_FINALIZATION_EPOCH_BLOCKS,
    CHAIN_RPC_TIMEOUT_MS,
    decode_subnet_epoch_storage,
    json_rpc_request,
    normalize_raw_hash,
    parse_finalized_header,
    parse_json_rpc_response,
    subnet_epoch_storage_key,
)


class QualificationEpochGuardV2Error(RuntimeError):
    """Current chain epoch could not be authenticated inside the job."""


class QualificationEpochGuardV2:
    """Match the legacy ``file_epoch > current_epoch`` decision using TLS RPC."""

    def __init__(
        self,
        transport: BrokeredProviderTransportV2,
        *,
        epoch_authority: Mapping[str, Any] | None = None,
        netuid: int = 71,
    ) -> None:
        self._transport = transport
        authority = dict(
            epoch_authority
            if epoch_authority is not None
            else {"mode": LEGACY_EPOCH_MODE, "cutover": None}
        )
        if set(authority) != {"mode", "cutover"}:
            raise QualificationEpochGuardV2Error(
                "qualification epoch authority fields are invalid"
            )
        self._epoch_mode = str(authority.get("mode") or "").strip().lower()
        self._netuid = int(netuid)
        if not 0 < self._netuid <= 0xFFFF:
            raise QualificationEpochGuardV2Error(
                "qualification epoch netuid is invalid"
            )
        self._cutover = None
        if self._epoch_mode == STATEFUL_EPOCH_MODE:
            try:
                self._cutover = SubnetEpochCutover.from_mapping(
                    authority.get("cutover")
                )
            except (SubnetEpochError, TypeError) as exc:
                raise QualificationEpochGuardV2Error(
                    "qualification epoch cutover is invalid"
                ) from exc
            if self._cutover.netuid != self._netuid:
                raise QualificationEpochGuardV2Error(
                    "qualification epoch cutover netuid differs"
                )
            if self._cutover.cutover_block <= 0:
                raise QualificationEpochGuardV2Error(
                    "qualification epoch cutover requires a predecessor"
                )
        elif self._epoch_mode != LEGACY_EPOCH_MODE or authority.get("cutover") is not None:
            raise QualificationEpochGuardV2Error(
                "qualification epoch authority is invalid"
            )

    def _rpc(
        self,
        method: str,
        params: tuple[Any, ...],
        request_id: int,
        *,
        historical: bool = False,
    ) -> Any:
        body = json_rpc_request(method, params, request_id)
        host = (
            CHAIN_ARCHIVE_ENDPOINT_HOST if historical else CHAIN_ENDPOINT_HOST
        )
        result = self._transport.execute_http(
            method="POST",
            url="https://%s%s" % (host, CHAIN_ENDPOINT_PATH),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            body=body,
            timeout_ms=CHAIN_RPC_TIMEOUT_MS,
        )
        if result.get("terminal_status") != "authenticated_response":
            raise QualificationEpochGuardV2Error(
                "chain epoch transport failed: %s" % result.get("failure_code")
            )
        if int(result.get("http_status", 0)) != 200:
            raise QualificationEpochGuardV2Error(
                "chain epoch RPC returned authenticated HTTP %s"
                % result.get("http_status")
            )
        try:
            response_body = base64.b64decode(
                str(result.get("body_b64") or ""), validate=True
            )
            return parse_json_rpc_response(response_body, request_id)
        except Exception as exc:
            raise QualificationEpochGuardV2Error(
                "chain epoch RPC response is malformed"
            ) from exc

    def _stateful_epoch(self) -> int:
        if self._cutover is None:
            raise QualificationEpochGuardV2Error(
                "qualification stateful epoch cutover is unavailable"
            )
        finalized_hash = "0x" + normalize_raw_hash(
            self._rpc("chain_getFinalizedHead", (), 1), "finalized block hash"
        )
        header = parse_finalized_header(
            self._rpc("chain_getHeader", (finalized_hash,), 2)
        )
        genesis_hash = "0x" + normalize_raw_hash(
            self._rpc(
                "chain_getBlockHash", (0,), 3, historical=True
            ),
            "genesis block hash",
        )
        cutover_hash = "0x" + normalize_raw_hash(
            self._rpc(
                "chain_getBlockHash",
                (self._cutover.cutover_block,),
                4,
                historical=True,
            ),
            "cutover block hash",
        )
        if genesis_hash != self._cutover.network_genesis_hash:
            raise QualificationEpochGuardV2Error(
                "qualification epoch cutover targets a different chain"
            )
        if cutover_hash != self._cutover.cutover_block_hash:
            raise QualificationEpochGuardV2Error(
                "qualification epoch cutover block hash differs"
            )
        if int(header["block"]) < self._cutover.cutover_block:
            raise QualificationEpochGuardV2Error(
                "qualification finalized head predates the epoch cutover"
            )

        # Prove that the configured cutover is the exact block where the
        # official per-subnet counter advanced.  Hash equality alone would let
        # an otherwise valid historical block be paired with an invented
        # settlement offset.
        cutover_header = parse_finalized_header(
            self._rpc(
                "chain_getHeader", (cutover_hash,), 5, historical=True
            )
        )
        predecessor_hash = "0x" + normalize_raw_hash(
            self._rpc(
                "chain_getBlockHash",
                (self._cutover.cutover_block - 1,),
                6,
                historical=True,
            ),
            "cutover predecessor block hash",
        )
        if (
            int(cutover_header["block"]) != self._cutover.cutover_block
            or "0x" + cutover_header["parent_hash"] != predecessor_hash
        ):
            raise QualificationEpochGuardV2Error(
                "qualification epoch cutover boundary is inconsistent"
            )

        def storage(
            storage_name: str,
            at_hash: str,
            request_id: int,
            *,
            historical: bool = False,
        ) -> int:
            return decode_subnet_epoch_storage(
                self._rpc(
                    "state_getStorage",
                    (
                        subnet_epoch_storage_key(
                            storage_name=storage_name, netuid=self._netuid
                        ),
                        at_hash,
                    ),
                    request_id,
                    historical=historical,
                ),
                storage_name=storage_name,
            )

        cutover_index = storage(
            "SubnetEpochIndex", cutover_hash, 7, historical=True
        )
        cutover_last_epoch_block = storage(
            "LastEpochBlock", cutover_hash, 8, historical=True
        )
        predecessor_index = storage(
            "SubnetEpochIndex", predecessor_hash, 9, historical=True
        )
        if (
            cutover_index != self._cutover.first_subnet_epoch_index
            or cutover_last_epoch_block != self._cutover.cutover_block
            or predecessor_index + 1 != cutover_index
        ):
            raise QualificationEpochGuardV2Error(
                "qualification epoch cutover is not an official transition"
            )

        storage_names = (
            "Tempo",
            "LastEpochBlock",
            "PendingEpochAt",
            "SubnetEpochIndex",
            "BlocksSinceLastStep",
        )
        state = {}
        for offset, storage_name in enumerate(storage_names, start=10):
            state[storage_name] = storage(
                storage_name, finalized_hash, offset
            )
        if (
            state["Tempo"] <= 0
            or state["LastEpochBlock"] > int(header["block"])
        ):
            raise QualificationEpochGuardV2Error(
                "qualification subnet epoch state is inconsistent"
            )
        try:
            return self._cutover.settlement_epoch_id(
                state["SubnetEpochIndex"]
            )
        except SubnetEpochError as exc:
            raise QualificationEpochGuardV2Error(
                "qualification subnet epoch predates the cutover"
            ) from exc

    def __call__(self, current_epoch: int, container_id: int = 0) -> bool:
        if (
            not isinstance(current_epoch, int)
            or isinstance(current_epoch, bool)
            or current_epoch < 0
            or not isinstance(container_id, int)
            or isinstance(container_id, bool)
            or container_id < 0
        ):
            raise QualificationEpochGuardV2Error("qualification epoch scope is invalid")
        if self._epoch_mode == STATEFUL_EPOCH_MODE:
            observed_epoch = self._stateful_epoch()
        else:
            header = parse_finalized_header(self._rpc("chain_getHeader", (), 1))
            observed_epoch = int(header["block"]) // CHAIN_FINALIZATION_EPOCH_BLOCKS
        if observed_epoch > current_epoch:
            print(
                "   WARNING: Container %s: authenticated epoch changed %s -> %s "
                "during batch processing!"
                % (container_id, current_epoch, observed_epoch)
            )
            return True
        return False
