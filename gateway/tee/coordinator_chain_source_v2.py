"""Authenticated chain and price inputs for coordinator allocation authority."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import logging
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from Leadpoet.utils.subnet_epoch import (
    SubnetEpochCutover,
    SubnetEpochError,
)
from gateway.tee.provider_broker_v2 import PROVIDER_BROKER_SCHEMA_VERSION
from leadpoet_canonical.attested_v2 import sha256_bytes, sha256_json
from leadpoet_canonical.hotkey_authority_v2 import (
    select_chain_signing_profile,
    validate_chain_signing_profile,
)
from leadpoet_canonical.chain_source_v2 import (
    CHAIN_ARCHIVE_ENDPOINT_HOST,
    CHAIN_FINALIZATION_EPOCH_BLOCKS,
    CHAIN_SUBTENSOR_MAX_TEMPO,
    CHAIN_ENDPOINT_HOST,
    CHAIN_RPC_METHOD,
    CHAIN_RPC_RETRY_BACKOFF_SECONDS,
    CHAIN_RPC_TIMEOUT_MS,
    ChainSourceV2Error,
    decode_last_update_storage,
    decode_reveal_period_epochs_storage,
    decode_runtime_metadata_commitment,
    decode_subnet_epoch_storage,
    decode_timelocked_weight_commits,
    decode_weights_storage,
    decode_selective_metagraph_result,
    encode_selective_metagraph_params,
    json_rpc_request,
    last_update_storage_key,
    normalize_raw_hash,
    parse_finalized_header,
    parse_json_rpc_response,
    parse_runtime_version,
    reveal_period_epochs_storage_key,
    resolve_reveal_period_metadata_default_v2,
    ss58_encode_account_id,
    subnet_epoch_storage_key,
    system_event_count_storage_key,
    system_events_storage_key,
    timelocked_weight_commits_storage_key,
    weights_storage_key,
)
from leadpoet_canonical.subtensor_events_v2 import (
    RUNTIME_CODE_STORAGE_KEY,
    SubtensorEventsV2Error,
    load_subtensor_events_profile_v2,
    prove_timelocked_weights_reveal_v2,
    validate_subtensor_events_profile_v2,
)


CHAIN_ENDPOINT_URL = "https://%s/" % CHAIN_ENDPOINT_HOST
CHAIN_ARCHIVE_ENDPOINT_URL = "https://%s/" % CHAIN_ARCHIVE_ENDPOINT_HOST
COINGECKO_TAO_USD_URL = (
    "https://api.coingecko.com/api/v3/simple/price"
    "?ids=bittensor&vs_currencies=usd"
)
ALPHA_PRICE_RUNTIME_METHOD = "SwapRuntimeApi_current_alpha_price"
ALPHA_PRICE_TIMEOUT_MS = 8_000
ALPHA_PRICE_MAX_ATTEMPTS = 3
ALPHA_PRICE_RETRY_BACKOFF_SECONDS = (0.25, 0.5)
logger = logging.getLogger(__name__)


class CoordinatorChainSourceV2Error(RuntimeError):
    """An authenticated chain or price source could not be validated."""


def _utc_now_iso(clock: Callable[[], datetime]) -> str:
    value = clock()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class CoordinatorChainSourceV2:
    """Read finalized Bittensor state and TAO/USD through the measured broker."""

    def __init__(
        self,
        *,
        execute_provider: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        retry_policy_hashes: Mapping[str, str],
        epoch_authority: Optional[Mapping[str, Any]] = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._execute_provider = execute_provider
        self._retry_policy_hashes = {
            str(key): str(value) for key, value in retry_policy_hashes.items()
        }
        self._sleep = sleep
        self._clock = clock
        if epoch_authority is None:
            raise CoordinatorChainSourceV2Error(
                "coordinator epoch authority is unavailable"
            )
        authority = dict(epoch_authority)
        if set(authority) not in (
            {"mode", "cutover"},
            {"mode", "cutover", "chain_signing_profile"},
        ):
            raise CoordinatorChainSourceV2Error(
                "coordinator epoch authority fields are invalid"
            )
        if str(authority.get("mode") or "").strip().lower() != "stateful_v1":
            raise CoordinatorChainSourceV2Error(
                "coordinator epoch authority is invalid"
            )
        try:
            self._epoch_cutover = SubnetEpochCutover.from_mapping(
                authority.get("cutover")
            )
        except (SubnetEpochError, TypeError) as exc:
            raise CoordinatorChainSourceV2Error(
                "coordinator stateful epoch cutover is invalid"
            ) from exc
        profile = authority.get("chain_signing_profile")
        try:
            self._chain_signing_profile = (
                validate_chain_signing_profile(profile)
                if profile is not None
                else None
            )
        except Exception as exc:
            raise CoordinatorChainSourceV2Error(
                "coordinator chain signing profile is invalid"
            ) from exc
        if self._chain_signing_profile is not None:
            cutover_genesis = str(
                self._epoch_cutover.network_genesis_hash
            ).lower().removeprefix("0x")
            if cutover_genesis != str(
                self._chain_signing_profile["genesis_hash"]
            ).lower():
                raise CoordinatorChainSourceV2Error(
                    "coordinator epoch and signing genesis differ"
                )
        for provider_id in ("bittensor_chain", "coingecko"):
            if not self._retry_policy_hashes.get(provider_id):
                raise CoordinatorChainSourceV2Error(
                    "%s retry policy is unavailable" % provider_id
                )
        if not self._retry_policy_hashes.get("bittensor_archive"):
            raise CoordinatorChainSourceV2Error(
                "bittensor_archive retry policy is unavailable"
            )

    def _read_finalized_epoch_authority(
        self,
        *,
        netuid: int,
        finalized_hash: str,
        header: Mapping[str, Any],
        context: Any,
        attempt_number: int,
    ) -> Dict[str, Any]:
        """Bind a finalized header to the configured workflow epoch scheme."""

        cutover = self._epoch_cutover
        if cutover is None or cutover.netuid != int(netuid):
            raise CoordinatorChainSourceV2Error(
                "coordinator epoch cutover netuid differs"
            )
        if int(header["block"]) < cutover.cutover_block:
            raise CoordinatorChainSourceV2Error(
                "coordinator finalized head predates the epoch cutover"
            )

        def chain_call(
            method: str,
            params: Sequence[Any],
            request_id: int,
            operation: str,
        ) -> Any:
            return self._chain_call(
                method=method,
                params=params,
                request_id=request_id,
                logical_operation_id=(
                    "%s:epoch-authority:%s" % (context.job_id, operation)
                ),
                attempt_number=attempt_number,
                context=context,
            )

        def archive_call(
            method: str,
            params: Sequence[Any],
            request_id: int,
            operation: str,
        ) -> Any:
            return self._archive_call(
                method=method,
                params=params,
                request_id=request_id,
                logical_operation_id=(
                    "%s:epoch-authority:%s" % (context.job_id, operation)
                ),
                context=context,
            )

        genesis_hash = "0x" + normalize_raw_hash(
            archive_call("chain_getBlockHash", (0,), 10, "genesis-hash"),
            "coordinator genesis block hash",
        )
        cutover_hash = "0x" + normalize_raw_hash(
            archive_call(
                "chain_getBlockHash",
                (cutover.cutover_block,),
                11,
                "cutover-hash",
            ),
            "coordinator cutover block hash",
        )
        if genesis_hash != cutover.network_genesis_hash:
            raise CoordinatorChainSourceV2Error(
                "coordinator epoch cutover targets a different chain"
            )
        if cutover_hash != cutover.cutover_block_hash:
            raise CoordinatorChainSourceV2Error(
                "coordinator epoch cutover block hash differs"
            )
        cutover_header = parse_finalized_header(
            archive_call(
                "chain_getHeader",
                (cutover_hash,),
                12,
                "cutover-header",
            )
        )
        predecessor_hash = "0x" + normalize_raw_hash(
            archive_call(
                "chain_getBlockHash",
                (cutover.cutover_block - 1,),
                13,
                "cutover-predecessor-hash",
            ),
            "coordinator cutover predecessor block hash",
        )
        if (
            cutover.cutover_block <= 0
            or int(cutover_header["block"]) != cutover.cutover_block
            or "0x" + str(cutover_header["parent_hash"]) != predecessor_hash
        ):
            raise CoordinatorChainSourceV2Error(
                "coordinator epoch cutover boundary is inconsistent"
            )

        def storage(
            storage_name: str,
            at_hash: str,
            request_id: int,
            operation: str,
            *,
            historical: bool = False,
        ) -> int:
            try:
                return decode_subnet_epoch_storage(
                    (archive_call if historical else chain_call)(
                        "state_getStorage",
                        (
                            subnet_epoch_storage_key(
                                storage_name=storage_name,
                                netuid=int(netuid),
                            ),
                            at_hash,
                        ),
                        request_id,
                        operation,
                    ),
                    storage_name=storage_name,
                )
            except ChainSourceV2Error as exc:
                raise CoordinatorChainSourceV2Error(
                    "coordinator subnet epoch storage is invalid"
                ) from exc

        cutover_index = storage(
            "SubnetEpochIndex",
            cutover_hash,
            14,
            "cutover-index",
            historical=True,
        )
        cutover_last_epoch_block = storage(
            "LastEpochBlock",
            cutover_hash,
            15,
            "cutover-last-epoch-block",
            historical=True,
        )
        predecessor_index = storage(
            "SubnetEpochIndex",
            predecessor_hash,
            16,
            "cutover-predecessor-index",
            historical=True,
        )
        if (
            cutover_index != cutover.first_subnet_epoch_index
            or cutover_last_epoch_block != cutover.cutover_block
            or predecessor_index + 1 != cutover_index
        ):
            raise CoordinatorChainSourceV2Error(
                "coordinator epoch cutover is not an official transition"
            )

        state = {}
        for offset, storage_name in enumerate(
            (
                "Tempo",
                "LastEpochBlock",
                "PendingEpochAt",
                "SubnetEpochIndex",
                "BlocksSinceLastStep",
            ),
            start=17,
        ):
            state[storage_name] = storage(
                storage_name,
                "0x" + finalized_hash,
                offset,
                "finalized-%s" % storage_name.lower(),
            )
        if (
            state["Tempo"] <= 0
            or state["LastEpochBlock"] > int(header["block"])
        ):
            raise CoordinatorChainSourceV2Error(
                "coordinator finalized subnet epoch state is inconsistent"
            )
        try:
            workflow_epoch = cutover.settlement_epoch_id(
                state["SubnetEpochIndex"]
            )
        except SubnetEpochError as exc:
            raise CoordinatorChainSourceV2Error(
                "coordinator finalized subnet epoch predates the cutover"
            ) from exc
        return {
            "mode": "stateful_v1",
            "workflow_epoch_id": workflow_epoch,
            "official_subnet_epoch_id": state["SubnetEpochIndex"],
            "cutover_mapping_hash": cutover.mapping_hash,
            "state": state,
        }

    def read_finalized_metagraph(
        self,
        *,
        netuid: int,
        context: Any,
        attempt_number: int = 0,
    ) -> Dict[str, Any]:
        finalized = self._chain_call(
            method="chain_getFinalizedHead",
            params=(),
            request_id=1,
            logical_operation_id="%s:allocation:finalized-head" % context.job_id,
            attempt_number=attempt_number,
            context=context,
        )
        finalized_hash = normalize_raw_hash(finalized, "finalized head")
        header_value = self._chain_call(
            method="chain_getHeader",
            params=("0x" + finalized_hash,),
            request_id=2,
            logical_operation_id="%s:allocation:finalized-header" % context.job_id,
            attempt_number=attempt_number,
            context=context,
        )
        header = parse_finalized_header(header_value)
        metagraph_value = self._chain_call(
            method="state_call",
            params=(
                CHAIN_RPC_METHOD,
                encode_selective_metagraph_params(netuid=int(netuid)),
                "0x" + finalized_hash,
            ),
            request_id=3,
            logical_operation_id="%s:allocation:metagraph" % context.job_id,
            attempt_number=attempt_number,
            context=context,
        )
        metagraph = decode_selective_metagraph_result(metagraph_value)
        if int(metagraph["netuid"]) != int(netuid):
            raise CoordinatorChainSourceV2Error("allocation metagraph netuid differs")
        if int(metagraph["block"]) != int(header["block"]):
            raise CoordinatorChainSourceV2Error(
                "allocation metagraph and finalized header differ"
            )
        epoch_authority = self._read_finalized_epoch_authority(
            netuid=int(netuid),
            finalized_hash=finalized_hash,
            header=header,
            context=context,
            attempt_number=attempt_number,
        )
        return {
            "finalized_block_hash": finalized_hash,
            "header": header,
            "metagraph": metagraph,
            "workflow_epoch_id": epoch_authority["workflow_epoch_id"],
            "official_subnet_epoch_id": epoch_authority[
                "official_subnet_epoch_id"
            ],
            "epoch_authority": epoch_authority,
        }

    def read_tao_per_alpha(
        self,
        *,
        netuid: int,
        finalized_block_hash: str,
        context: Any,
        attempt_number: int,
    ) -> float:
        result = self._chain_call(
            method="state_call",
            params=(
                ALPHA_PRICE_RUNTIME_METHOD,
                "0x" + int(netuid).to_bytes(2, "little").hex(),
                "0x" + normalize_raw_hash(
                    finalized_block_hash, "alpha price finalized block"
                ),
            ),
            request_id=4,
            logical_operation_id="%s:allocation:alpha-price" % context.job_id,
            attempt_number=attempt_number,
            context=context,
        )
        text = str(result or "").lower()
        if not text.startswith("0x") or len(text) != 18:
            raise CoordinatorChainSourceV2Error("alpha price response is invalid")
        try:
            raw = bytes.fromhex(text[2:])
        except ValueError as exc:
            raise CoordinatorChainSourceV2Error(
                "alpha price response is invalid hex"
            ) from exc
        return int.from_bytes(raw, "little") / 1_000_000_000.0

    def read_tao_usd(self, *, context: Any, attempt_number: int) -> float:
        result = self._provider_call(
            provider_id="coingecko",
            logical_operation_id="%s:allocation:tao-usd" % context.job_id,
            attempt_number=attempt_number,
            method="GET",
            url=COINGECKO_TAO_USD_URL,
            headers={"accept": "application/json"},
            body=b"",
            timeout_ms=ALPHA_PRICE_TIMEOUT_MS,
            context=context,
        )
        try:
            parsed = json.loads(result["body"].decode("utf-8"))
            value = float(parsed["bittensor"]["usd"])
        except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
            raise CoordinatorChainSourceV2Error(
                "TAO/USD response is malformed"
            ) from exc
        if value < 0:
            raise CoordinatorChainSourceV2Error("TAO/USD response is negative")
        return value

    def resolve_live_prices(
        self,
        *,
        netuid: int,
        context: Any,
    ) -> Dict[str, Any]:
        last_error: Optional[BaseException] = None
        for attempt_number in range(ALPHA_PRICE_MAX_ATTEMPTS):
            try:
                snapshot = self.read_finalized_metagraph(
                    netuid=netuid,
                    context=context,
                    attempt_number=attempt_number,
                )
                tao_per_alpha = self.read_tao_per_alpha(
                    netuid=netuid,
                    finalized_block_hash=snapshot["finalized_block_hash"],
                    context=context,
                    attempt_number=attempt_number,
                )
                tao_usd = self.read_tao_usd(
                    context=context,
                    attempt_number=attempt_number,
                )
                return {
                    **snapshot,
                    "tao_per_alpha": tao_per_alpha,
                    "tao_usd": tao_usd,
                    "fetched_at": _utc_now_iso(self._clock),
                }
            except Exception as exc:  # every failed attempt has terminal records
                last_error = exc
                if attempt_number < len(ALPHA_PRICE_RETRY_BACKOFF_SECONDS):
                    self._sleep(ALPHA_PRICE_RETRY_BACKOFF_SECONDS[attempt_number])
        raise CoordinatorChainSourceV2Error(
            "live allocation price exhausted measured retries"
        ) from last_error

    def read_historical_finalized_weights(
        self,
        *,
        netuid: int,
        epoch_id: int,
        validator_hotkey: str,
        context: Any,
    ) -> Dict[str, Any]:
        """Read one epoch-end weight vector from the canonical archive node."""

        if not self._retry_policy_hashes.get("bittensor_archive"):
            raise CoordinatorChainSourceV2Error(
                "bittensor_archive retry policy is unavailable"
            )
        normalized_epoch = int(epoch_id)
        if normalized_epoch < 0 or int(netuid) <= 0 or not str(validator_hotkey):
            raise CoordinatorChainSourceV2Error("historical weight request is invalid")
        target_block = (
            (normalized_epoch + 1) * CHAIN_FINALIZATION_EPOCH_BLOCKS - 1
        )
        finalized_hash = normalize_raw_hash(
            self._archive_call(
                method="chain_getFinalizedHead",
                params=(),
                request_id=101,
                logical_operation_id=(
                    "%s:legacy-settlement:%d:finalized-head"
                    % (context.job_id, normalized_epoch)
                ),
                context=context,
            ),
            "archive finalized head",
        )
        finalized_header = parse_finalized_header(
            self._archive_call(
                method="chain_getHeader",
                params=("0x" + finalized_hash,),
                request_id=102,
                logical_operation_id=(
                    "%s:legacy-settlement:%d:finalized-header"
                    % (context.job_id, normalized_epoch)
                ),
                context=context,
            )
        )
        if int(finalized_header["block"]) < target_block:
            raise CoordinatorChainSourceV2Error(
                "historical settlement block is not finalized"
            )
        target_hash = normalize_raw_hash(
            self._archive_call(
                method="chain_getBlockHash",
                params=(target_block,),
                request_id=103,
                logical_operation_id=(
                    "%s:legacy-settlement:%d:block-hash"
                    % (context.job_id, normalized_epoch)
                ),
                context=context,
            ),
            "historical settlement block hash",
        )
        target_header = parse_finalized_header(
            self._archive_call(
                method="chain_getHeader",
                params=("0x" + target_hash,),
                request_id=104,
                logical_operation_id=(
                    "%s:legacy-settlement:%d:block-header"
                    % (context.job_id, normalized_epoch)
                ),
                context=context,
            )
        )
        if int(target_header["block"]) != target_block:
            raise CoordinatorChainSourceV2Error(
                "historical settlement header differs from target"
            )
        metagraph = decode_selective_metagraph_result(
            self._archive_call(
                method="state_call",
                params=(
                    CHAIN_RPC_METHOD,
                    encode_selective_metagraph_params(netuid=int(netuid)),
                    "0x" + target_hash,
                ),
                request_id=105,
                logical_operation_id=(
                    "%s:legacy-settlement:%d:metagraph"
                    % (context.job_id, normalized_epoch)
                ),
                context=context,
            )
        )
        if (
            int(metagraph["netuid"]) != int(netuid)
            or int(metagraph["block"]) != target_block
        ):
            raise CoordinatorChainSourceV2Error(
                "historical settlement metagraph differs from target"
            )
        matching_uids = [
            uid
            for uid, hotkey in enumerate(metagraph["hotkeys"])
            if hotkey == str(validator_hotkey)
        ]
        if len(matching_uids) != 1:
            raise CoordinatorChainSourceV2Error(
                "historical validator UID is absent or ambiguous"
            )
        validator_uid = matching_uids[0]
        storage_key = weights_storage_key(
            netuid=int(netuid), validator_uid=validator_uid
        )
        weights = decode_weights_storage(
            self._archive_call(
                method="state_getStorage",
                params=(storage_key, "0x" + target_hash),
                request_id=106,
                logical_operation_id=(
                    "%s:legacy-settlement:%d:weights"
                    % (context.job_id, normalized_epoch)
                ),
                context=context,
            )
        )
        return {
            "epoch_id": normalized_epoch,
            "netuid": int(netuid),
            "target_block": target_block,
            "target_block_hash": target_hash,
            "target_header": target_header,
            "finalized_head_block": int(finalized_header["block"]),
            "finalized_head_hash": finalized_hash,
            "validator_hotkey": str(validator_hotkey),
            "validator_uid": validator_uid,
            "weights_storage_key": storage_key,
            "weights": [[int(uid), int(weight)] for uid, weight in weights],
        }

    def read_stateful_epoch_close_weights(
        self,
        *,
        netuid: int,
        epoch_id: int,
        validator_hotkey: str,
        context: Any,
    ) -> Dict[str, Any]:
        """Read the active validator vector at one exact stateful epoch close."""

        cutover = self._epoch_cutover
        chain_signing_profile = self._chain_signing_profile
        normalized_netuid = int(netuid)
        normalized_epoch = int(epoch_id)
        normalized_hotkey = str(validator_hotkey or "")
        if (
            cutover is None
            or chain_signing_profile is None
            or cutover.netuid != normalized_netuid
            or normalized_epoch < cutover.first_settlement_epoch_id
            or not normalized_hotkey
        ):
            raise CoordinatorChainSourceV2Error(
                "stateful epoch weight request is invalid"
            )
        try:
            official_epoch = cutover.subnet_epoch_index(normalized_epoch)
        except SubnetEpochError as exc:
            raise CoordinatorChainSourceV2Error(
                "stateful epoch weight request predates the cutover"
            ) from exc
        target_next_official = official_epoch + 1

        request_id = 200

        def live_call(
            method: str,
            params: Sequence[Any],
            operation: str,
        ) -> Any:
            nonlocal request_id
            request_id += 1
            return self._chain_call(
                method=method,
                params=params,
                request_id=request_id,
                logical_operation_id=(
                    "%s:chain-realized:%d:%s"
                    % (context.job_id, normalized_epoch, operation)
                ),
                attempt_number=0,
                context=context,
            )

        def archive_call(
            method: str,
            params: Sequence[Any],
            operation: str,
        ) -> Any:
            nonlocal request_id
            request_id += 1
            return self._archive_call(
                method=method,
                params=params,
                request_id=request_id,
                logical_operation_id=(
                    "%s:chain-realized:%d:%s"
                    % (context.job_id, normalized_epoch, operation)
                ),
                context=context,
            )

        def block_hash(block: int, operation: str) -> str:
            return normalize_raw_hash(
                archive_call(
                    "chain_getBlockHash",
                    (int(block),),
                    operation,
                ),
                "stateful epoch block hash",
            )

        def epoch_index_at(block: int, operation: str) -> tuple[str, int]:
            observed_hash = block_hash(block, "%s-hash" % operation)
            try:
                observed_index = decode_subnet_epoch_storage(
                    archive_call(
                        "state_getStorage",
                        (
                            subnet_epoch_storage_key(
                                storage_name="SubnetEpochIndex",
                                netuid=normalized_netuid,
                            ),
                            "0x" + observed_hash,
                        ),
                        "%s-index" % operation,
                    ),
                    storage_name="SubnetEpochIndex",
                )
            except ChainSourceV2Error as exc:
                raise CoordinatorChainSourceV2Error(
                    "stateful epoch index storage is invalid"
                ) from exc
            return observed_hash, observed_index

        def epoch_transition_at(
            target_index: int,
            high_bound: int,
            operation: str,
        ) -> tuple[int, str]:
            """Find the first exact block of one official subnet epoch."""

            low_bound = cutover.cutover_block - 1
            prior_hash, prior_index = epoch_index_at(
                low_bound,
                "%s-low" % operation,
            )
            boundary_hash, boundary_index = epoch_index_at(
                high_bound,
                "%s-high" % operation,
            )
            if (
                int(target_index) < cutover.first_subnet_epoch_index
                or prior_index >= int(target_index)
                or boundary_index < int(target_index)
            ):
                raise CoordinatorChainSourceV2Error(
                    "stateful epoch transition search bounds are invalid"
                )
            while high_bound - low_bound > 1:
                midpoint = low_bound + ((high_bound - low_bound) // 2)
                midpoint_hash, midpoint_index = epoch_index_at(
                    midpoint,
                    "%s-search-%d" % (operation, midpoint),
                )
                if midpoint_index >= int(target_index):
                    high_bound = midpoint
                    boundary_hash = midpoint_hash
                    boundary_index = midpoint_index
                else:
                    low_bound = midpoint
                    prior_hash = midpoint_hash
                    prior_index = midpoint_index
            if boundary_index != int(target_index):
                boundary_hash, boundary_index = epoch_index_at(
                    high_bound,
                    "%s-boundary" % operation,
                )
            if low_bound != high_bound - 1:
                prior_hash, prior_index = epoch_index_at(
                    high_bound - 1,
                    "%s-prior" % operation,
                )
            if (
                boundary_index != int(target_index)
                or prior_index + 1 != int(target_index)
                or high_bound < cutover.cutover_block
            ):
                raise CoordinatorChainSourceV2Error(
                    "stateful epoch transition is inconsistent"
                )
            transition_header = parse_finalized_header(
                archive_call(
                    "chain_getHeader",
                    ("0x" + boundary_hash,),
                    "%s-header" % operation,
                )
            )
            if (
                int(transition_header["block"]) != high_bound
                or transition_header["parent_hash"] != prior_hash
            ):
                raise CoordinatorChainSourceV2Error(
                    "stateful epoch transition header is inconsistent"
                )
            return high_bound, boundary_hash

        finalized_hash = normalize_raw_hash(
            live_call(
                "chain_getFinalizedHead",
                (),
                "finalized-head",
            ),
            "live finalized head",
        )
        finalized_header = parse_finalized_header(
            live_call(
                "chain_getHeader",
                ("0x" + finalized_hash,),
                "finalized-header",
            )
        )
        high_block = int(finalized_header["block"])
        _high_hash, high_index = epoch_index_at(
            high_block,
            "search-high",
        )
        if high_index < target_next_official:
            raise CoordinatorChainSourceV2Error(
                "stateful epoch close is not finalized"
            )

        low_block = cutover.cutover_block - 1
        _low_hash, low_index = epoch_index_at(
            low_block,
            "search-low",
        )
        if low_index >= target_next_official:
            raise CoordinatorChainSourceV2Error(
                "stateful transition search low block is invalid"
            )

        while high_block - low_block > 1:
            midpoint = low_block + ((high_block - low_block) // 2)
            _mid_hash, mid_index = epoch_index_at(
                midpoint,
                "search-%d" % midpoint,
            )
            if mid_index >= target_next_official:
                high_block = midpoint
            else:
                low_block = midpoint

        boundary_hash, boundary_index = epoch_index_at(
            high_block,
            "boundary",
        )
        close_block = high_block - 1
        close_hash, close_index = epoch_index_at(
            close_block,
            "close",
        )
        if (
            boundary_index != target_next_official
            or close_index != official_epoch
            or close_block < cutover.cutover_block
        ):
            raise CoordinatorChainSourceV2Error(
                "stateful epoch close boundary is inconsistent"
            )
        boundary_header = parse_finalized_header(
            archive_call(
                "chain_getHeader",
                ("0x" + boundary_hash,),
                "boundary-header",
            )
        )
        close_header = parse_finalized_header(
            archive_call(
                "chain_getHeader",
                ("0x" + close_hash,),
                "close-header",
            )
        )
        if (
            int(boundary_header["block"]) != high_block
            or int(close_header["block"]) != close_block
            or boundary_header["parent_hash"] != close_hash
        ):
            raise CoordinatorChainSourceV2Error(
                "stateful epoch close headers are inconsistent"
            )

        epoch_start_block, epoch_start_hash = epoch_transition_at(
            official_epoch,
            close_block,
            "epoch-start",
        )
        reveal_window_start_block = epoch_start_block
        reveal_window_start_hash = epoch_start_hash

        metagraph = decode_selective_metagraph_result(
            archive_call(
                "state_call",
                (
                    CHAIN_RPC_METHOD,
                    encode_selective_metagraph_params(
                        netuid=normalized_netuid
                    ),
                    "0x" + close_hash,
                ),
                "close-metagraph",
            )
        )
        if (
            int(metagraph["netuid"]) != normalized_netuid
            or int(metagraph["block"]) != close_block
        ):
            raise CoordinatorChainSourceV2Error(
                "stateful epoch close metagraph differs"
            )
        matching_uids = [
            uid
            for uid, hotkey in enumerate(metagraph["hotkeys"])
            if hotkey == normalized_hotkey
        ]
        if len(matching_uids) != 1:
            raise CoordinatorChainSourceV2Error(
                "stateful epoch validator UID is absent or ambiguous"
            )
        validator_uid = matching_uids[0]
        storage_key = weights_storage_key(
            netuid=normalized_netuid,
            validator_uid=validator_uid,
        )
        try:
            weights = decode_weights_storage(
                archive_call(
                    "state_getStorage",
                    (storage_key, "0x" + close_hash),
                    "close-weights",
                )
            )
        except ChainSourceV2Error as exc:
            raise CoordinatorChainSourceV2Error(
                "stateful epoch close weights are invalid"
            ) from exc
        last_update_key = last_update_storage_key(netuid=normalized_netuid)
        try:
            last_updates = decode_last_update_storage(
                archive_call(
                    "state_getStorage",
                    (last_update_key, "0x" + close_hash),
                    "close-last-update",
                )
            )
        except ChainSourceV2Error as exc:
            raise CoordinatorChainSourceV2Error(
                "stateful epoch close LastUpdate is invalid"
            ) from exc
        if validator_uid >= len(last_updates):
            raise CoordinatorChainSourceV2Error(
                "stateful epoch validator LastUpdate is absent"
            )
        last_update_block = int(last_updates[validator_uid])
        if (
            last_update_block < cutover.cutover_block
            or last_update_block > close_block
        ):
            raise CoordinatorChainSourceV2Error(
                "stateful epoch validator LastUpdate is outside the cutover"
            )
        last_update_hash, last_update_official_epoch = epoch_index_at(
            last_update_block,
            "last-update",
        )
        try:
            latest_commit_source_epoch_id = cutover.settlement_epoch_id(
                last_update_official_epoch
            )
        except SubnetEpochError as exc:
            raise CoordinatorChainSourceV2Error(
                "stateful epoch validator LastUpdate predates the cutover"
            ) from exc
        if latest_commit_source_epoch_id > normalized_epoch:
            raise CoordinatorChainSourceV2Error(
                "stateful epoch latest commit source is in the future"
            )
        reveal_period_key = reveal_period_epochs_storage_key(
            netuid=normalized_netuid
        )
        try:
            reveal_period_override = decode_reveal_period_epochs_storage(
                archive_call(
                    "state_getStorage",
                    (reveal_period_key, "0x" + close_hash),
                    "close-reveal-period",
                )
            )
            close_runtime = parse_runtime_version(
                archive_call(
                    "state_getRuntimeVersion",
                    ("0x" + close_hash,),
                    "close-runtime-version",
                )
            )
        except ChainSourceV2Error as exc:
            raise CoordinatorChainSourceV2Error(
                "stateful epoch reveal-period authority is invalid"
            ) from exc
        try:
            selected_profile = select_chain_signing_profile(
                chain_signing_profile,
                runtime_version={
                    "specVersion": int(close_runtime["spec_version"]),
                    "transactionVersion": int(
                        close_runtime["transaction_version"]
                    ),
                },
                genesis_hash=str(chain_signing_profile["genesis_hash"]),
            )
        except ValueError as exc:
            raise CoordinatorChainSourceV2Error(
                "stateful epoch reveal-period runtime is not explicitly supported"
            ) from exc
        configured_reveal_period = int(
            selected_profile["subnet_reveal_period_epochs"]
        )
        try:
            metadata_commitment = decode_runtime_metadata_commitment(
                archive_call(
                    "state_getMetadata",
                    ("0x" + close_hash,),
                    "close-runtime-metadata",
                )
            )
            reviewed_reveal_period_default = (
                resolve_reveal_period_metadata_default_v2(
                    genesis_hash=chain_signing_profile["genesis_hash"],
                    runtime_spec_version=close_runtime["spec_version"],
                    runtime_transaction_version=close_runtime[
                        "transaction_version"
                    ],
                    metadata_hash=metadata_commitment["metadata_hash"],
                )
            )
        except ChainSourceV2Error as exc:
            raise CoordinatorChainSourceV2Error(
                "stateful epoch reveal-period metadata is invalid"
            ) from exc
        reveal_period_metadata_hash = metadata_commitment["metadata_hash"]
        reveal_period_epochs = (
            int(reviewed_reveal_period_default)
            if reveal_period_override is None
            else int(reveal_period_override)
        )
        if reveal_period_epochs != configured_reveal_period:
            raise CoordinatorChainSourceV2Error(
                "stateful epoch reveal period differs from the measured profile"
            )
        scheduled_reveal_official_epoch = None
        scheduled_reveal_source_epoch_id = None
        if official_epoch - reveal_period_epochs >= 0:
            scheduled_reveal_official_epoch = (
                official_epoch - reveal_period_epochs
            )
            try:
                scheduled_reveal_source_epoch_id = (
                    cutover.settlement_epoch_id(
                        scheduled_reveal_official_epoch
                    )
                )
            except SubnetEpochError:
                scheduled_reveal_source_epoch_id = None
        return {
            "schema_version": "leadpoet.stateful_epoch_close_weights.v1",
            "netuid": normalized_netuid,
            "epoch_id": normalized_epoch,
            "official_subnet_epoch_id": official_epoch,
            "cutover_mapping_hash": str(cutover.mapping_hash),
            "epoch_start_block": epoch_start_block,
            "epoch_start_block_hash": epoch_start_hash,
            "reveal_window_start_block": reveal_window_start_block,
            "reveal_window_start_block_hash": reveal_window_start_hash,
            "close_block": close_block,
            "close_block_hash": close_hash,
            "close_header": close_header,
            "next_epoch_block": high_block,
            "next_epoch_block_hash": boundary_hash,
            "finalized_head_block": int(finalized_header["block"]),
            "finalized_head_hash": finalized_hash,
            "validator_hotkey": normalized_hotkey,
            "validator_uid": validator_uid,
            "metagraph_hotkeys": list(metagraph["hotkeys"]),
            "weights_storage_key": storage_key,
            "last_update_storage_key": last_update_key,
            "reveal_period_storage_key": reveal_period_key,
            "reveal_period_storage_override": reveal_period_override,
            "reveal_period_metadata_hash": reveal_period_metadata_hash,
            "reveal_period_runtime_spec_version": int(
                close_runtime["spec_version"]
            ),
            "last_update_block": last_update_block,
            "last_update_block_hash": last_update_hash,
            "last_update_official_subnet_epoch_id": (
                last_update_official_epoch
            ),
            "latest_commit_source_epoch_id": latest_commit_source_epoch_id,
            "scheduled_reveal_subnet_epoch_id": (
                scheduled_reveal_official_epoch
            ),
            "scheduled_reveal_source_epoch_id": (
                scheduled_reveal_source_epoch_id
            ),
            "subnet_reveal_period_epochs": reveal_period_epochs,
            "chain_signing_profile": dict(chain_signing_profile),
            "chain_signing_profile_hash": sha256_json(
                chain_signing_profile
            ),
            "weights": [[int(uid), int(weight)] for uid, weight in weights],
        }

    def read_timelocked_reveal_proof(
        self,
        *,
        chain_state: Mapping[str, Any],
        authority: Mapping[str, Any],
        context: Any,
    ) -> Optional[Dict[str, Any]]:
        """Prove one finalized commit was successfully revealed on chain.

        The exact commit tuple is monotonic after its finalized inclusion: it
        remains in the source-epoch queue while a missing pulse is retried and
        becomes absent when processing finishes. The first absent post-state is
        authoritative only when that block has the exact successful reveal
        event and exact resulting vector.
        """

        request_id = 400
        proof_scope = sha256_json(
            {"bundle_hash": str(authority.get("bundle_hash") or "")}
        )[-16:]

        def archive_call(
            method: str,
            params: Sequence[Any],
            operation: str,
        ) -> Any:
            nonlocal request_id
            request_id += 1
            return self._archive_call(
                method=method,
                params=params,
                request_id=request_id,
                logical_operation_id=(
                    "%s:chain-reveal-proof:%s:%s"
                    % (context.job_id, proof_scope, operation)
                ),
                context=context,
            )

        hash_cache: dict[int, str] = {}

        def block_hash(block: int, operation: str) -> str:
            normalized_block = int(block)
            cached = hash_cache.get(normalized_block)
            if cached is not None:
                return cached
            observed = normalize_raw_hash(
                archive_call(
                    "chain_getBlockHash",
                    (normalized_block,),
                    operation,
                ),
                "timelocked reveal block hash",
            )
            hash_cache[normalized_block] = observed
            return observed

        def commits_at(
            block: int,
            *,
            storage_key: str,
            operation: str,
        ) -> tuple[str, Sequence[Dict[str, Any]]]:
            observed_hash = block_hash(block, "%s-hash" % operation)
            commits = decode_timelocked_weight_commits(
                archive_call(
                    "state_getStorage",
                    (storage_key, "0x" + observed_hash),
                    "%s-state" % operation,
                )
            )
            return observed_hash, commits

        try:
            netuid = int(chain_state["netuid"])
            close_block = int(chain_state["close_block"])
            close_hash = str(chain_state["close_block_hash"])
            reveal_window_start_block = int(
                chain_state["reveal_window_start_block"]
            )
            reveal_window_start_hash = str(
                chain_state["reveal_window_start_block_hash"]
            )
            scheduled_source_epoch = chain_state[
                "scheduled_reveal_source_epoch_id"
            ]
            scheduled_subnet_epoch = chain_state[
                "scheduled_reveal_subnet_epoch_id"
            ]
            if (
                scheduled_source_epoch is None
                or scheduled_subnet_epoch is None
            ):
                return None
            scheduled_source_epoch = int(scheduled_source_epoch)
            scheduled_subnet_epoch = int(scheduled_subnet_epoch)
            authorization = authority["extrinsic_authorization"]
            if not isinstance(authorization, Mapping):
                return None
            final_block = int(authority["finalized_block"])
            final_block_hash = str(authority["finalized_block_hash"])
            public_key = str(authorization["hotkey_public_key"])
            commitment_hex = str(authorization["commitment_hex"])
            commitment_hash = str(authorization["commitment_hash"])
            reveal_round = int(authorization["reveal_round"])
            expected_weights = [
                [int(uid), int(weight)]
                for uid, weight in zip(
                    authority["uids"], authority["weights_u16"]
                )
            ]
            if (
                int(authority["netuid"]) != netuid
                or int(authority["epoch_id"]) != scheduled_source_epoch
                or int(authority["subnet_epoch_index"])
                != scheduled_subnet_epoch
                or authority["validator_hotkey"]
                != chain_state["validator_hotkey"]
                or int(authorization["netuid"]) != netuid
                or int(authorization["epoch_id"]) != scheduled_source_epoch
                or int(authorization["subnet_epoch_index"])
                != scheduled_subnet_epoch
                or authorization["validator_hotkey"]
                != chain_state["validator_hotkey"]
                or ss58_encode_account_id(bytes.fromhex(public_key))
                != chain_state["validator_hotkey"]
                or sha256_bytes(bytes.fromhex(commitment_hex))
                != commitment_hash
                or expected_weights != chain_state["weights"]
                or final_block < 0
                or final_block >= close_block
                or close_block - max(final_block, reveal_window_start_block)
                > CHAIN_SUBTENSOR_MAX_TEMPO * 4
            ):
                return None
            actual_window_start_hash = block_hash(
                reveal_window_start_block,
                "window-start",
            )
            actual_close_hash = block_hash(close_block, "close")
            actual_final_hash = block_hash(final_block, "commit-finalized")
            if (
                actual_window_start_hash != reveal_window_start_hash
                or actual_close_hash != close_hash
                or actual_final_hash != final_block_hash
            ):
                return None
            commit_storage_key = timelocked_weight_commits_storage_key(
                netuid=netuid,
                subnet_epoch_index=scheduled_subnet_epoch,
            )
            _commit_hash, finalized_commits = commits_at(
                final_block,
                storage_key=commit_storage_key,
                operation="commit-finalized",
            )
            final_matches = [
                item
                for item in finalized_commits
                if item.get("hotkey_public_key") == public_key
                and item.get("commitment_hex") == commitment_hex
                and int(item.get("reveal_round", -1)) == reveal_round
                and sha256_json(item) == authority["state_transition_hash"]
            ]
            if len(final_matches) != 1:
                return None
            exact_entry = dict(final_matches[0])
            _close_hash, close_commits = commits_at(
                close_block,
                storage_key=commit_storage_key,
                operation="close",
            )
            if exact_entry in close_commits:
                return None

            low = final_block
            high = close_block
            while high - low > 1:
                midpoint = low + ((high - low) // 2)
                _mid_hash, midpoint_commits = commits_at(
                    midpoint,
                    storage_key=commit_storage_key,
                    operation="search-%d" % midpoint,
                )
                if exact_entry in midpoint_commits:
                    low = midpoint
                else:
                    high = midpoint
            reveal_block = high
            pre_reveal_block = reveal_block - 1
            if reveal_block < max(
                reveal_window_start_block,
                final_block + 1,
            ):
                return None
            pre_reveal_hash, pre_reveal_commits = commits_at(
                pre_reveal_block,
                storage_key=commit_storage_key,
                operation="pre-reveal",
            )
            reveal_hash, reveal_commits = commits_at(
                reveal_block,
                storage_key=commit_storage_key,
                operation="reveal",
            )
            pre_target_entries = [
                item
                for item in pre_reveal_commits
                if item.get("hotkey_public_key") == public_key
            ]
            reveal_target_entries = [
                item
                for item in reveal_commits
                if item.get("hotkey_public_key") == public_key
            ]
            if pre_target_entries != [exact_entry] or reveal_target_entries:
                return None
            pre_header = parse_finalized_header(
                archive_call(
                    "chain_getHeader",
                    ("0x" + pre_reveal_hash,),
                    "pre-reveal-header",
                )
            )
            reveal_header = parse_finalized_header(
                archive_call(
                    "chain_getHeader",
                    ("0x" + reveal_hash,),
                    "reveal-header",
                )
            )
            if (
                int(pre_header["block"]) != pre_reveal_block
                or int(reveal_header["block"]) != reveal_block
                or reveal_header["parent_hash"] != pre_reveal_hash
            ):
                return None

            reveal_runtime = parse_runtime_version(
                archive_call(
                    "state_getRuntimeVersion",
                    ("0x" + pre_reveal_hash,),
                    "pre-reveal-runtime",
                )
            )
            selected_profile = select_chain_signing_profile(
                self._chain_signing_profile,
                runtime_version={
                    "specVersion": int(reveal_runtime["spec_version"]),
                    "transactionVersion": int(
                        reveal_runtime["transaction_version"]
                    ),
                },
                genesis_hash=str(
                    self._chain_signing_profile["genesis_hash"]
                ),
            )
            metadata_value = archive_call(
                "state_getMetadata",
                ("0x" + pre_reveal_hash,),
                "pre-reveal-metadata",
            )
            if not isinstance(metadata_value, str) or not metadata_value.startswith(
                "0x"
            ):
                return None
            metadata_raw = bytes.fromhex(metadata_value[2:])
            metadata_commitment = decode_runtime_metadata_commitment(
                metadata_value
            )
            runtime_code_hash = normalize_raw_hash(
                archive_call(
                    "state_getStorageHash",
                    (RUNTIME_CODE_STORAGE_KEY, "0x" + pre_reveal_hash),
                    "pre-reveal-runtime-code-hash",
                ),
                "pre-reveal runtime code hash",
            )
            event_profile = validate_subtensor_events_profile_v2(
                load_subtensor_events_profile_v2(),
                genesis_hash=self._chain_signing_profile["genesis_hash"],
                spec_version=reveal_runtime["spec_version"],
                transaction_version=reveal_runtime["transaction_version"],
                metadata_raw=metadata_raw,
                runtime_code_hash="0x" + runtime_code_hash,
            )
            reveal_period_key = reveal_period_epochs_storage_key(netuid=netuid)
            reveal_period_override = decode_reveal_period_epochs_storage(
                archive_call(
                    "state_getStorage",
                    (reveal_period_key, "0x" + pre_reveal_hash),
                    "pre-reveal-period",
                )
            )
            reviewed_default = resolve_reveal_period_metadata_default_v2(
                genesis_hash=self._chain_signing_profile["genesis_hash"],
                runtime_spec_version=reveal_runtime["spec_version"],
                runtime_transaction_version=reveal_runtime[
                    "transaction_version"
                ],
                metadata_hash=metadata_commitment["metadata_hash"],
            )
            reveal_period = (
                reviewed_default
                if reveal_period_override is None
                else int(reveal_period_override)
            )
            if (
                int(reveal_period)
                != int(selected_profile["subnet_reveal_period_epochs"])
                or int(chain_state["official_subnet_epoch_id"])
                - int(reveal_period)
                != scheduled_subnet_epoch
                or int(chain_state["subnet_reveal_period_epochs"])
                != int(reveal_period)
            ):
                return None

            reveal_metagraph = decode_selective_metagraph_result(
                archive_call(
                    "state_call",
                    (
                        CHAIN_RPC_METHOD,
                        encode_selective_metagraph_params(netuid=netuid),
                        "0x" + reveal_hash,
                    ),
                    "reveal-metagraph",
                )
            )
            if (
                int(reveal_metagraph["netuid"]) != netuid
                or int(reveal_metagraph["block"]) != reveal_block
            ):
                return None
            matching_uids = [
                uid
                for uid, hotkey in enumerate(reveal_metagraph["hotkeys"])
                if hotkey == chain_state["validator_hotkey"]
            ]
            if matching_uids != [int(chain_state["validator_uid"])]:
                return None
            validator_uid = matching_uids[0]
            reveal_weights_key = weights_storage_key(
                netuid=netuid,
                validator_uid=validator_uid,
            )
            revealed_weights = [
                [int(uid), int(weight)]
                for uid, weight in decode_weights_storage(
                    archive_call(
                        "state_getStorage",
                        (reveal_weights_key, "0x" + reveal_hash),
                        "reveal-weights",
                    )
                )
            ]
            if revealed_weights != expected_weights:
                return None
            events_value = archive_call(
                "state_getStorage",
                (system_events_storage_key(), "0x" + reveal_hash),
                "reveal-events",
            )
            event_count_value = archive_call(
                "state_getStorage",
                (system_event_count_storage_key(), "0x" + reveal_hash),
                "reveal-event-count",
            )
            if (
                not isinstance(events_value, str)
                or not events_value.startswith("0x")
                or not isinstance(event_count_value, str)
                or not event_count_value.startswith("0x")
            ):
                return None
            event_witness = prove_timelocked_weights_reveal_v2(
                bytes.fromhex(events_value[2:]),
                profile=event_profile,
                event_count_raw=bytes.fromhex(event_count_value[2:]),
                expected_netuid=netuid,
                expected_uid=validator_uid,
                expected_account_id_hex=public_key,
            )
            revealed_vector_hash = sha256_json(
                {
                    "uids": [item[0] for item in revealed_weights],
                    "weights_u16": [item[1] for item in revealed_weights],
                }
            )
            proof_body = {
                "schema_version": (
                    "leadpoet.chain_realized_timelocked_reveal_proof.v2"
                ),
                "bundle_hash": str(authority["bundle_hash"]),
                "source_epoch_id": scheduled_source_epoch,
                "source_official_subnet_epoch_id": scheduled_subnet_epoch,
                "netuid": netuid,
                "validator_hotkey": str(chain_state["validator_hotkey"]),
                "validator_hotkey_public_key": public_key,
                "validator_uid": validator_uid,
                "commitment_hash": commitment_hash,
                "reveal_round": reveal_round,
                "commit_storage_key": commit_storage_key,
                "commit_finalized_block": final_block,
                "commit_finalized_block_hash": final_block_hash,
                "commit_state_transition_hash": str(
                    authority["state_transition_hash"]
                ),
                "reveal_window_start_block": reveal_window_start_block,
                "reveal_window_start_block_hash": reveal_window_start_hash,
                "pre_reveal_block": pre_reveal_block,
                "pre_reveal_block_hash": pre_reveal_hash,
                "pre_reveal_state_root": str(pre_header["state_root"]),
                "pre_reveal_commit_entry_hash": sha256_json(exact_entry),
                "reveal_block": reveal_block,
                "reveal_block_hash": reveal_hash,
                "reveal_parent_block_hash": str(
                    reveal_header["parent_hash"]
                ),
                "reveal_state_root": str(reveal_header["state_root"]),
                "reveal_commit_entry_absent": True,
                "reveal_runtime_spec_version": int(
                    reveal_runtime["spec_version"]
                ),
                "reveal_runtime_transaction_version": int(
                    reveal_runtime["transaction_version"]
                ),
                "reveal_runtime_code_hash": runtime_code_hash,
                "reveal_metadata_hash": str(
                    metadata_commitment["metadata_hash"]
                ),
                "reveal_period_epochs": int(reveal_period),
                "system_events_storage_key": system_events_storage_key(),
                "system_event_count_storage_key": (
                    system_event_count_storage_key()
                ),
                "event_witness": dict(event_witness),
                "weights_storage_key": reveal_weights_key,
                "revealed_weights": revealed_weights,
                "revealed_weights_vector_hash": revealed_vector_hash,
            }
            return {
                **proof_body,
                "proof_hash": sha256_json(proof_body),
            }
        except (
            ChainSourceV2Error,
            SubtensorEventsV2Error,
            ValueError,
        ) as exc:
            logger.warning(
                "timelocked reveal proof is unavailable: %s",
                exc.__class__.__name__,
            )
            return None

    def _chain_call(
        self,
        *,
        method: str,
        params: Sequence[Any],
        request_id: int,
        logical_operation_id: str,
        attempt_number: int,
        context: Any,
    ) -> Any:
        body = json_rpc_request(method, params, request_id)
        result = self._provider_call(
            provider_id="bittensor_chain",
            logical_operation_id=logical_operation_id,
            attempt_number=attempt_number,
            method="POST",
            url=CHAIN_ENDPOINT_URL,
            headers={"accept": "application/json", "content-type": "application/json"},
            body=body,
            timeout_ms=CHAIN_RPC_TIMEOUT_MS,
            context=context,
        )
        try:
            return parse_json_rpc_response(result["body"], request_id)
        except ChainSourceV2Error as exc:
            raise CoordinatorChainSourceV2Error(
                "authenticated chain response is invalid"
            ) from exc

    def _archive_call(
        self,
        *,
        method: str,
        params: Sequence[Any],
        request_id: int,
        logical_operation_id: str,
        context: Any,
    ) -> Any:
        body = json_rpc_request(method, params, request_id)
        last_error: Optional[BaseException] = None
        for attempt_number in range(len(CHAIN_RPC_RETRY_BACKOFF_SECONDS) + 1):
            try:
                result = self._provider_call(
                    provider_id="bittensor_archive",
                    logical_operation_id=logical_operation_id,
                    attempt_number=attempt_number,
                    method="POST",
                    url=CHAIN_ARCHIVE_ENDPOINT_URL,
                    headers={
                        "accept": "application/json",
                        "content-type": "application/json",
                    },
                    body=body,
                    timeout_ms=CHAIN_RPC_TIMEOUT_MS,
                    context=context,
                )
                return parse_json_rpc_response(result["body"], request_id)
            except (CoordinatorChainSourceV2Error, ChainSourceV2Error) as exc:
                last_error = exc
                if attempt_number < len(CHAIN_RPC_RETRY_BACKOFF_SECONDS):
                    self._sleep(CHAIN_RPC_RETRY_BACKOFF_SECONDS[attempt_number])
        raise CoordinatorChainSourceV2Error(
            "authenticated archive request exhausted measured retries"
        ) from last_error

    def _provider_call(
        self,
        *,
        provider_id: str,
        logical_operation_id: str,
        attempt_number: int,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout_ms: int,
        context: Any,
    ) -> Dict[str, Any]:
        result = dict(
            self._execute_provider(
                {
                    "schema_version": PROVIDER_BROKER_SCHEMA_VERSION,
                    "logical_operation_id": logical_operation_id,
                    "job_id": context.job_id,
                    "purpose": context.purpose,
                    "provider_id": provider_id,
                    "attempt_number": int(attempt_number),
                    "method": method,
                    "url": url,
                    "headers": dict(headers),
                    "body_b64": base64.b64encode(body).decode("ascii"),
                    "timeout_ms": int(timeout_ms),
                    "retry_policy_hash": self._retry_policy_hashes[provider_id],
                }
            )
        )
        attempt = result.get("transport_attempt")
        if not isinstance(attempt, Mapping):
            raise CoordinatorChainSourceV2Error("provider terminal attempt is missing")
        context.record_transport(attempt)
        context.record_artifact(str(attempt["request_artifact_hash"]))
        if attempt.get("terminal_status") == "authenticated_response":
            context.record_artifact(str(attempt["response_artifact_hash"]))
        if (
            result.get("terminal_status") != "authenticated_response"
            or not 200 <= int(result.get("http_status") or 0) < 300
        ):
            raise CoordinatorChainSourceV2Error(
                "%s request failed: %s"
                % (
                    provider_id,
                    result.get("failure_code")
                    or "http_%s" % result.get("http_status"),
                )
            )
        try:
            response_body = base64.b64decode(
                str(result.get("body_b64") or ""), validate=True
            )
        except Exception as exc:
            raise CoordinatorChainSourceV2Error(
                "%s response encoding is invalid" % provider_id
            ) from exc
        if sha256_bytes(response_body) != attempt.get("response_hash"):
            raise CoordinatorChainSourceV2Error(
                "%s response differs from terminal record" % provider_id
            )
        return {"body": response_body, "attempt": dict(attempt)}
