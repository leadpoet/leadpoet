from __future__ import annotations

import json
from typing import Any

from Leadpoet.utils.subnet_epoch import CUTOVER_JSON_ENV, SubnetEpochCutover


def epoch_test_cutover() -> SubnetEpochCutover:
    return SubnetEpochCutover(
        network_genesis_hash=(
            "0x2f0555cc76fc2840a25a6ea3b9637146806f1f44"
            "b090c175ffde2a7e5ab36c03"
        ),
        netuid=71,
        cutover_block=8_637_156,
        cutover_block_hash="0x" + "2" * 64,
        first_subnet_epoch_index=23_927,
        first_settlement_epoch_id=23_992,
        last_legacy_epoch_id=23_991,
    )


def epoch_test_environment(**updates: Any) -> dict[str, Any]:
    return {
        CUTOVER_JSON_ENV: json.dumps(epoch_test_cutover().to_dict()),
        **updates,
    }
