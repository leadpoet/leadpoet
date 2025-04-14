import random
import numpy as np
import bittensor as bt
from typing import List


def check_uid_availability(
    metagraph: "bt.metagraph", uid: int, vpermit_tao_limit: int
) -> bool:
    """Check if a UID is available and meets staking requirements."""
    if uid >= len(metagraph.hotkeys):
        return False

    is_mock = getattr(metagraph.subtensor, 'is_mock', False)
    if is_mock:
        return True  # In mock mode, all UIDs are available

    if not metagraph.axons[uid].is_serving:
        return False

    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] < vpermit_tao_limit:
            return False

    return True


def get_random_uids(self, k: int, exclude: List[int] = None) -> np.ndarray:
    """Return k random UIDs, excluding specified UIDs and unavailable neurons."""
    exclude = [] if exclude is None else exclude
    is_mock = getattr(self.config, 'mock', False)
    vpermit_tao_limit = (
        0 if is_mock else self.config.neuron.vpermit_tao_limit
    )

    # Fallback if k is None or invalid
    k = k if k is not None else getattr(self.config.neuron, 'sample_size', 10)

    candidate_uids = []
    for uid in range(self.metagraph.n):
        if uid in exclude:
            continue
        if check_uid_availability(
            self.metagraph, uid, vpermit_tao_limit
        ):
            candidate_uids.append(uid)

    if len(candidate_uids) == 0:
        bt.logging.warning("No available UIDs found.")
        return np.array([], dtype=np.int64)

    k = min(k, len(candidate_uids))
    return np.array(
        random.sample(candidate_uids, k), dtype=np.int64
    )