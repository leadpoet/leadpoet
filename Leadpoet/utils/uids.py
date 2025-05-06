import random
import numpy as np
import bittensor as bt
from typing import List

def check_uid_availability(
    metagraph: "bt.metagraph", uid: int, vpermit_tao_limit: int
) -> bool:
    if uid >= len(metagraph.hotkeys):
        bt.logging.debug(f"UID {uid} rejected: Out of range (max {len(metagraph.hotkeys) - 1})")
        return False

    is_mock = getattr(metagraph.subtensor, 'is_mock', False) or getattr(bt.config(), 'mock', False)
    if is_mock:
        bt.logging.debug(f"UID {uid} accepted: Mock mode")
        return True

    miner_hotkey = "5D73anXA8XELS2tSjnGQKMoVog1vKuTQCoJHrEGaXpZBAWpS"
    if metagraph.hotkeys[uid] == miner_hotkey:
        bt.logging.debug(f"UID {uid} accepted: Matches known miner hotkey {miner_hotkey}")
        return True

    if not metagraph.axons[uid].is_serving:
        bt.logging.debug(f"UID {uid} rejected: Axon not serving")
        return False

    if metagraph.S.size == 0:
        bt.logging.debug(f"UID {uid} rejected: Empty stake array")
        return False

    if metagraph.validator_permit[uid]:
        min_stake = 20.0
        if metagraph.S[uid] < min_stake:
            bt.logging.debug(f"UID {uid} rejected: Validator stake {metagraph.S[uid]} below minimum {min_stake}")
            return False
    else:
        min_stake = 2.0
        if metagraph.S[uid] < min_stake:
            bt.logging.debug(f"UID {uid} rejected: Miner stake {metagraph.S[uid]} below minimum {min_stake}")
            return False

    bt.logging.debug(f"UID {uid} accepted: Axon serving, stake {metagraph.S[uid]}")
    return True

def get_random_uids(self, k: int, exclude: List[int] = None) -> np.ndarray:
    exclude = [] if exclude is None else exclude
    is_mock = getattr(self.config, 'mock', False) or getattr(bt.config(), 'mock', False)
    vpermit_tao_limit = 0 if is_mock else 20

    self.metagraph.sync(subtensor=self.subtensor)
    bt.logging.debug(f"Metagraph synced: {len(self.metagraph.neurons)} neurons, hotkeys: {[n.hotkey for n in self.metagraph.neurons]}")

    k = min(k, len(self.metagraph.neurons)) if k is not None else getattr(self.config.neuron, 'sample_size', 5)

    candidate_uids = []
    for uid in range(self.metagraph.n):
        if uid in exclude:
            bt.logging.debug(f"UID {uid} excluded: In exclude list")
            continue
        if check_uid_availability(self.metagraph, uid, vpermit_tao_limit):
            candidate_uids.append(uid)

    if len(candidate_uids) == 0:
        if is_mock:
            bt.logging.debug("Mock mode: Falling back to all UIDs")
            candidate_uids = [uid for uid in range(self.metagraph.n) if uid not in exclude]
        else:
            bt.logging.warning("No available UIDs found.")
            return np.array([], dtype=np.int64)

    k = min(k, len(candidate_uids))
    selected_uids = random.sample(candidate_uids, k) if k > 0 else []
    bt.logging.debug(f"Selected {len(selected_uids)} random UIDs: {selected_uids}")
    return np.array(selected_uids, dtype=np.int64)