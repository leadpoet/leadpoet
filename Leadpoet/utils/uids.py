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

    if not metagraph.axons[uid].is_serving:
        bt.logging.debug(f"UID {uid} rejected: Axon not serving")
        return False

    if metagraph.S.size == 0:
        bt.logging.debug(f"UID {uid} rejected: Empty stake array")
        return False

    if metagraph.validator_permit[uid]:
        min_stake = 0.0  # Allow 0 stake for testnet CHANGE to 20 FOR MAINNET
        if metagraph.S[uid] < min_stake:
            bt.logging.debug(f"UID {uid} rejected: Validator stake {metagraph.S[uid]} below minimum {min_stake}")
            return False
    else:
        min_stake = 0.0  # Allow 0 stake for testnet CHANGE to 2 FOR MAINNET
        if metagraph.S[uid] < min_stake:
            bt.logging.debug(f"UID {uid} rejected: Miner stake {metagraph.S[uid]} below minimum {min_stake}")
            return False

    bt.logging.debug(f"UID {uid} accepted: Axon serving, stake {metagraph.S[uid]}")
    return True

def get_random_uids(self, k: int, exclude: List[int] = []) -> np.ndarray:
    vpermit_tao_limit = 20

    self.metagraph.sync(subtensor=self.subtensor)
    print(f"🔍 Validator metagraph synced: {len(self.metagraph.neurons)} neurons")
    print(f"   Hotkeys: {[n.hotkey for n in self.metagraph.neurons]}")
    print(f"   Axons serving: {[i for i, axon in enumerate(self.metagraph.axons) if axon.is_serving]}")
    print(f"   Stakes: {self.metagraph.S.tolist()}")

    k = min(k, len(self.metagraph.neurons)) if k is not None else getattr(self.config.neuron, 'sample_size', 5)

    candidate_uids = []
    for uid in range(self.metagraph.n):
        if uid in exclude:
            print(f"   UID {uid} rejected: In exclude list")
            continue
        # Exclude validators from being queried for curation
        if self.metagraph.validator_permit[uid]:
            print(f"   UID {uid} rejected: Is validator (not a miner)")
            continue
        if check_uid_availability(self.metagraph, uid, vpermit_tao_limit):
            candidate_uids.append(uid)
        else:
            print(f"   UID {uid} rejected: Not available")

    if len(candidate_uids) == 0:
        print("❌ No available UIDs found.")
        return np.array([], dtype=np.int64)

    k = min(k, len(candidate_uids))
    selected_uids = random.sample(candidate_uids, k) if k > 0 else []
    print(f"✅ Selected {len(selected_uids)} random UIDs: {selected_uids}")
    return np.array(selected_uids, dtype=np.int64)