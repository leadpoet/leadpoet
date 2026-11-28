"""Only purchases depend on the active emission lane; zero outstanding work is a normal first sale."""
import pytest
from lab_arena import chain
from lab_arena.tool_market import wiring


class Collection:
    start = None
    def collection_reward_start_epoch(self):
        return self.start


class Market:
    probes = 0
    def has_unallocated(self, *, epoch):
        self.probes += 1
        return False


def test_admission_uses_existing_cutover_and_schema_without_requiring_a_previous_sale():
    collection, market = Collection(), Market()
    current = [70]
    paid_epoch = wiring._purchase_epoch(lambda: current[0], collection, market)
    with pytest.raises(chain.ArenaChainError):
        paid_epoch()
    collection.start = 72
    with pytest.raises(chain.ArenaChainError):
        paid_epoch()
    current[0] = 72
    assert paid_epoch() == 72
    current[0] += 1
    assert paid_epoch() == 73 and market.probes == 1
