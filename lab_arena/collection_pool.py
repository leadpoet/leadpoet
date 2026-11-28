"""Consent shared by standing miner funding, verified sandbox work and training exports."""
from lab_arena import contracts

POOL_SCOPE = "sales_sandbox"
POOL_PAYMENT_RULE = "verified_actual_cost_v1"
# Frozen in every standing experiment manifest, including later assignments.
POOL_TERMS = (
    "I authorize Leadpoet to use my OpenRouter key for operator-run synthetic sales "
    "experiments on Finney subnet 71, including comparisons, same-model repeats, fresh expert "
    "runs and corrections from saved states. The subnet selects the tasks, agents and approved "
    "model routes. My signed spending limit applies to all calls dispatched under this enrollment and does not "
    "reset daily. Settled charges and unresolved reservations consume this limit. I may revoke "
    "funding at any time. Revocation stops new calls; already dispatched calls may still be billed. "
    "Normal keys have priority over reserve keys. Within each priority, available authorized "
    "budget and then earliest enrollment determine selection. For each authorized, verified "
    "complete run, my hotkey earns a USD reward target of 1.25 times its actual settled generator "
    "charges, whether the agent passes or fails the task. Partial or unverifiable work may cost "
    "credits and earns no target. Leadpoet funds private verification. If total targets exceed "
    "available emissions, targets scale proportionally; unpaid differences do not carry forward. "
    "Validator votes, weight rounding and token prices affect realized alpha payments. "
    "Leadpoet may record, store, evaluate, train on, publish and license eligible messages, "
    "provider-exposed reasoning, tools, results, evidence, intermediate states, costs, checks and "
    "outcomes from successful, failed and partial runs, including full trajectories and chosen/"
    "rejected pairs. I assign any rights I hold in these outputs to Leadpoet, or grant a perpetual "
    "worldwide royalty-free transferable and sublicensable license where assignment is unavailable. "
    "Other rights holders' licenses and privacy protections still apply. I will disable provider "
    "input/output logging, sharing and external trace broadcast, keep private run data confidential "
    "and retain no copies beyond my funding statements. The read-only management key verifies "
    "ownership and settings of my runtime key. Live web, hosted search, Deepline and other paid "
    "research tools are outside this authorization."
)
POOL_TERMS_HASH = contracts.hash_bytes(POOL_TERMS.encode())


POOL_TERMS_150 = POOL_TERMS.replace("1.25 times", "1.50 times")
POOL_TERMS_BY_RATE = {
    12500: (POOL_TERMS, POOL_TERMS_HASH),
    15000: (POOL_TERMS_150, contracts.hash_bytes(POOL_TERMS_150.encode())),
}
POOL_GRANT_HASHES = frozenset(digest for _, digest in POOL_TERMS_BY_RATE.values())


def require_reward_rate(rate_bps):
    if type(rate_bps) is not int or rate_bps not in POOL_TERMS_BY_RATE:
        raise ValueError("unknown standing reward rate")
    return rate_bps


def job_reward_rate(job):
    # Readbacks for grants assigned before the rate migration omit this field.
    return require_reward_rate(job.get("reward_rate_bps", 12500))


def pool_target_microusd(job, settled_microusd):
    if type(settled_microusd) is not int or settled_microusd < 0:
        raise ValueError("invalid settled generator cost")
    return (settled_microusd * job_reward_rate(job) + 9999) // 10000
