"""Deterministic crowning statistics for the open verifier.

This module publishes the arithmetic described in the Research Lab v5 plan:

* paired delta / lower-confidence-bound calculations for probation looks,
* derived ``delta_min`` from paired variance plus temporal buffer,
* same-day activation gates,
* operating-characteristic inputs for attacker-EV analysis, and
* fixed-term grant expected-value helpers.

It is intentionally dependency-free and does not call production champion
selection. Production workflows should treat this as a verifier/calculator
surface until the lab-only probation workflow is wired explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, erf, log, pi, sqrt
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


MIN_PROBATION_LOOKS = 3
MAX_PROBATION_LOOKS = 5
PROBATION_LOOKS = (3, 4, 5)

DEFAULT_DELTA_MIN_FLOOR = 2.0
DEFAULT_DELTA_MIN_SE_MULTIPLIER = 2.0
DEFAULT_TEMPORAL_BUFFER_MAX_SAME_DAY = 1.5
DEFAULT_REQUIRED_TEMPORAL_BUFFER_DAYS = 20

DEFAULT_CONTINUITY_BASELINE_MARGIN = 10.0
DEFAULT_CONTINUITY_ABSOLUTE_FLOOR = 20.0

DEFAULT_REENTRY_CAP_PER_7_DAYS = 3
DEFAULT_OC_SIMULATION_ITERATIONS = 20000
DEFAULT_OC_SIMULATION_SEED = 8675309

# Plan-published one-sided 95% t constants, with the N=4 df=3 value pinned so
# all three adaptive looks are reproducible in the open verifier.
STUDENT_T_ONE_SIDED_95_BY_N = {
    3: 2.920,
    4: 2.353,
    5: 2.132,
}

# O'Brien-Fleming-type early-look tightening used by Phase 0 fixtures:
# t_n / sqrt(n / max_n). These constants are published here so validators and
# reviewers can reproduce the exact decision boundary byte-for-byte. If a
# future governance/review pass adopts a different calibrated table, the table
# and golden fixtures must change together. This table is an explicit Phase 0
# boundary table, not by itself a proof that realized family-wise alpha is 5%.
GROUP_SEQUENTIAL_T_BOUNDARY_BY_N = {
    n: STUDENT_T_ONE_SIDED_95_BY_N[n] / sqrt(float(n) / float(MAX_PROBATION_LOOKS))
    for n in PROBATION_LOOKS
}


@dataclass(frozen=True)
class PairedStats:
    n: int
    mean_delta: float
    sd_delta: float
    se_delta: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "n": self.n,
            "mean_delta": self.mean_delta,
            "sd_delta": self.sd_delta,
            "se_delta": self.se_delta,
        }


def paired_deltas(challenger_scores: Sequence[float], champion_scores: Sequence[float]) -> List[float]:
    """Return challenger-minus-champion deltas for paired probation sets."""
    if len(challenger_scores) != len(champion_scores):
        raise ValueError("challenger_scores and champion_scores must have the same length")
    if not challenger_scores:
        raise ValueError("at least one paired score is required")
    return [
        float(challenger) - float(champion)
        for challenger, champion in zip(challenger_scores, champion_scores)
    ]


def paired_stats(deltas: Iterable[float]) -> PairedStats:
    """Compute sample SD and SE for paired score deltas."""
    values = [float(value) for value in deltas]
    if not values:
        raise ValueError("at least one paired delta is required")
    mean_delta = sum(values) / len(values)
    if len(values) == 1:
        sd_delta = 0.0
    else:
        variance = sum((value - mean_delta) ** 2 for value in values) / (len(values) - 1)
        sd_delta = sqrt(variance)
    se_delta = sd_delta / sqrt(len(values))
    return PairedStats(
        n=len(values),
        mean_delta=mean_delta,
        sd_delta=sd_delta,
        se_delta=se_delta,
    )


def boundary_for_look(n: int, *, use_alpha_spending: bool = True) -> float:
    """Return the published t boundary for a probation look."""
    if n not in PROBATION_LOOKS:
        raise ValueError("probation look n must be one of 3, 4, or 5")
    if use_alpha_spending:
        return GROUP_SEQUENTIAL_T_BOUNDARY_BY_N[n]
    return STUDENT_T_ONE_SIDED_95_BY_N[n]


def lower_confidence_bound(
    deltas: Iterable[float],
    *,
    use_alpha_spending: bool = True,
) -> Dict[str, float]:
    """Return mean(d) - boundary * sd(d) / sqrt(N) for N in {3,4,5}."""
    stats = paired_stats(deltas)
    if stats.n not in PROBATION_LOOKS:
        raise ValueError("crowning LCB requires exactly 3, 4, or 5 paired deltas")
    boundary = boundary_for_look(stats.n, use_alpha_spending=use_alpha_spending)
    lcb = stats.mean_delta - boundary * stats.se_delta
    return {
        "n": stats.n,
        "mean_delta": stats.mean_delta,
        "sd_delta": stats.sd_delta,
        "se_delta": stats.se_delta,
        "boundary": boundary,
        "lower_confidence_bound": lcb,
    }


def continuity_floor(
    public_baseline_score: float,
    *,
    baseline_margin: float = DEFAULT_CONTINUITY_BASELINE_MARGIN,
    absolute_floor: float = DEFAULT_CONTINUITY_ABSOLUTE_FLOOR,
) -> float:
    """Return max(public_baseline + margin, absolute_floor)."""
    return max(float(public_baseline_score) + float(baseline_margin), float(absolute_floor))


def derive_delta_min(
    within_day_paired_deltas: Iterable[float],
    *,
    temporal_buffer_b: float = 0.0,
    paired_days_for_b: int = 0,
    truth_supply_ready: bool = False,
    floor: float = DEFAULT_DELTA_MIN_FLOOR,
    se_multiplier: float = DEFAULT_DELTA_MIN_SE_MULTIPLIER,
    required_temporal_buffer_days: int = DEFAULT_REQUIRED_TEMPORAL_BUFFER_DAYS,
    temporal_buffer_max_same_day: float = DEFAULT_TEMPORAL_BUFFER_MAX_SAME_DAY,
) -> Dict[str, object]:
    """Derive same-day ``delta_min`` and activation state.

    The v5 default is ``max(2 * SE_paired_within_day + B, 2.0)``. Same-day
    mode remains blocked until the temporal buffer has enough paired days, B is
    below the launch threshold, and the external truth-supply gate is ready.
    """
    stats = paired_stats(within_day_paired_deltas)
    if stats.n < 2:
        raise ValueError("delta_min derivation requires at least two paired deltas")

    temporal_buffer = float(temporal_buffer_b)
    delta_min = max(
        float(floor),
        float(se_multiplier) * stats.se_delta + temporal_buffer,
    )

    blockers: List[str] = []
    if int(paired_days_for_b) < int(required_temporal_buffer_days):
        blockers.append("insufficient_paired_days_for_temporal_buffer")
    if temporal_buffer > float(temporal_buffer_max_same_day):
        blockers.append("temporal_buffer_above_same_day_threshold")
    if not truth_supply_ready:
        blockers.append("truth_supply_not_ready")

    return {
        "paired_count": stats.n,
        "paired_mean_delta": stats.mean_delta,
        "paired_sd": stats.sd_delta,
        "se_paired": stats.se_delta,
        "temporal_buffer_b": temporal_buffer,
        "delta_min_same_day": delta_min,
        "fallback_delta_min": float(floor),
        "same_day_ready": not blockers,
        "mode": "same_day" if not blockers else "two_day_fallback",
        "blockers": blockers,
    }


def evaluate_crowning(
    challenger_scores: Sequence[float],
    champion_scores: Sequence[float],
    *,
    public_baseline_score: float,
    delta_min: float,
    canary_fabrication_confirmed: bool = False,
    challenger_mean: Optional[float] = None,
    use_alpha_spending: bool = True,
) -> Dict[str, object]:
    """Evaluate the open crowning rule for one probation look."""
    deltas = paired_deltas(challenger_scores, champion_scores)
    lcb_record = lower_confidence_bound(deltas, use_alpha_spending=use_alpha_spending)
    challenger_average = (
        float(challenger_mean)
        if challenger_mean is not None
        else sum(float(score) for score in challenger_scores) / len(challenger_scores)
    )
    floor = continuity_floor(public_baseline_score)

    statistical_pass = lcb_record["lower_confidence_bound"] >= float(delta_min)
    continuity_pass = challenger_average >= floor
    canary_pass = not canary_fabrication_confirmed
    crown = bool(statistical_pass and continuity_pass and canary_pass)

    if crown:
        decision = "crown"
    elif not canary_pass:
        decision = "reject"
    elif len(deltas) < MAX_PROBATION_LOOKS:
        decision = "continue"
    else:
        decision = "reject"

    result: Dict[str, object] = dict(lcb_record)
    result.update(
        {
            "delta_min": float(delta_min),
            "public_baseline_score": float(public_baseline_score),
            "continuity_floor": floor,
            "challenger_mean": challenger_average,
            "statistical_pass": statistical_pass,
            "continuity_pass": continuity_pass,
            "canary_pass": canary_pass,
            "crown": crown,
            "decision": decision,
        }
    )
    return result


def normal_cdf(value: float) -> float:
    """Dependency-free standard normal CDF."""
    return 0.5 * (1.0 + erf(float(value) / sqrt(2.0)))


def probability_crosses_boundary(
    true_delta: float,
    *,
    paired_sd: float,
    n: int,
    delta_min: float,
    use_alpha_spending: bool = True,
) -> float:
    """Normal-approx probability of clearing the published LCB boundary by look N."""
    sd = float(paired_sd)
    if sd < 0:
        raise ValueError("paired_sd must be non-negative")
    boundary = boundary_for_look(int(n), use_alpha_spending=use_alpha_spending)
    if sd == 0:
        return 1.0 if float(true_delta) >= float(delta_min) else 0.0
    se = sd / sqrt(int(n))
    required_observed_mean = float(delta_min) + boundary * se
    z = (required_observed_mean - float(true_delta)) / se
    return max(0.0, min(1.0, 1.0 - normal_cdf(z)))


def simulate_sequential_crown_probability(
    true_delta: float,
    *,
    paired_sd: float,
    delta_min: float,
    iterations: int = DEFAULT_OC_SIMULATION_ITERATIONS,
    seed: int = DEFAULT_OC_SIMULATION_SEED,
    use_alpha_spending: bool = True,
) -> float:
    """Seeded path simulation for crown-at-any-look probability.

    This estimates the sequential union across looks 3, 4, and 5 using
    simulated paired deltas, including the sample SD used by the actual LCB
    rule. It is deterministic for a fixed seed and iteration count, but it is
    still a Phase 0 OC input, not the final Phase-0-exit certification.
    """
    sd = float(paired_sd)
    if sd < 0.0:
        raise ValueError("paired_sd must be non-negative")
    if int(iterations) <= 0:
        raise ValueError("iterations must be positive")
    if sd == 0.0:
        return 1.0 if float(true_delta) >= float(delta_min) else 0.0

    rng = random.Random(int(seed))
    crowns = 0
    for _ in range(int(iterations)):
        path = [
            float(true_delta) + sd * _standard_normal(rng)
            for _ in range(MAX_PROBATION_LOOKS)
        ]
        for n in PROBATION_LOOKS:
            lcb = lower_confidence_bound(
                path[:n],
                use_alpha_spending=use_alpha_spending,
            )["lower_confidence_bound"]
            if lcb >= float(delta_min):
                crowns += 1
                break
    return crowns / float(iterations)


def compound_reentry_probability(
    per_attempt_probability: float,
    *,
    attempts: int = DEFAULT_REENTRY_CAP_PER_7_DAYS,
) -> float:
    """Compound independent attempt probability under the re-entry cap."""
    p = float(per_attempt_probability)
    if p < 0.0 or p > 1.0:
        raise ValueError("per_attempt_probability must be in [0, 1]")
    if int(attempts) < 1:
        raise ValueError("attempts must be positive")
    return 1.0 - (1.0 - p) ** int(attempts)


def operating_characteristic_curve(
    true_deltas: Sequence[float],
    *,
    paired_sd: float,
    delta_min: float,
    l2_entry_probability: float = 1.0,
    reentry_attempts: int = DEFAULT_REENTRY_CAP_PER_7_DAYS,
    simulation_iterations: int = DEFAULT_OC_SIMULATION_ITERATIONS,
    simulation_seed: int = DEFAULT_OC_SIMULATION_SEED,
    use_alpha_spending: bool = True,
) -> List[Dict[str, object]]:
    """Publish OC inputs at each look for supplied true deltas.

    The per-look probabilities are a normal approximation to the chance that
    the observed paired mean clears the published LCB boundary by that look.
    ``p_crown_per_attempt`` uses a seeded path simulation of the sequential
    union across looks, then applies the L2 entry probability. This is safer
    for attacker-EV inputs than using the look-5 marginal, but remains a Phase
    0 input rather than the final certification.
    """
    if not true_deltas:
        raise ValueError("at least one true_delta is required")
    entry_p = float(l2_entry_probability)
    if entry_p < 0.0 or entry_p > 1.0:
        raise ValueError("l2_entry_probability must be in [0, 1]")

    rows: List[Dict[str, object]] = []
    for index, true_delta in enumerate(true_deltas):
        looks: List[Dict[str, float]] = []
        for n in PROBATION_LOOKS:
            p_by_look = probability_crosses_boundary(
                true_delta,
                paired_sd=paired_sd,
                n=n,
                delta_min=delta_min,
                use_alpha_spending=use_alpha_spending,
            )
            looks.append(
                {
                    "n": n,
                    "boundary": boundary_for_look(n, use_alpha_spending=use_alpha_spending),
                    "p_crown_by_look": p_by_look,
                }
            )
        row_seed = int(simulation_seed) + index * 1000003
        sequential_l3_probability = simulate_sequential_crown_probability(
            true_delta,
            paired_sd=paired_sd,
            delta_min=delta_min,
            iterations=simulation_iterations,
            seed=row_seed,
            use_alpha_spending=use_alpha_spending,
        )
        final_per_attempt = entry_p * sequential_l3_probability
        rows.append(
            {
                "true_delta": float(true_delta),
                "paired_sd": float(paired_sd),
                "delta_min": float(delta_min),
                "l2_entry_probability": entry_p,
                "oc_method": "normal_marginal_by_look_plus_seeded_path_simulation",
                "looks": looks,
                "p_l3_final_look_marginal": looks[-1]["p_crown_by_look"],
                "p_l3_sequential_simulated": sequential_l3_probability,
                "simulation_iterations": int(simulation_iterations),
                "simulation_seed": row_seed,
                "p_crown_per_attempt": final_per_attempt,
                "p_crown_with_reentries": compound_reentry_probability(
                    final_per_attempt,
                    attempts=reentry_attempts,
                ),
            }
        )
    return rows


def _standard_normal(rng: random.Random) -> float:
    # Box-Muller with deterministic stdlib RNG. Clamp away from log(0).
    u1 = max(rng.random(), 1e-12)
    u2 = rng.random()
    return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)


def attacker_expected_value(
    *,
    p_crown: float,
    grant_value: float,
    loops_consumed: float,
    loop_price_usd: float = 10.0,
    probation_cost_usd: float = 0.0,
) -> Dict[str, float]:
    """Return fixed-term grant EV for a miner or attacker attempt."""
    probability = float(p_crown)
    if probability < 0.0 or probability > 1.0:
        raise ValueError("p_crown must be in [0, 1]")
    spend = float(loops_consumed) * float(loop_price_usd) + float(probation_cost_usd)
    grant = float(grant_value)
    expected_grant = probability * grant
    break_even_probability = 1.0 if grant <= 0 else min(1.0, spend / grant)
    return {
        "p_crown": probability,
        "grant_value": grant,
        "loops_consumed": float(loops_consumed),
        "loop_price_usd": float(loop_price_usd),
        "probation_cost_usd": float(probation_cost_usd),
        "attempt_cost_usd": spend,
        "expected_grant_value": expected_grant,
        "expected_net": expected_grant - spend,
        "break_even_p_crown": break_even_probability,
    }


def ceiling_utilization(
    *,
    aggregate_funded_loop_spend: float,
    total_crown_ev: float,
) -> Dict[str, float]:
    """Return aggregate spend divided by total crown EV."""
    spend = float(aggregate_funded_loop_spend)
    prize = float(total_crown_ev)
    if prize <= 0.0:
        raise ValueError("total_crown_ev must be positive")
    return {
        "aggregate_funded_loop_spend": spend,
        "total_crown_ev": prize,
        "ceiling_utilization": spend / prize,
    }


def check_grant_curve_shape(
    points: Sequence[Mapping[str, float]],
    *,
    typical_winning_spend: float,
) -> Dict[str, object]:
    """Check the fixed-term K(delta)-share curve shape for salami resistance.

    The Phase 0 economic requirement is that the grant curve is linear or
    convex, and that the smallest passing grant covers typical winning spend.
    A concave premium floor would make split/salami crowns more attractive.
    """
    if len(points) < 2:
        raise ValueError("at least two grant-curve points are required")

    sorted_points = sorted(
        (
            (float(point["delta"]), float(point["grant_value"]))
            for point in points
        ),
        key=lambda item: item[0],
    )
    if sorted_points[0][0] <= 0.0:
        raise ValueError("grant-curve deltas must be positive")

    slopes: List[float] = []
    for (delta_a, value_a), (delta_b, value_b) in zip(sorted_points, sorted_points[1:]):
        if delta_b <= delta_a:
            raise ValueError("grant-curve deltas must be strictly increasing")
        slopes.append((value_b - value_a) / (delta_b - delta_a))

    linear_or_convex = all(
        slopes[index] + 1e-12 >= slopes[index - 1]
        for index in range(1, len(slopes))
    )
    break_even_floor_ok = sorted_points[0][1] >= float(typical_winning_spend)

    return {
        "linear_or_convex": linear_or_convex,
        "break_even_floor_ok": break_even_floor_ok,
        "salami_resistant": bool(linear_or_convex and break_even_floor_ok),
        "slopes": slopes,
        "floor_grant_value": sorted_points[0][1],
        "typical_winning_spend": float(typical_winning_spend),
    }
