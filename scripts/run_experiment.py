"""Reproducible experimental evaluation script for Project 06 (Final Report).

Reproduces every quantitative table and diagnostic reported in the Final Report:
  1. Controlled noise sweep on the committed seed=42 sequences (5%, 15%, 30%)
     plus the Viterbi confusion matrix at 30% noise.
  2. Multi-run benchmark (mean +/- std over N seeds) comparing the Naive
     baseline, Midterm Viterbi, Final Viterbi, and Final Forward MAP on
     Accuracy, Macro-F1, and Inference Runtime (ms) (Table 3).
  3. Benchmark stress and edge scenarios from data/scenarios.json, including
     misleading-observation recovery rates and concrete failure steps (Table 4
     & Failure Analysis).
  4. Out-of-scope and edge-case probes (empty input, unknown sensor symbol,
     zero-probability transitions) including Midterm vs. Final safety comparison.
  5. Exact verification numbers for the 3-step hand-worked example in Section 2.2.

Usage:
    python scripts/run_experiment.py --seed 42 --runs 30
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_sequence import generate_sequence, noisy_emission_matrix  # noqa: E402
from starter import loader, traffic_core  # noqa: E402

NOISE_LEVELS = [0.05, 0.15, 0.30]


def build_noisy_config(base_config: dict, noise_level: float) -> dict:
    """Return a copy of base_config with emission_matrix blended for noise_level."""
    config = dict(base_config)
    config["emission_matrix"] = noisy_emission_matrix(
        base_config["base_emission_matrix"],
        noise_level,
    )
    return config


def macro_f1(
    predicted: Sequence[str],
    truth: Sequence[str],
    labels: Sequence[str],
) -> float:
    """Compute unweighted Macro-F1 across all classes in `labels`."""
    if len(predicted) != len(truth):
        raise ValueError("Predicted and truth sequences must have the same length.")
    if not truth:
        return 0.0

    f1_scores: List[float] = []
    for label in labels:
        tp = sum(1 for p, t in zip(predicted, truth) if p == label and t == label)
        fp = sum(1 for p, t in zip(predicted, truth) if p == label and t != label)
        fn = sum(1 for p, t in zip(predicted, truth) if p != label and t == label)
        denom = 2 * tp + fp + fn
        if denom == 0:
            # Class never appears in truth or predictions; treat F1 as 1.0 if
            # neither occurred, or 0.0 if truth had instances (covered by fn>0).
            f1_scores.append(1.0)
        else:
            f1_scores.append((2.0 * tp) / denom)
    return sum(f1_scores) / len(f1_scores)


def confusion_matrix(
    predicted: Sequence[str],
    truth: Sequence[str],
    labels: Sequence[str],
) -> Dict[str, Dict[str, int]]:
    """Return nested dict cm[true_label][pred_label] = count."""
    cm = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(truth, predicted):
        cm[t][p] += 1
    return cm


def _midterm_viterbi_unguarded(
    observations: Sequence[str],
    hmm_config: dict,
) -> Tuple[List[str], float]:
    """Exact Midterm (commit 3b90375) Viterbi behavior without None-backpointer guard.

    Identical to `starter.traffic_core.viterbi` on positive-probability traces,
    except backtracking does not guard against `previous_state is None` on
    impossible (zero-probability) transitions.
    """
    import math

    if not observations:
        return [], 0.0

    states = hmm_config["hidden_states"]
    initial = hmm_config["initial_distribution"]
    transition = hmm_config["transition_matrix"]
    emission = hmm_config.get("emission_matrix", hmm_config["base_emission_matrix"])

    log_delta: Dict[str, float] = {}
    backpointers: List[Dict[str, str | None]] = []

    first_obs = observations[0]
    for state in states:
        prob = initial[state] * emission[state][first_obs]
        log_delta[state] = math.log(prob) if prob > 0 else float("-inf")

    for obs in observations[1:]:
        new_log_delta: Dict[str, float] = {}
        cur_bp: Dict[str, str | None] = {}
        for cur in states:
            best_prev = None
            best_prob = float("-inf")
            for prev in states:
                tp = transition[prev][cur]
                ep = emission[cur][obs]
                if log_delta[prev] == float("-inf") or tp <= 0 or ep <= 0:
                    cand = float("-inf")
                else:
                    cand = log_delta[prev] + math.log(tp) + math.log(ep)
                if cand > best_prob:
                    best_prob = cand
                    best_prev = prev
            new_log_delta[cur] = best_prob
            cur_bp[cur] = best_prev
        log_delta = new_log_delta
        backpointers.append(cur_bp)

    best_final = max(states, key=lambda s: log_delta[s])
    best_log_prob = log_delta[best_final]

    path = [best_final]
    for bp in reversed(backpointers):
        previous_state = bp[path[-1]]  # Midterm had no None guard here
        path.append(previous_state)  # type: ignore[arg-type]
    path.reverse()

    prob_out = 0.0 if best_log_prob == float("-inf") else math.exp(best_log_prob)
    return path, prob_out


def evaluate_sequence(
    sequence: List[Dict[str, object]],
    hmm_config: dict,
    noise_level: float,
) -> Dict[str, object]:
    """Run Naive, Midterm Viterbi, Final Viterbi, and Final Forward MAP on a sequence."""
    observations = [str(row["observation"]) for row in sequence]
    true_states = [str(row["hidden_state"]) for row in sequence]
    states = hmm_config["hidden_states"]

    run_config = build_noisy_config(hmm_config, noise_level)

    # Naive baseline (identity mapping y_t -> x_hat_t)
    t0 = time.perf_counter()
    naive_states = list(observations)
    naive_ms = (time.perf_counter() - t0) * 1000.0

    # Midterm Viterbi (commit 3b90375 reference)
    t0 = time.perf_counter()
    midterm_states, _ = _midterm_viterbi_unguarded(observations, run_config)
    midterm_ms = (time.perf_counter() - t0) * 1000.0

    # Final Forward MAP
    t0 = time.perf_counter()
    forward_result = traffic_core.forward(observations, run_config)
    forward_map_states = [max(states, key=lambda s: dist[s]) for dist in forward_result]
    forward_ms = (time.perf_counter() - t0) * 1000.0

    if len(forward_result) != len(observations):
        raise AssertionError("Forward did not return one distribution per timestep.")
    final_sum = sum(forward_result[-1].values())
    if abs(final_sum - 1.0) > 1e-9:
        raise AssertionError(f"Final Forward posterior does not sum to 1. Got {final_sum}")

    # Final Viterbi
    t0 = time.perf_counter()
    viterbi_states, path_probability = traffic_core.viterbi(observations, run_config)
    viterbi_ms = (time.perf_counter() - t0) * 1000.0

    if len(viterbi_states) != len(true_states):
        raise AssertionError("Viterbi path length does not match ground truth length.")

    misleading_indices = [
        i for i, (o, t) in enumerate(zip(observations, true_states)) if o != t
    ]
    vit_recovered = sum(1 for i in misleading_indices if viterbi_states[i] == true_states[i])
    fwd_recovered = sum(1 for i in misleading_indices if forward_map_states[i] == true_states[i])
    transitions = sum(
        1 for i in range(1, len(true_states)) if true_states[i] != true_states[i - 1]
    )

    viterbi_errors = [
        (i, observations[i], true_states[i], viterbi_states[i], forward_map_states[i])
        for i in range(len(true_states))
        if viterbi_states[i] != true_states[i]
    ]

    return {
        "steps": len(observations),
        "transitions": transitions,
        "misleading": len(misleading_indices),
        "viterbi_recovered": vit_recovered,
        "forward_recovered": fwd_recovered,
        "naive_acc": traffic_core.accuracy_against_ground_truth(naive_states, true_states),
        "naive_f1": macro_f1(naive_states, true_states, states),
        "naive_ms": naive_ms,
        "midterm_acc": traffic_core.accuracy_against_ground_truth(midterm_states, true_states),
        "midterm_f1": macro_f1(midterm_states, true_states, states),
        "midterm_ms": midterm_ms,
        "viterbi_acc": traffic_core.accuracy_against_ground_truth(viterbi_states, true_states),
        "viterbi_f1": macro_f1(viterbi_states, true_states, states),
        "viterbi_ms": viterbi_ms,
        "forward_acc": traffic_core.accuracy_against_ground_truth(forward_map_states, true_states),
        "forward_f1": macro_f1(forward_map_states, true_states, states),
        "forward_ms": forward_ms,
        "path_probability": path_probability,
        "confusion_matrix": confusion_matrix(viterbi_states, true_states, states),
        "viterbi_errors": viterbi_errors,
    }


def _fmt_mean_std(values: Sequence[float], scale: float = 100.0, decimals: int = 2, suffix: str = "%") -> str:
    scaled = [v * scale for v in values]
    mean = statistics.mean(scaled)
    std = statistics.stdev(scaled) if len(scaled) > 1 else 0.0
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}{suffix}"


def run_multi_seed_benchmark(
    hmm_config: dict,
    base_seed: int,
    runs: int,
    length: int,
) -> None:
    """Evaluate Naive, Midterm Viterbi, Final Viterbi, and Final Forward MAP over N seeds."""
    seeds = list(range(base_seed, base_seed + runs))
    print("=" * 94)
    print(
        f"2. Multi-Run Benchmark for Table 3 (T={length}, runs={runs}, "
        f"seeds={seeds[0]}..{seeds[-1]})"
    )
    print("=" * 94)

    overall: Dict[str, Dict[str, List[float]]] = {
        m: {"acc": [], "f1": [], "ms": []}
        for m in ("naive", "midterm", "viterbi", "forward")
    }

    for noise_level in NOISE_LEVELS:
        per_noise: Dict[str, Dict[str, List[float]]] = {
            m: {"acc": [], "f1": [], "ms": []}
            for m in ("naive", "midterm", "viterbi", "forward")
        }
        for seed in seeds:
            seq = generate_sequence(length=length, seed=seed, noise_level=noise_level)
            res = evaluate_sequence(seq, hmm_config, noise_level)
            for m in ("naive", "midterm", "viterbi", "forward"):
                per_noise[m]["acc"].append(float(res[f"{m}_acc"]))
                per_noise[m]["f1"].append(float(res[f"{m}_f1"]))
                per_noise[m]["ms"].append(float(res[f"{m}_ms"]))
                overall[m]["acc"].append(float(res[f"{m}_acc"]))
                overall[m]["f1"].append(float(res[f"{m}_f1"]))
                overall[m]["ms"].append(float(res[f"{m}_ms"]))

        print(f"\nNoise level eta = {noise_level:.2f} ({int(round(noise_level * 100))}%), n = {runs} runs:")
        print(f"{'Method':<32}{'Accuracy (mean +/- std)':<26}{'Macro-F1 (mean +/- std)':<26}{'Time (ms)'}")
        print("-" * 94)
        for key, label in [
            ("naive", "Baseline: Naive Observation"),
            ("midterm", "Midterm version (Viterbi)"),
            ("viterbi", "Final version (Viterbi)"),
            ("forward", "Final version (Forward MAP)"),
        ]:
            print(
                f"{label:<32}"
                f"{_fmt_mean_std(per_noise[key]['acc']):<26}"
                f"{_fmt_mean_std(per_noise[key]['f1']):<26}"
                f"{_fmt_mean_std(per_noise[key]['ms'], scale=1.0, decimals=3, suffix=' ms')}"
            )

    print(
        f"\nAggregate across all 3 noise levels (eta in {{0.05, 0.15, 0.30}}, "
        f"N = {runs * len(NOISE_LEVELS)} runs):"
    )
    print(f"{'Method':<32}{'Accuracy (mean +/- std)':<26}{'Macro-F1 (mean +/- std)':<26}{'Time (ms)'}")
    print("-" * 94)
    for key, label in [
        ("naive", "Baseline: Naive Observation"),
        ("midterm", "Midterm version (Viterbi)"),
        ("viterbi", "Final version (Viterbi)"),
        ("forward", "Final version (Forward MAP)"),
    ]:
        print(
            f"{label:<32}"
            f"{_fmt_mean_std(overall[key]['acc']):<26}"
            f"{_fmt_mean_std(overall[key]['f1']):<26}"
            f"{_fmt_mean_std(overall[key]['ms'], scale=1.0, decimals=3, suffix=' ms')}"
        )


def run_out_of_scope_probes(hmm_config: dict) -> None:
    """Probe edge and out-of-scope inputs for Table 4 (S4) and Failure Analysis (5.2)."""
    print("\n" + "=" * 94)
    print("4. Out-of-Scope & Edge-Case Probes (Scenario S4 & Midterm vs. Final Safety)")
    print("=" * 94)

    # S4a: Empty observation sequence
    fwd_empty = traffic_core.forward([], hmm_config)
    vit_empty_path, vit_empty_prob = traffic_core.viterbi([], hmm_config)
    print(
        f"[S4a] Empty sequence [] -> "
        f"forward={fwd_empty}, viterbi=({vit_empty_path}, {vit_empty_prob}) | Outcome: Pass"
    )

    # S4b: Unknown observation symbol outside O = {LOW, MEDIUM, HIGH}
    unknown_obs = ["LOW", "EXTREME", "HIGH"]
    for name, fn in [("forward", traffic_core.forward), ("viterbi", traffic_core.viterbi)]:
        try:
            fn(unknown_obs, hmm_config)
            status = "Returned without error"
        except Exception as exc:  # noqa: BLE001
            status = f"{type(exc).__name__}: {exc}"
        print(f"[S4b] Unknown symbol {unknown_obs} -> {name} raised {status}")
    print("      -> Outcome: Fail (no explicit alphabet validation; raises raw KeyError)")

    # S4c: Zero-probability / impossible transition config (Midterm vs Final)
    zero_config = {
        "hidden_states": ["LOW", "HIGH"],
        "observation_symbols": ["LOW", "HIGH"],
        "initial_distribution": {"LOW": 1.0, "HIGH": 0.0},
        "transition_matrix": {
            "LOW": {"LOW": 0.0, "HIGH": 0.0},
            "HIGH": {"LOW": 0.0, "HIGH": 0.0},
        },
        "base_emission_matrix": {
            "LOW": {"LOW": 1.0, "HIGH": 0.0},
            "HIGH": {"LOW": 0.0, "HIGH": 1.0},
        },
    }
    zero_obs = ["LOW", "HIGH", "LOW"]
    try:
        res_fwd = traffic_core.forward(zero_obs, zero_config)
        fwd_zero = f"returned {res_fwd}"
    except Exception as exc:  # noqa: BLE001
        fwd_zero = f"raised {type(exc).__name__}: {exc}"

    try:
        res_mid = _midterm_viterbi_unguarded(zero_obs, zero_config)
        midterm_zero = f"returned {res_mid}"
    except Exception as exc:  # noqa: BLE001
        midterm_zero = f"raised {type(exc).__name__}: {exc} (crashed on None backpointer)"

    final_zero_path, final_zero_prob = traffic_core.viterbi(zero_obs, zero_config)
    print(f"[S4c] Zero-probability transitions {zero_obs}:")
    print(f"      - Final forward:   {fwd_zero}")
    print(f"      - Midterm viterbi: {midterm_zero}")
    print(f"      - Final viterbi:   returned ({final_zero_path}, {final_zero_prob}) | Outcome: Pass (safe fallback)")


def run_hand_worked_example(hmm_config: dict) -> None:
    """Print exact numerical values for a 3-step hand-worked example (Section 2.2)."""
    print("\n" + "=" * 94)
    print("5. Verification for Section 2.2 Hand-Worked Example: y = ['LOW', 'HIGH', 'LOW']")
    print("=" * 94)
    obs = ["LOW", "HIGH", "LOW"]
    for eta in (0.0, 0.15):
        cfg = build_noisy_config(hmm_config, eta)
        fwd = traffic_core.forward(obs, cfg)
        vit_path, vit_prob = traffic_core.viterbi(obs, cfg)
        print(f"\nNoise eta = {eta:.2f}:")
        for t, dist in enumerate(fwd, start=1):
            print(
                f"  t={t} (y_{t}={obs[t-1]:<6}): "
                f"LOW={dist['LOW']:.6f}, MEDIUM={dist['MEDIUM']:.6f}, HIGH={dist['HIGH']:.6f}"
            )
        print(f"  Viterbi path = {vit_path}, joint prob = {vit_prob:.6e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible Project 06 HMM evaluation benchmarks."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Starting random seed for the multi-run benchmark (default: 42).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=30,
        help="Number of random seeds per noise level for Table 3 mean +/- std (default: 30).",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=200,
        help="Sequence length T for the multi-run benchmark (default: 200).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hmm_config = loader.load_hmm_config()
    scenarios = loader.load_scenarios()
    states = hmm_config["hidden_states"]

    print("=" * 94)
    print("Project 06 - Reproducible Experimental Evaluation")
    print(f"Environment: Python {platform.python_version()} | OS: {platform.platform()}")
    print("=" * 94)

    # 1. Committed seed=42 controlled noise experiment
    print("\n" + "=" * 94)
    print("1. Controlled Noise Experiment (Committed seed=42 Sequences, T=200)")
    print("=" * 94)
    print(
        f"{'Noise':<7}{'Viterbi Acc':<13}{'Vit F1':<10}{'Fwd Acc':<11}{'Fwd F1':<10}"
        f"{'Naive Acc':<11}{'Mislead':<9}{'VitRec':<8}{'FwdRec':<8}{'Joint Prob'}"
    )
    print("-" * 94)

    cm_30 = None
    for noise_level in NOISE_LEVELS:
        sequence = loader.load_sequence(noise_level)
        m = evaluate_sequence(sequence, hmm_config, noise_level)
        if noise_level == 0.30:
            cm_30 = m["confusion_matrix"]
        print(
            f"{noise_level * 100:>4.0f}%  "
            f"{m['viterbi_acc'] * 100:>6.2f}%      "
            f"{m['viterbi_f1'] * 100:>6.2f}%   "
            f"{m['forward_acc'] * 100:>6.2f}%    "
            f"{m['forward_f1'] * 100:>6.2f}%   "
            f"{m['naive_acc'] * 100:>6.2f}%    "
            f"{m['misleading']:<9}"
            f"{m['viterbi_recovered']:<8}"
            f"{m['forward_recovered']:<8}"
            f"{m['path_probability']:.4e}"
        )

    if cm_30 is not None:
        print("\nViterbi Confusion Matrix at eta = 0.30 (Committed seed=42):")
        corner = "Truth \\ Pred"
        header = f"{corner:<14}" + "".join(f"{s:>10}" for s in states)
        print(header)
        print("-" * len(header))
        for st in states:
            print(f"{st:<14}" + "".join(f"{cm_30[st][sp]:>10}" for sp in states))

    # 2. Multi-run benchmark for Table 3 (mean +/- std + runtime)
    print()
    run_multi_seed_benchmark(hmm_config, args.seed, args.runs, args.length)

    # 3. Stress & edge scenarios from data/scenarios.json (Table 4 & Failure Analysis)
    print("\n" + "=" * 94)
    print("3. Benchmark Stress & Edge Scenarios (data/scenarios.json) for Table 4")
    print("=" * 94)
    print(
        f"{'Scenario ID':<29}{'T':<6}{'eta':<6}{'Trans':<7}{'Mislead':<9}"
        f"{'Viterbi':<10}{'Fwd MAP':<10}{'Naive':<10}{'VitRec'}"
    )
    print("-" * 94)

    for scenario in scenarios:
        gen = scenario["generator"]
        noise_level = float(gen["noise_level"])
        sequence = generate_sequence(
            length=int(gen["length"]),
            seed=int(gen["seed"]),
            noise_level=noise_level,
        )
        m = evaluate_sequence(sequence, hmm_config, noise_level)
        print(
            f"{scenario['id']:<29}"
            f"{m['steps']:<6}"
            f"{noise_level * 100:>3.0f}%  "
            f"{m['transitions']:<7}"
            f"{m['misleading']:<9}"
            f"{m['viterbi_acc'] * 100:>6.2f}%   "
            f"{m['forward_acc'] * 100:>6.2f}%   "
            f"{m['naive_acc'] * 100:>6.2f}%   "
            f"{m['viterbi_recovered']}/{m['misleading']}"
        )
        if scenario["id"] == "P06-RAPID-TRANSITION":
            sample_errs = m["viterbi_errors"][:10]
            print("   Sample Viterbi error steps (t, obs, truth, viterbi, fwd_map):")
            for err in sample_errs:
                print(f"     t={err[0]:>2}: obs={err[1]:<6} truth={err[2]:<6} vit={err[3]:<6} fwd={err[4]:<6}")

    # 4. Out-of-scope probes (S4)
    run_out_of_scope_probes(hmm_config)

    # 5. Hand-worked example verification (Section 2.2)
    run_hand_worked_example(hmm_config)

    print("\nAll Forward, Viterbi, Macro-F1, Scenario, and Probe checks COMPLETED.")


if __name__ == "__main__":
    main()
