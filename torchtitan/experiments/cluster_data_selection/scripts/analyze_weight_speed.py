"""Analyze PMP weight update dynamics at different batch sizes.

This script doesn't require GPUs — it simulates the PMP weight update
mechanism to estimate whether batch=1024 causes weights to shift too fast.

Key insight: PMP's grad_gamma_delta is computed from FIXED-SIZE mini-batches
(dev_batch=8, n_samples_per_cluster=4), NOT from the training global batch.
The training batch size only affects how much the MODEL changes between PMP
triggers.

Usage:
    python torchtitan/experiments/cluster_data_selection/scripts/analyze_weight_speed.py
"""

import numpy as np


def simulate_pmp_weight_updates(
    num_clusters: int = 30,
    pmp_lr: float = 0.01,
    temperature: float = 1.0,
    min_weight: float = 0.01,
    num_pmp_steps: int = 38,         # 38000 steps / 1000 interval
    accumulate: bool = True,
    # Simulate different gradient signal strengths
    # (larger batch → model converges more between PMPs → different signal)
    gradient_signal_scale: float = 1.0,
    seed: int = 42,
):
    """Simulate PMP weight trajectory.

    The grad_gamma_delta[k] ≈ pmp_lr * <sketch(∇L_dev), sketch(∇L_k)>.

    In practice, <sketch(∇L_dev), sketch(∇L_k)> has:
    - Some clusters positively correlated (helpful) → positive dot product
    - Some clusters negatively correlated (harmful) → negative dot product
    - Most clusters near zero (neutral)

    We simulate this with a random distribution that becomes more
    concentrated as training progresses (model learns what's useful).
    """
    rng = np.random.default_rng(seed)

    # Initialize: size-proportional weights (typical for bucketed data)
    # Simulate cluster sizes with Zipf-like distribution
    cluster_sizes = rng.pareto(1.5, num_clusters) + 1
    cluster_sizes = cluster_sizes / cluster_sizes.sum()
    weights = cluster_sizes.copy()

    grad_gamma = np.zeros(num_clusters, dtype=np.float64)

    # Simulate "ground truth" utility of each cluster
    # Some clusters are genuinely helpful, some harmful, most neutral
    true_utility = rng.normal(0, 0.3, num_clusters)
    # A few clusters are clearly good/bad
    true_utility[rng.choice(num_clusters, 3, replace=False)] = rng.uniform(0.5, 1.0, 3)
    true_utility[rng.choice(num_clusters, 3, replace=False)] = rng.uniform(-1.0, -0.5, 3)

    history = [{"step": 0, "weights": weights.copy(), "grad_gamma": grad_gamma.copy()}]

    for t in range(1, num_pmp_steps + 1):
        # Simulate grad_gamma_delta:
        # <sketch(∇L_dev), sketch(∇L_k)> ≈ true_utility[k] + noise
        # Noise decreases as model trains more (signal gets clearer)
        noise_level = 0.2 / (1 + 0.1 * t)  # Signal-to-noise improves over time
        delta = pmp_lr * (
            true_utility * gradient_signal_scale
            + rng.normal(0, noise_level, num_clusters)
        )

        if accumulate:
            grad_gamma += delta
        else:
            grad_gamma = delta.copy()

        # Softmax weights (same as ClusterWeightState.update)
        logits = -grad_gamma / max(temperature, 1e-6)
        logits -= logits.max()
        w = np.exp(logits)
        w = np.clip(w, a_min=min_weight, a_max=None)
        w = w / w.sum()
        weights = w

        history.append({
            "step": t,
            "weights": weights.copy(),
            "grad_gamma": grad_gamma.copy(),
        })

    return history, true_utility


def print_analysis(history, label, true_utility):
    print(f"\n{'='*70}")
    print(f" {label}")
    print(f"{'='*70}")

    K = len(history[0]["weights"])
    init_w = history[0]["weights"]
    final_w = history[-1]["weights"]
    num_steps = len(history) - 1

    print(f"  Clusters: {K}, PMP steps: {num_steps}")
    print(f"\n  Weight distribution over time:")
    print(f"  {'PMP step':>10s} | {'min(w)':>10s} | {'max(w)':>10s} | {'max/min':>10s} | {'std(w)':>10s} | {'top5_share':>10s}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    check_steps = [0, 1, 2, 5, 10, 19, 28, 38]
    for s in check_steps:
        if s >= len(history):
            continue
        w = history[s]["weights"]
        ratio = max(w) / max(min(w), 1e-10)
        top5 = sum(sorted(w, reverse=True)[:5])
        print(f"  {s:>10d} | {min(w):>10.4e} | {max(w):>10.4e} | {ratio:>10.1f}x | {np.std(w):>10.4e} | {top5:>10.2%}")

    # Total drift
    drift = np.linalg.norm(final_w - init_w)
    print(f"\n  Total L2 drift (init→final): {drift:.6f}")
    print(f"  Entropy: init={-np.sum(init_w * np.log(init_w+1e-10)):.3f}, "
          f"final={-np.sum(final_w * np.log(final_w+1e-10)):.3f} "
          f"(uniform={np.log(K):.3f})")

    # Correlation with true utility
    corr = np.corrcoef(final_w, true_utility)[0, 1]
    print(f"  Correlation(final_weights, true_utility): {corr:.3f}")

    # Detect problematic behavior
    final_ratio = max(final_w) / max(min(final_w), 1e-10)
    if final_ratio > 100:
        print(f"\n  [DANGER] Final max/min ratio = {final_ratio:.0f}x — weights collapsed!")
        print(f"  Some clusters effectively starved. Training may be unstable.")
    elif final_ratio > 20:
        print(f"\n  [WARNING] Final max/min ratio = {final_ratio:.0f}x — aggressive.")
        print(f"  Consider increasing temperature or reducing pmp_lr.")
    elif final_ratio > 5:
        print(f"\n  [OK] Final max/min ratio = {final_ratio:.1f}x — moderate differentiation.")
    else:
        print(f"\n  [OK] Final max/min ratio = {final_ratio:.1f}x — conservative.")


def main():
    print("=" * 70)
    print(" PMP Weight Update Speed Analysis")
    print(" (Simulated, no GPU needed)")
    print("=" * 70)

    print("""
  Background:
  - PMP computes: grad_gamma[k] += pmp_lr * <sketch(∇L_dev), sketch(∇L_k)>
  - The gradient sketches use FIXED batch sizes (dev_batch=8, n_samples=4)
  - Training global batch does NOT directly affect PMP delta magnitude
  - BUT: larger batch → model changes more between PMP → clearer signal

  Scenarios to compare:
    A) batch=256,  interval=1000 → 38 PMP triggers in 38000 steps
    B) batch=1024, interval=1000 → 38 PMP triggers in 38000 steps
       (model sees 4x more tokens between PMPs)
    """)

    # =========================================================
    # Scenario A: batch=256, pmp_lr=0.01, temperature=1
    # =========================================================
    hist_a, utility = simulate_pmp_weight_updates(
        num_clusters=30,
        pmp_lr=0.01,
        temperature=1.0,
        num_pmp_steps=38,
        gradient_signal_scale=1.0,  # normal signal
        seed=42,
    )
    print_analysis(hist_a, "Scenario A: batch=256, lr=0.01, temp=1.0, interval=1000", utility)

    # =========================================================
    # Scenario B: batch=1024 → model learns 4x faster
    # Signal is ~1.5x stronger (model is more trained, gradient is more aligned)
    # =========================================================
    hist_b, _ = simulate_pmp_weight_updates(
        num_clusters=30,
        pmp_lr=0.01,
        temperature=1.0,
        num_pmp_steps=38,
        gradient_signal_scale=1.5,  # stronger signal (more trained model)
        seed=42,
    )
    print_analysis(hist_b, "Scenario B: batch=1024, lr=0.01, temp=1.0, interval=1000 (4x more tokens)", utility)

    # =========================================================
    # Scenario C: batch=1024 with ADJUSTED hyperparameters
    # Option 1: lower pmp_lr
    # =========================================================
    hist_c, _ = simulate_pmp_weight_updates(
        num_clusters=30,
        pmp_lr=0.005,  # halved
        temperature=1.0,
        num_pmp_steps=38,
        gradient_signal_scale=1.5,
        seed=42,
    )
    print_analysis(hist_c, "Scenario C: batch=1024, lr=0.005 (halved), temp=1.0", utility)

    # =========================================================
    # Scenario D: batch=1024 with higher temperature
    # =========================================================
    hist_d, _ = simulate_pmp_weight_updates(
        num_clusters=30,
        pmp_lr=0.01,
        temperature=2.0,  # doubled
        num_pmp_steps=38,
        gradient_signal_scale=1.5,
        seed=42,
    )
    print_analysis(hist_d, "Scenario D: batch=1024, lr=0.01, temp=2.0 (doubled)", utility)

    # =========================================================
    # Scenario E: batch=1024 with larger interval
    # =========================================================
    hist_e, _ = simulate_pmp_weight_updates(
        num_clusters=30,
        pmp_lr=0.01,
        temperature=1.0,
        num_pmp_steps=19,  # half as many PMP steps (interval=2000)
        gradient_signal_scale=1.5,
        seed=42,
    )
    print_analysis(hist_e, "Scenario E: batch=1024, lr=0.01, temp=1.0, interval=2000 (halved PMPs)", utility)

    # =========================================================
    # Recommendations
    # =========================================================
    print(f"\n{'='*70}")
    print(f" RECOMMENDATIONS for batch=1024")
    print(f"{'='*70}")

    final_ratio_a = max(hist_a[-1]["weights"]) / max(min(hist_a[-1]["weights"]), 1e-10)
    final_ratio_b = max(hist_b[-1]["weights"]) / max(min(hist_b[-1]["weights"]), 1e-10)
    final_ratio_c = max(hist_c[-1]["weights"]) / max(min(hist_c[-1]["weights"]), 1e-10)
    final_ratio_d = max(hist_d[-1]["weights"]) / max(min(hist_d[-1]["weights"]), 1e-10)
    final_ratio_e = max(hist_e[-1]["weights"]) / max(min(hist_e[-1]["weights"]), 1e-10)

    print(f"""
  Summary of final max/min weight ratios:
    A) batch=256,  lr=0.01, temp=1, int=1000:   {final_ratio_a:>6.1f}x
    B) batch=1024, lr=0.01, temp=1, int=1000:   {final_ratio_b:>6.1f}x  ← current concern
    C) batch=1024, lr=0.005, temp=1, int=1000:  {final_ratio_c:>6.1f}x  ← halve lr
    D) batch=1024, lr=0.01, temp=2, int=1000:   {final_ratio_d:>6.1f}x  ← double temp
    E) batch=1024, lr=0.01, temp=1, int=2000:   {final_ratio_e:>6.1f}x  ← double interval

  Key findings:
  1. PMP delta magnitude does NOT directly scale with training batch size.
     The gradient is from fixed-size PMP batches (dev=8, cluster=4).

  2. Indirect effect: with batch=1024, the model trains 4x faster between
     PMP updates. This means the gradient SIGNAL is stronger (model better
     separates useful vs harmful clusters), but the MAGNITUDE per step is
     similar.

  3. If ratio > 20x is too aggressive for your use case, recommended fixes
     (in order of preference):
     a) Increase temperature: 1.0 → 2.0 (smooths the softmax, most robust)
     b) Reduce pmp_lr: 0.01 → 0.005 (directly slows accumulation)
     c) Increase interval: 1000 → 2000 (fewer updates, but loses reactivity)
    """)


if __name__ == "__main__":
    main()
