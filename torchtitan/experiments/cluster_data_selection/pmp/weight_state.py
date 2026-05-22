# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cluster-weight bookkeeping.

``grad_gamma`` is the accumulator of per-cluster contributions.  After each
PMP step we convert it into a sampling distribution by

    w_k ∝ exp( grad_gamma[k] / τ )

where positive grad_gamma means the cluster is beneficial for the dev
objective — so it receives higher sampling weight.  An optional floor
``min_weight`` prevents starvation, and ``drop_bad_clusters`` permanently
removes clusters that consistently hurt validation.

A decay factor ``gamma_decay`` (default 1.0 = no decay) applies exponential
discounting on historical grad_gamma each PMP step:

    grad_gamma = gamma_decay * grad_gamma + delta

This makes recent PMP signals weigh more than stale ones, allowing the
distribution to adapt as training progresses and cluster utility changes.

The state lives in a simple dataclass so it is trivially picklable /
stateful — the trainer persists it through torchtitan's DCP mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from torchtitan.tools.logging import logger


@dataclass
class ClusterWeightState:
    """All PMP sampling state in one place so the trainer can checkpoint it."""

    num_clusters: int
    temperature: float = 0.5
    min_weight: float = 0.01
    accumulate: bool = True
    gamma_decay: float = 1.0
    drop_bad_clusters: bool = False
    drop_patience: int = 5

    # Filled in by __post_init__.
    grad_gamma: np.ndarray = None  # type: ignore[assignment]
    weights: np.ndarray = None  # type: ignore[assignment]
    negative_streak: np.ndarray = None  # type: ignore[assignment]
    dead: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        K = int(self.num_clusters)
        self.grad_gamma = np.zeros(K, dtype=np.float64)
        # Uniform initial weights.
        self.weights = np.ones(K, dtype=np.float64) / max(K, 1)
        self.negative_streak = np.zeros(K, dtype=np.int32)
        self.dead = np.zeros(K, dtype=bool)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self, grad_gamma_delta: torch.Tensor | np.ndarray) -> None:
        """Apply one PMP delta and recompute weights."""
        if isinstance(grad_gamma_delta, torch.Tensor):
            delta = grad_gamma_delta.detach().cpu().double().numpy()
        else:
            delta = np.asarray(grad_gamma_delta, dtype=np.float64)
        if delta.shape != (self.num_clusters,):
            raise ValueError(
                f"delta shape {delta.shape} != ({self.num_clusters},)"
            )

        if self.accumulate:
            # Exponential decay on historical gamma before adding new delta.
            # decay=1.0 means no decay (pure accumulation, original behavior).
            # decay<1.0 (e.g. 0.9) discounts stale signals so recent PMP
            # updates dominate — useful when cluster utility shifts over time.
            self.grad_gamma = self.gamma_decay * self.grad_gamma + delta
        else:
            self.grad_gamma = delta.copy()

        # Dead-cluster tracking from the raw delta.
        if self.drop_bad_clusters:
            newly_dropped = 0
            for k in range(self.num_clusters):
                if self.dead[k]:
                    continue
                if delta[k] < 0:
                    self.negative_streak[k] += 1
                elif delta[k] > 0:
                    self.negative_streak[k] = 0
                if self.negative_streak[k] >= self.drop_patience:
                    self.dead[k] = True
                    newly_dropped += 1
            if newly_dropped:
                logger.info(
                    "[ClusterWeight] dropped %d clusters (total dead=%d)",
                    newly_dropped,
                    int(self.dead.sum()),
                )

        # Softmax weights.
        # Positive grad_gamma → cluster is beneficial → higher weight.
        logits = self.grad_gamma / max(self.temperature, 1e-6)
        logits -= logits.max()
        w = np.exp(logits)
        w = np.clip(w, a_min=self.min_weight, a_max=None)
        if self.drop_bad_clusters:
            w[self.dead] = 0.0
        total = w.sum()
        if total <= 0.0:
            alive = (~self.dead).astype(np.float64)
            if alive.sum() == 0:
                alive = np.ones_like(w)
            w = alive / alive.sum()
        else:
            w = w / total
        self.weights = w

        logger.info(
            "[ClusterWeight] updated: min=%.4e max=%.4e dead=%d/%d",
            float(w.min()),
            float(w.max()),
            int(self.dead.sum()),
            self.num_clusters,
        )

    # ------------------------------------------------------------------
    # Checkpoint interop
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "grad_gamma": self.grad_gamma.tolist(),
            "weights": self.weights.tolist(),
            "negative_streak": self.negative_streak.tolist(),
            "dead": self.dead.tolist(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if not state_dict:
            return
        self.grad_gamma = np.asarray(state_dict["grad_gamma"], dtype=np.float64)
        self.weights = np.asarray(state_dict["weights"], dtype=np.float64)
        self.negative_streak = np.asarray(
            state_dict["negative_streak"], dtype=np.int32
        )
        self.dead = np.asarray(state_dict["dead"], dtype=bool)
