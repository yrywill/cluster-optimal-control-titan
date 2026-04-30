# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CountSketch random projection for gradient inner products.

Replaces an explicit projection matrix ``P ∈ R^{d×m}`` (which is tens of GB
for multi-billion-parameter models) with a pair of hash / sign tables that
require only a few MB of CPU memory.

Mathematical guarantee
----------------------
    E[<sketch(g1), sketch(g2)>] == <g1, g2>

so the sketch is an unbiased inner-product estimator.

FSDP2 / DTensor behaviour
-------------------------
Under FSDP2 each parameter's ``.grad`` is a :class:`DTensor` sharded over
the ``batch`` mesh.  CountSketch is *linear*, so

    sketch(shard_0) + sketch(shard_1) + ... == sketch(full)

provided each element is sketched with the hash/sign of its *global*
position.  To keep the implementation simple and correct we take the safe
path: before sketching we materialise each parameter's gradient as a
replicated full tensor via :meth:`DTensor.full_tensor` and sketch locally.
The trainer calls this helper *after* a ``summon_full_params``-style
context where FSDP2 has all-gathered parameters, so the extra all-gather
triggered by ``full_tensor`` only touches gradients (not parameters).

A faster "sketch-local-shard-then-all_reduce" implementation is possible
but requires per-parameter global offsets to line up the hash tables
across ranks.  We punt on it for the first version.
"""

from __future__ import annotations

from typing import Optional

import torch

from torch.distributed.tensor import DTensor, Replicate
from torchtitan.tools.logging import logger


class CountSketchProjector:
    """Streaming CountSketch over a model's ``.grad`` tensors.

    Args:
        sketch_dim: Output dimension ``m``; 8192 is the paper default.
        seed:       Base RNG seed for hash/sign tables.
    """

    def __init__(self, sketch_dim: int = 8192, seed: int = 42) -> None:
        if sketch_dim <= 0:
            raise ValueError(f"sketch_dim must be positive, got {sketch_dim}")
        self.m = int(sketch_dim)
        self.seed = int(seed)
        # Cached on CPU so GPU memory doesn't balloon.  Keyed by parameter
        # name; values shipped to the current device on demand.
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        logger.info(
            "[CountSketch] sketch_dim=%d, seed=%d", self.m, self.seed
        )

    def _get_hash_sign(
        self, name: str, numel: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lazily build and cache per-parameter hash/sign tables."""
        entry = self._cache.get(name)
        if entry is None or entry[0].numel() != numel:
            # Deterministic seed per parameter name.
            name_hash = hash(name) & 0xFFFFFFFF
            g = torch.Generator(device="cpu").manual_seed(self.seed + name_hash)
            h = torch.randint(
                0, self.m, (numel,), generator=g, dtype=torch.int64
            )
            sign = (
                torch.randint(0, 2, (numel,), generator=g, dtype=torch.float32)
                * 2
                - 1
            )
            self._cache[name] = (h, sign)
        h_cpu, sign_cpu = self._cache[name]
        return (
            h_cpu.to(device, non_blocking=True),
            sign_cpu.to(device, non_blocking=True),
        )

    @staticmethod
    def _materialize(t: torch.Tensor) -> torch.Tensor:
        """Return a local ``torch.Tensor`` for a possibly-DTensor input.

        For a sharded DTensor we use :meth:`full_tensor` with an explicit
        ``Replicate()`` grad placement — since CountSketch is linear, the
        gradient w.r.t. the replicated tensor is simply replicated too.
        """
        if isinstance(t, DTensor):
            # Grad placements match forward placements for the replicated
            # full tensor (Replicate() for both the value and its upstream
            # gradient, if any).  See .claude/rules/distributed.md.
            return t.full_tensor(grad_placements=[Replicate()])
        return t

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def sketch_grad(
        self,
        named_params: list[tuple[str, torch.nn.Parameter]],
        *,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Sketch the current ``.grad`` of the supplied parameters.

        Only parameters whose ``.grad`` is not ``None`` contribute to the
        returned sketch.
        """
        if device is None:
            for _, p in named_params:
                if p.grad is not None:
                    device = p.device
                    break
        if device is None:
            raise RuntimeError("No parameters with non-None .grad provided")

        s = torch.zeros(self.m, device=device, dtype=torch.float32)
        for name, p in named_params:
            if p.grad is None:
                continue
            g = self._materialize(p.grad)
            g_flat = g.detach().float().reshape(-1)
            h, sign = self._get_hash_sign(name, g_flat.numel(), g_flat.device)
            s.scatter_add_(0, h, g_flat * sign)
        return s

    def sketch_named_grads(
        self,
        named_grads: list[tuple[str, torch.Tensor | None]],
        *,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Sketch pre-computed per-parameter gradients.

        ``named_grads`` is typically the output of :func:`torch.autograd.grad`
        paired back with parameter names.  Entries with ``None`` gradients
        are skipped silently (consistent with ``allow_unused=True``).
        """
        if device is None:
            for _, g in named_grads:
                if g is not None:
                    device = g.device
                    break
        if device is None:
            raise RuntimeError("All supplied gradients are None")
        s = torch.zeros(self.m, device=device, dtype=torch.float32)
        for name, g in named_grads:
            if g is None:
                continue
            g_local = self._materialize(g)
            g_flat = g_local.detach().float().reshape(-1)
            h, sign = self._get_hash_sign(name, g_flat.numel(), g_flat.device)
            s.scatter_add_(0, h, g_flat * sign)
        return s

    def clear_cache(self) -> None:
        self._cache.clear()

    def memory_usage_mb(self) -> float:
        total = 0
        for (h, sign) in self._cache.values():
            total += h.element_size() * h.numel()
            total += sign.element_size() * sign.numel()
        return total / 1e6
