# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU unit tests for CountSketch.

These tests verify the two properties we actually rely on at scale:

  1. Unbiased inner-product estimate:
         E[<sketch(a), sketch(b)>] == <a, b>
  2. Additivity / linearity:
         sketch(a + b) == sketch(a) + sketch(b)

We also check determinism (same seed => same sketch) because the online
trainer re-builds hash tables lazily and must land on the same values
across ranks.
"""

from __future__ import annotations

import math

import torch

from torchtitan.experiments.cluster_data_selection.pmp.count_sketch import (
    CountSketchProjector,
)


def _fake_grads(name_to_numel: dict[str, int], seed: int):
    g = torch.Generator().manual_seed(seed)
    out: list[tuple[str, torch.Tensor | None]] = []
    for name, n in name_to_numel.items():
        out.append((name, torch.randn(n, generator=g)))
    return out


def test_sketch_linearity() -> None:
    sketcher = CountSketchProjector(sketch_dim=4096, seed=13)
    names = {"a": 7919, "b": 1024, "c": 256}
    a = _fake_grads(names, seed=1)
    b = _fake_grads(names, seed=2)
    summed = [(na, ga + gb) for (na, ga), (_, gb) in zip(a, b)]

    sa = sketcher.sketch_named_grads(a)
    sb = sketcher.sketch_named_grads(b)
    ssum = sketcher.sketch_named_grads(summed)

    # Linearity should be exact, not statistical, because CountSketch is
    # deterministic given identical hash/sign tables.
    assert torch.allclose(sa + sb, ssum, atol=1e-5), "CountSketch not linear"


def test_inner_product_unbiased() -> None:
    sketcher = CountSketchProjector(sketch_dim=8192, seed=7)
    names = {"w1": 4096, "w2": 2048}

    # Use CORRELATED vectors so the true inner product is far from zero;
    # otherwise the relative error of a sketch-based estimate is
    # ill-defined (division by ~0).
    a = _fake_grads(names, seed=10)
    noise = _fake_grads(names, seed=11)
    b = [(na, ga + 0.1 * gn) for (na, ga), (_, gn) in zip(a, noise)]

    true_ip = sum(
        float((ga * gb).sum())
        for (_, ga), (_, gb) in zip(a, b)
    )
    sa = sketcher.sketch_named_grads(a)
    sb = sketcher.sketch_named_grads(b)
    sketched_ip = float(torch.dot(sa, sb))

    # With sketch_dim=8192 and total_dim=6144 the relative error is
    # well under 10% for correlated vectors.
    rel = abs(sketched_ip - true_ip) / (abs(true_ip) + 1e-6)
    assert rel < 0.1, f"CountSketch IP too inaccurate: true={true_ip}, sketched={sketched_ip}, rel={rel}"


def test_deterministic_sketch() -> None:
    s1 = CountSketchProjector(sketch_dim=1024, seed=42)
    s2 = CountSketchProjector(sketch_dim=1024, seed=42)
    grads = _fake_grads({"layer.0.weight": 500, "layer.1.weight": 700}, seed=3)
    out1 = s1.sketch_named_grads(grads)
    out2 = s2.sketch_named_grads(grads)
    assert torch.equal(out1, out2), "CountSketch is not deterministic for same seed"


def test_sketch_skips_none() -> None:
    sketcher = CountSketchProjector(sketch_dim=512, seed=5)
    grads = [
        ("w0", torch.randn(100)),
        ("w1", None),
        ("w2", torch.randn(50)),
    ]
    sk = sketcher.sketch_named_grads(grads)
    assert sk.shape == (512,)
    assert torch.isfinite(sk).all()


if __name__ == "__main__":
    for fn in (
        test_sketch_linearity,
        test_inner_product_unbiased,
        test_deterministic_sketch,
        test_sketch_skips_none,
    ):
        fn()
        print(f"OK: {fn.__name__}")
