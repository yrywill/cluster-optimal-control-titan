# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cluster-based data selection experiment.

Ports the PMP (Perturbation-based Meta-Policy) cluster-weight update loop on
top of torchtitan, keeping torchtitan's native FSDP2/HSDP multi-node
multi-GPU machinery unchanged.  See ``README.md`` for design & usage.

We intentionally do NOT eagerly import ``ClusterSelectionTrainer`` here —
that drags in ``torchtitan.trainer`` → ``tyro``, which is overkill for
offline utility scripts (``scripts.prepare_clusters``,
``scripts.download_dclm``, ...).  Users who need the trainer import it
explicitly::

    from torchtitan.experiments.cluster_data_selection.trainer import (
        ClusterSelectionTrainer,
    )
"""

