"""Entry point for cluster_data_selection training (PyTorch 2.5 compatible).

Imports the compat shim FIRST to patch missing APIs, then runs the trainer.
"""

# Must be first import — patches torch APIs for PyTorch 2.5.x
import torchtitan.experiments.cluster_data_selection.compat_shim  # noqa: F401

from torchtitan.experiments.cluster_data_selection.trainer import (
    ClusterSelectionTrainer,
)


def main():
    trainer = ClusterSelectionTrainer(ClusterSelectionTrainer.build_config())
    trainer.train()


if __name__ == "__main__":
    main()
