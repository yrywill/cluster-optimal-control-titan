"""Entry point for cluster_data_selection on PyTorch 2.5.1+cu121.

Usage (in launch script):
    torchrun ... -m torchtitan.experiments.cluster_data_selection.train_compat \
        --module cluster_data_selection --config llama3_3b_cluster_16gpu ...

This module applies the compatibility shim BEFORE any torchtitan core imports,
allowing the training to run on NVIDIA driver 535 (CUDA 12.1/12.2).
"""

# Apply compat patches FIRST — must precede any torchtitan import.
import torchtitan.experiments.cluster_data_selection.compat_shim  # noqa: F401

from torchtitan.train import main  # noqa: E402

if __name__ == "__main__":
    main()
