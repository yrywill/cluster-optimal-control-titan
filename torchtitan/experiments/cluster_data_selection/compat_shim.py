"""PyTorch 2.5.1+cu121 compatibility shim for cluster_data_selection.

This module patches sys.modules and monkey-patches APIs that don't exist in
PyTorch 2.5.1 but are imported at module level by torchtitan core.

MUST be imported BEFORE any torchtitan module (i.e., first line in the entry
point).  Only stubs features UNUSED by the cluster_data_selection path:
  - HF checkpoint saving (last_save_in_hf=False)
  - Async checkpointing (mode="disabled")
  - Context Parallelism (cp=1)
  - VarlenAttention (attn_backend="sdpa")
  - Flash attention impl selection

The one ACTIVE code-path fix: wraps sdpa_kernel to drop the `set_priority`
kwarg which was added in PyTorch 2.6.

Safety: every stub raises RuntimeError if actually called at runtime.
"""

import functools
import importlib
import sys
import types

import torch


def _version_tuple():
    """Return (major, minor, patch) from torch.__version__."""
    ver = torch.__version__.split("+")[0]
    parts = ver.split(".")
    return tuple(int(p) for p in parts[:3])


_TORCH_VERSION = _version_tuple()

# Only apply shim if torch < 2.6
if _TORCH_VERSION >= (2, 6, 0):
    pass  # Nothing to do, all APIs exist
else:
    # ================================================================
    # Helper: create a stub that raises if called
    # ================================================================
    def _make_stub(name):
        """Create a callable stub that raises RuntimeError if invoked."""

        def _stub(*args, **kwargs):
            raise RuntimeError(
                f"{name} requires PyTorch >= 2.6. "
                f"Current version: {torch.__version__}. "
                f"This code path should not be reached with the "
                f"cluster_data_selection configuration."
            )

        _stub.__name__ = name
        _stub.__qualname__ = name
        return _stub

    def _make_stub_class(name):
        """Create a stub class that raises on instantiation."""

        def _init(self, *args, **kwargs):
            raise RuntimeError(
                f"{name} requires PyTorch >= 2.6. "
                f"Current version: {torch.__version__}."
            )

        cls = type(name, (), {"__init__": _init})
        return cls

    # ================================================================
    # Patch 1: torch.distributed.checkpoint — HF + async APIs
    # ================================================================
    import torch.distributed.checkpoint as _dcp

    # HuggingFaceStorageWriter
    if not hasattr(_dcp, "HuggingFaceStorageWriter"):
        _dcp.HuggingFaceStorageWriter = _make_stub_class(
            "HuggingFaceStorageWriter"
        )

    # HuggingFaceStorageReader
    if not hasattr(_dcp, "HuggingFaceStorageReader"):
        _dcp.HuggingFaceStorageReader = _make_stub_class(
            "HuggingFaceStorageReader"
        )

    # _consolidate_hf_safetensors module
    _consolidate_mod_name = (
        "torch.distributed.checkpoint._consolidate_hf_safetensors"
    )
    if _consolidate_mod_name not in sys.modules:
        _mod = types.ModuleType(_consolidate_mod_name)
        _mod.consolidate_safetensors_files_on_every_rank = _make_stub(
            "consolidate_safetensors_files_on_every_rank"
        )
        sys.modules[_consolidate_mod_name] = _mod

    # AsyncCheckpointerType, AsyncSaveResponse in state_dict_saver
    try:
        from torch.distributed.checkpoint.state_dict_saver import (
            AsyncCheckpointerType,  # noqa: F401
        )
    except ImportError:
        import torch.distributed.checkpoint.state_dict_saver as _sds

        _sds.AsyncCheckpointerType = _make_stub_class("AsyncCheckpointerType")
        _sds.AsyncSaveResponse = _make_stub_class("AsyncSaveResponse")

    # DefaultStager, StagingOptions in staging
    _staging_mod_name = "torch.distributed.checkpoint.staging"
    try:
        from torch.distributed.checkpoint.staging import (  # noqa: F401
            DefaultStager,
        )
    except (ImportError, ModuleNotFoundError):
        _staging_mod = types.ModuleType(_staging_mod_name)
        _staging_mod.DefaultStager = _make_stub_class("DefaultStager")
        _staging_mod.StagingOptions = _make_stub_class("StagingOptions")
        sys.modules[_staging_mod_name] = _staging_mod

    # ================================================================
    # Patch 2: torch.distributed.tensor.experimental._attention (CP)
    # ================================================================
    _attn_exp_mod_name = "torch.distributed.tensor.experimental._attention"
    if _attn_exp_mod_name not in sys.modules:
        _attn_mod = types.ModuleType(_attn_exp_mod_name)
        _attn_mod._context_parallel_shard = _make_stub(
            "_context_parallel_shard"
        )
        _attn_mod._ContextParallel = _make_stub_class("_ContextParallel")
        _attn_mod._enable_context_parallel_dispatcher = _make_stub(
            "_enable_context_parallel_dispatcher"
        )
        _attn_mod._HeadTailLoadBalancer = _make_stub_class(
            "_HeadTailLoadBalancer"
        )
        _attn_mod._PTRRLoadBalancer = _make_stub_class("_PTRRLoadBalancer")
        sys.modules[_attn_exp_mod_name] = _attn_mod

    # ================================================================
    # Patch 3: torch.nn.attention — new APIs in 2.6
    # ================================================================
    import torch.nn.attention as _nn_attn

    # activate_flash_attention_impl / current_flash_attention_impl
    if not hasattr(_nn_attn, "activate_flash_attention_impl"):
        # No-op context manager stub
        import contextlib

        @contextlib.contextmanager
        def _activate_flash_attention_impl(*args, **kwargs):
            yield

        _nn_attn.activate_flash_attention_impl = _activate_flash_attention_impl

    if not hasattr(_nn_attn, "current_flash_attention_impl"):
        _nn_attn.current_flash_attention_impl = lambda: None

    # Wrap sdpa_kernel to accept and ignore `set_priority` kwarg
    _original_sdpa_kernel = _nn_attn.sdpa_kernel

    @functools.wraps(_original_sdpa_kernel)
    def _patched_sdpa_kernel(*args, **kwargs):
        kwargs.pop("set_priority", None)
        return _original_sdpa_kernel(*args, **kwargs)

    _nn_attn.sdpa_kernel = _patched_sdpa_kernel

    # ================================================================
    # Patch 4: torch.nn.attention.flex_attention — AuxRequest
    # ================================================================
    _flex_mod_name = "torch.nn.attention.flex_attention"
    _flex_mod = sys.modules.get(_flex_mod_name)
    if _flex_mod is None:
        try:
            _flex_mod = importlib.import_module(_flex_mod_name)
        except (ImportError, ModuleNotFoundError):
            _flex_mod = types.ModuleType(_flex_mod_name)
            sys.modules[_flex_mod_name] = _flex_mod

    if not hasattr(_flex_mod, "AuxRequest"):
        _flex_mod.AuxRequest = _make_stub_class("AuxRequest")

    # Ensure and_masks exists (it should in 2.5.1, but be safe)
    if not hasattr(_flex_mod, "and_masks"):
        _flex_mod.and_masks = _make_stub("and_masks")

    # ================================================================
    # Patch 5: torch.nn.attention.varlen module
    # ================================================================
    _varlen_mod_name = "torch.nn.attention.varlen"
    if _varlen_mod_name not in sys.modules:
        _varlen_mod = types.ModuleType(_varlen_mod_name)
        _varlen_mod.varlen_attn = _make_stub("varlen_attn")
        sys.modules[_varlen_mod_name] = _varlen_mod

    # ================================================================
    # Patch 6: torch.distributed.fsdp — re-export composable FSDP2 APIs
    # ================================================================
    # In 2.5.1 these live in torch.distributed._composable.fsdp, but
    # torchtitan imports them from torch.distributed.fsdp (added in 2.6).
    import torch.distributed.fsdp as _fsdp_mod

    if not hasattr(_fsdp_mod, "fully_shard"):
        from torch.distributed._composable.fsdp import (
            CPUOffloadPolicy as _CPUOffloadPolicy,
            fully_shard as _fully_shard,
            MixedPrecisionPolicy as _MixedPrecisionPolicy,
        )

        _fsdp_mod.fully_shard = _fully_shard
        _fsdp_mod.CPUOffloadPolicy = _CPUOffloadPolicy
        _fsdp_mod.MixedPrecisionPolicy = _MixedPrecisionPolicy

    # ================================================================
    # Patch 7: torch.distributed.pipelining.schedules — new schedules
    # ================================================================
    try:
        import torch.distributed.pipelining.schedules as _pp_sched

        if not hasattr(_pp_sched, "ScheduleDualPipeV"):
            _pp_sched.ScheduleDualPipeV = _make_stub_class(
                "ScheduleDualPipeV"
            )
        if not hasattr(_pp_sched, "ScheduleZBVZeroBubble"):
            _pp_sched.ScheduleZBVZeroBubble = _make_stub_class(
                "ScheduleZBVZeroBubble"
            )
        if not hasattr(_pp_sched, "get_schedule_class"):
            _pp_sched.get_schedule_class = _make_stub("get_schedule_class")
        if not hasattr(_pp_sched, "_PipelineScheduleRuntime"):
            _pp_sched._PipelineScheduleRuntime = _make_stub_class(
                "_PipelineScheduleRuntime"
            )
    except ImportError:
        pass

    # ================================================================
    # Patch 7: torch.compile — filter out unknown inductor options
    # ================================================================
    # PyTorch 2.5.1 doesn't support 'wrap_inductor_compiled_regions' in
    # inductor options. The FlexAttention class uses it at class-body level.
    # We wrap torch.compile to silently strip unknown options.
    _original_torch_compile = torch.compile

    # Get known options from inductor config to filter against
    try:
        from torch._inductor import config as _inductor_config

        _known_options = set(dir(_inductor_config))
    except ImportError:
        _known_options = None

    @functools.wraps(_original_torch_compile)
    def _patched_torch_compile(*args, **kwargs):
        options = kwargs.get("options")
        if options and isinstance(options, dict) and _known_options is not None:
            filtered = {
                k: v for k, v in options.items() if k in _known_options
            }
            kwargs["options"] = filtered
        return _original_torch_compile(*args, **kwargs)

    torch.compile = _patched_torch_compile

    # ================================================================
    # Patch 8: torch.distributed.init_process_group — strip _ranks kwarg
    # ================================================================
    # PyTorch 2.6 added `_ranks` parameter to init_process_group.
    # torchtitan core passes it unconditionally, so we strip it for 2.5.1.
    import torch.distributed as _dist

    _original_init_process_group = _dist.init_process_group

    @functools.wraps(_original_init_process_group)
    def _patched_init_process_group(*args, **kwargs):
        kwargs.pop("_ranks", None)
        return _original_init_process_group(*args, **kwargs)

    _dist.init_process_group = _patched_init_process_group

    # ================================================================
    # Patch 9: DeviceMesh._unflatten — multi-dim mesh from flat mesh
    # ================================================================
    # PyTorch 2.6 added DeviceMesh._unflatten to create multi-dimensional
    # meshes from a 1D world mesh. In 2.5.1, we emulate this using
    # init_device_mesh with the same shape directly.
    from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

    def _unflatten_compat(
        self, dim, dim_degrees, dim_names, backend_override=None
    ):
        """Compatibility implementation of DeviceMesh._unflatten for PyTorch 2.5.1.

        Creates a multi-dimensional DeviceMesh by reshaping the flat mesh tensor.
        The backend_override parameter is accepted but ignored (all dims use the
        real backend). For dimensions with degree 1, process groups are trivial.
        """
        mesh_tensor = self.mesh.flatten().reshape(dim_degrees)
        device_type = self.device_type
        return init_device_mesh(
            device_type,
            tuple(dim_degrees),
            mesh_dim_names=tuple(dim_names),
        )

    DeviceMesh._unflatten = _unflatten_compat

    # ================================================================
    # Patch 10: torch.distributed.tensor._random.manual_seed — seed overflow
    # ================================================================
    # In PyTorch 2.5.1, OffsetBasedRNGTracker.set_seed does:
    #   torch.tensor([seed]).view(torch.uint8)
    # which overflows when seed >= 2^63 (int64 max). The new torchtitan code
    # uses `seed %= 2**64` which can produce such large values.
    # We wrap manual_seed to clamp the seed to int64 range.
    import torch.distributed.tensor._random as _dt_random

    _original_dt_manual_seed = _dt_random.manual_seed

    @functools.wraps(_original_dt_manual_seed)
    def _patched_dt_manual_seed(seed, device_mesh):
        # Clamp to int64 range to avoid overflow
        seed = seed % (2**63)
        return _original_dt_manual_seed(seed, device_mesh)

    _dt_random.manual_seed = _patched_dt_manual_seed

    # ================================================================
    # Patch 11: torch._C._dynamo.eval_frame._set_lru_cache stub
    # ================================================================
    # PyTorch 2.6 added _set_lru_cache to dynamo's eval_frame module.
    # torchtitan's activation_checkpoint.py calls it unconditionally.
    # In 2.5.1 we make it a no-op.
    import torch._C._dynamo.eval_frame as _eval_frame

    if not hasattr(_eval_frame, "_set_lru_cache"):
        _eval_frame._set_lru_cache = lambda *args, **kwargs: None

    # ================================================================
    # Patch 12: FSDPModule.set_gradient_divide_factor compatibility
    # ================================================================
    # In PyTorch 2.6 it was renamed from set_reduce_scatter_divide_factor
    # to set_gradient_divide_factor.
    from torch.distributed._composable.fsdp import FSDPModule as _FSDPModule

    if not hasattr(_FSDPModule, "set_gradient_divide_factor"):
        _FSDPModule.set_gradient_divide_factor = (
            _FSDPModule.set_reduce_scatter_divide_factor
        )

    # ================================================================
    # Patch 13: checkpoint_wrapper — strip early_stop kwarg
    # ================================================================
    # PyTorch 2.6 added `early_stop` to torch.utils.checkpoint.checkpoint.
    # In 2.5.1, it doesn't exist and gets passed through to the module's
    # forward() via **kwargs, causing TypeError.
    import torch.distributed.algorithms._checkpoint.checkpoint_wrapper as _ckpt_wrap_mod

    _original_checkpoint_wrapper = _ckpt_wrap_mod.checkpoint_wrapper

    @functools.wraps(_original_checkpoint_wrapper)
    def _patched_checkpoint_wrapper(module, *args, **kwargs):
        kwargs.pop("early_stop", None)
        return _original_checkpoint_wrapper(module, *args, **kwargs)

    _ckpt_wrap_mod.checkpoint_wrapper = _patched_checkpoint_wrapper

    # ================================================================
    # Patch 14: torch.nn.utils.get_total_norm / clip_grads_with_norm_
    # ================================================================
    # PyTorch 2.6 refactored clip_grad_norm_ into separate steps:
    #   get_total_norm (compute) + clip_grads_with_norm_ (apply).
    # In 2.5.1, these don't exist. We provide implementations based on
    # the 2.5.1 clip_grad_norm_ internals.
    import math
    from typing import Dict, List, Optional, Tuple, Union

    import torch.nn.utils as _nn_utils

    if not hasattr(_nn_utils, "get_total_norm"):

        def _get_total_norm(
            tensors: list,
            norm_type: float = 2.0,
            error_if_nonfinite: bool = False,
            foreach: Optional[bool] = None,
        ) -> torch.Tensor:
            """Compute total norm of a list of tensors (2.6 compat)."""
            norm_type = float(norm_type)
            if len(tensors) == 0:
                return torch.tensor(0.0)
            first_device = tensors[0].device
            if math.isinf(norm_type):
                norms = [t.detach().abs().max() for t in tensors]
                total_norm = norms[0] if len(norms) == 1 else torch.max(
                    torch.stack([n.to(first_device) for n in norms])
                )
            else:
                norms = [
                    torch.linalg.vector_norm(t.detach(), norm_type)
                    for t in tensors
                ]
                total_norm = torch.linalg.vector_norm(
                    torch.stack([n.to(first_device) for n in norms]), norm_type
                )
            if error_if_nonfinite and torch.logical_or(
                total_norm.isnan(), total_norm.isinf()
            ):
                raise RuntimeError(
                    f"The total norm of order {norm_type} for gradients from "
                    "`parameters` is non-finite, so it cannot be clipped."
                )
            return total_norm

        _nn_utils.get_total_norm = _get_total_norm

    if not hasattr(_nn_utils, "clip_grads_with_norm_"):

        def _clip_grads_with_norm_(
            parameters,
            max_norm: float,
            total_norm: torch.Tensor,
            foreach: Optional[bool] = None,
        ) -> None:
            """Clip gradients given a pre-computed total_norm (2.6 compat)."""
            max_norm = float(max_norm)
            clip_coef = max_norm / (total_norm + 1e-6)
            clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
            if isinstance(parameters, torch.Tensor):
                parameters = [parameters]
            else:
                parameters = list(parameters)
            grads = [p.grad for p in parameters if p.grad is not None]
            for grad in grads:
                grad.detach().mul_(clip_coef_clamped.to(grad.device))

        _nn_utils.clip_grads_with_norm_ = _clip_grads_with_norm_

    # ================================================================
    # Done
    # ================================================================
    import logging

    logging.getLogger(__name__).info(
        "[compat_shim] Applied PyTorch %s compatibility patches for "
        "cluster_data_selection (target: driver 535 / CUDA 12.1).",
        torch.__version__,
    )
