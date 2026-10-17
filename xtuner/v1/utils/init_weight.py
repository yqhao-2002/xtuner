from functools import partial
from typing import Callable, cast

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, distribute_tensor

from .device import get_device
from .misc import clean_param_name


DEVICE = get_device()


def _init_gated_delta_a_log(tensor: torch.Tensor) -> None:
    """Initialize GatedDeltaNet ``A_log`` as ``log(Uniform(0.01, 16))``."""
    tensor.uniform_(0.01, 16).log_()


def init_params(param: torch.Tensor, init_fn: Callable[[torch.Tensor], torch.Tensor | None]):
    """Initialize a single model parameter tensor, supporting both regular
    tensors and DTensors.

    Args:
        param (torch.Tensor): The parameter tensor to be initialized.
        init_fn (Callable[[torch.Tensor], torch.Tensor | None]): **in-place** initialization function to be applied.
    """
    assert not param.is_meta, "Internal Error. Found meta tensor during initialize model weight"
    device = param.device

    if isinstance(param, DTensor):
        # InterleavedShard cannot go through full_tensor/distribute_tensor because PyTorch has no
        # redistribute path for that placement chain. Initialize the values owned by this rank directly.
        from .interleaved_shard import RuntimeLayout

        if RuntimeLayout.from_dtensor(param).is_interleaved:
            init_fn(param._local_tensor)
        else:
            if param.device.type == "cpu":
                # cpu_offload: shards live on CPU while the mesh talks NCCL, so
                # full_tensor()/distribute_tensor() collectives are unavailable.
                _init_dtensor_collective_free(param, init_fn)
            else:
                full_param = torch.empty_like(param.full_tensor(), device=device)
                init_fn(full_param)
                param.copy_(distribute_tensor(full_param, param.device_mesh, param.placements))
    else:
        init_fn(param)


def _init_dtensor_collective_free(
    param: DTensor, init_fn: Callable[[torch.Tensor], torch.Tensor | None], base_seed: int = 1234
):
    """Initialize a DTensor's local shard without any collectives.

    Used when the mesh's process-group backend cannot run collectives on the
    parameter device (CPU shards under FSDP ``cpu_offload`` with an NCCL-only
    mesh: ``full_tensor()`` raises ``RuntimeError: No backend type associated
    with device type cpu``).

    The global RNG is reseeded deterministically from the mesh coordinate:

    * Replicate placements keep one shared seed, so every rank writes
      identical values (replicas stay consistent).
    * Shard placements mix the mesh coordinate into the seed, so each rank
      draws from an independent stream -- statistically equivalent to
      initializing the full tensor and sharding it for i.i.d. init functions
      (normal_/uniform_/zeros_/ones_).
    """
    local = param.to_local()
    seed = base_seed
    if not all(p.is_replicate() for p in param.placements):
        coord = param.device_mesh.get_coordinate()
        if coord is not None:
            for c in coord:
                seed = (seed * 1000003 + int(c) + 1) % (2**31 - 1)
    fork_devices = [] if local.device.type == "cpu" else [local.device]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        init_fn(local)


def default_init_weights(module: nn.Module) -> set[str]:
    initialized_params: set[str] = set()

    def _init_weights_recursive(name: str, module: nn.Module, seen: set | None = None):
        if seen is None:
            seen = set()

        if id(module) in seen:
            return

        _default_init_atom(name, module)
        for child_name, child in module.named_children():
            child_name = f"{child_name}" if name == "" else f"{name}.{child_name}"
            _init_weights_recursive(child_name, child, seen)

        seen.add(id(module))

    def _default_init_atom(name: str, module: nn.Module):
        if hasattr(module, "bias") and module.bias is not None:
            bias = cast(torch.Tensor, module.bias)
            init_params(bias, nn.init.zeros_)
            initialized_params.add(clean_param_name(f"{name}.bias"))

        if hasattr(module, "weight") and module.weight is not None:
            weight = cast(torch.Tensor, module.weight)
            if "norm" in name:
                init_params(weight, nn.init.ones_)
            else:
                init_params(weight, partial(nn.init.normal_, mean=0.0, std=0.02))
            initialized_params.add(clean_param_name(f"{name}.weight"))

        # GatedDeltaNet keeps decay/time-step as named parameters, not `.weight`.
        if hasattr(module, "dt_bias") and isinstance(module.dt_bias, nn.Parameter):
            init_params(module.dt_bias, nn.init.ones_)
            initialized_params.add(clean_param_name(f"{name}.dt_bias"))
        if hasattr(module, "A_log") and isinstance(module.A_log, nn.Parameter):
            init_params(module.A_log, _init_gated_delta_a_log)
            initialized_params.add(clean_param_name(f"{name}.A_log"))

    _init_weights_recursive("", module)
    return initialized_params
