import os, glob, torch
import torch.nn as nn
from typing import Union, Dict, Any

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None

from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
from torch.distributed._tensor.api import distribute_tensor  # DTensor API


def _load_full_state_dict(src: Union[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Load a *full* (unsharded) state dict from:
      - a directory with HF weights (safetensors/bin, possibly sharded)
      - a single file (.safetensors or .bin)
      - or directly a dict (already loaded).
    Returns CPU tensors.
    """
    if isinstance(src, dict):
        return src

    if not isinstance(src, str):
        raise TypeError(f"Unsupported src type: {type(src)}")

    if os.path.isdir(src):
        # Prefer safetensors shards if available
        st_files = sorted(glob.glob(os.path.join(src, "*.safetensors")))
        if st_files and safe_load_file is not None:
            full_sd: Dict[str, torch.Tensor] = {}
            for f in st_files:
                sd = safe_load_file(f)  # already on CPU
                full_sd.update(sd)
            return full_sd

        # Fallback to .bin shards
        bin_files = sorted(glob.glob(os.path.join(src, "pytorch_model*.bin")))
        if bin_files:
            full_sd = {}
            for f in bin_files:
                sd = torch.load(f, map_location="cpu", mmap=True, weights_only=True)
                full_sd.update(sd)
            return full_sd

        # Single-file conventional names
        for fname in ("pytorch_model.bin", "model.safetensors"):
            fpath = os.path.join(src, fname)
            if os.path.exists(fpath):
                if fpath.endswith(".safetensors"):
                    if safe_load_file is None:
                        raise RuntimeError("safetensors is not available to load model.safetensors")
                    return safe_load_file(fpath)
                return torch.load(fpath, map_location="cpu", mmap=True, weights_only=True)

        raise FileNotFoundError(f"No HF weight files found under: {src}")

    # Single file path
    if src.endswith(".safetensors"):
        if safe_load_file is None:
            raise RuntimeError("safetensors is not available to load the provided file")
        return safe_load_file(src)
    return torch.load(src, map_location="cpu", mmap=True, weights_only=True)


def _maybe_remap_keys_to_match_model(full_sd: Dict[str, torch.Tensor], model_keys) -> Dict[str, torch.Tensor]:
    """
    Light key remapping: some HF checkpoints omit or include a leading 'model.' prefix.
    """
    if set(full_sd).intersection(model_keys):
        return full_sd  # already overlaps

    # Try adding 'model.' prefix
    remapped = { (f"model.{k}" if not k.startswith("model.") else k): v for k, v in full_sd.items() }
    if set(remapped).intersection(model_keys):
        return remapped

    # Try removing leading 'model.' prefix
    stripped = { (k[6:] if k.startswith("model.") else k): v for k, v in full_sd.items() }
    if set(stripped).intersection(model_keys):
        return stripped

    return full_sd  # give up; let strict=False handle leftovers


def load_original_weights_fsdp2(
    model,                       # FSDP2-sharded module (after fully_shard)
    # src: Union[str, Dict[str, torch.Tensor]],
    # *,
    src,
    use_dcp_api: bool = False,
    strict: bool = False,
) -> None:
    """
    Load *original* (full/unsharded) model weights into an FSDP2-sharded `model`.

    Args:
        model: FSDP2-sharded module (parameters are DTensors / meta before assignment).
        src:   HF model dir, single file path (.bin / .safetensors), or a preloaded full state_dict (CPU tensors).
        use_dcp_api: If True, load via torch.distributed.checkpoint API with full_state_dict=True.
        strict: Passed to load_state_dict. Keep False to tolerate minor key/shape differences.

    Notes:
      - Call this *after* applying `fully_shard(...)` so the module’s parameters are DTensors (or meta),
        and *before* first optimizer step.
      - Uses `assign=True` so we never call `.copy_()` on meta tensors.
    """
    # full_sd = _load_full_state_dict(src)
    full_sd = src

    if use_dcp_api:
        # Let DCP handle mapping from a full state dict into the sharded module.
        set_model_state_dict(
            model=model,
            model_state_dict=full_sd,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
        )
        return

    # DTensor path: convert each full tensor into the model's DTensor placement.
    # We rely on the model's *meta* state_dict for mesh/placement hints.
    meta_sd = model.state_dict()  # keys exist; values are DTensor/meta shells
    full_sd = _maybe_remap_keys_to_match_model(full_sd, set(meta_sd.keys()))

    sharded_sd: Dict[str, Any] = {}
    missing = []
    for name, full_tensor in full_sd.items():
        dst_meta = meta_sd.get(name)
        if dst_meta is None:
            # Not present in the model (e.g., extra/tied/tok-emb mismatch); skip
            continue
        try:
            dt = distribute_tensor(
                full_tensor, dst_meta.device_mesh, dst_meta.placements
            )
        except Exception as e:
            missing.append((name, tuple(full_tensor.shape), str(e)))
            continue

        # Preserve requires_grad semantics for parameters
        if isinstance(dst_meta, nn.Parameter):
            sharded_sd[name] = nn.Parameter(dt, requires_grad=dst_meta.requires_grad)
        else:
            sharded_sd[name] = dt  # buffer

    # Assign directly into (meta/DTensor) parameters
    model.load_state_dict(sharded_sd, strict=strict, assign=True)

    if missing:
        # Optional: print a concise report (doesn't raise)
        msg = "\n".join([f"- {k} {shape}: {err}" for k, shape, err in missing[:10]])
        print(f"[load_original_weights_fsdp2] Skipped {len(missing)} entries (showing up to 10):\n{msg}")

        