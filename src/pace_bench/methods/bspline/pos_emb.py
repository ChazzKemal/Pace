"""Train the action slice of xVLA's positional embedding -- and, crucially, keep it.

`pos_emb` is the only channel through which sequence position reaches an action token.
`action_encoder` is one `Linear` applied identically to every row, so it never sees the
row index; the trunk's attention is permutation-invariant without a positional code. That
table was pretrained to mean "position k is the k-th *timestep* of a dense action chunk,
uniformly spaced". Under B-spline, position k is the k-th *control point*, and consecutive
control points are not uniformly spaced in time -- knot spacing adapts to how fast the
demonstration moves (measured on `ds_libero10_bspline`: spans of 7.8 to 49.1 source frames
for the same 16 rows). The rows also stop being homogeneous, since the first `degree` and
last `degree + 1` of them are the window's boundary rows rather than interior control
points.

Two things stop this being a config flag, and the second one is the dangerous one.

1. **PEFT cannot target it.** `--peft.full_training_modules` becomes PEFT's
   `modules_to_save`, which wraps `nn.Module`s. `pos_emb` is a bare `nn.Parameter` and
   does not appear in `named_modules()` at all, so naming it there either errors or does
   nothing depending on the PEFT version. It is created with `requires_grad=True`
   (`soft_transformer.py:334`); PEFT is what freezes it, by freezing everything it did
   not wrap.

2. **It would not be saved.** A PEFT checkpoint's `adapter_model.safetensors` holds the
   LoRA tensors plus the `modules_to_save` weights and nothing else -- verified against
   `ds_libero10_bspline`: 534 LoRA tensors and exactly five others (`action_encoder`,
   `action_decoder`, `soft_prompt_hub`). So simply flipping `requires_grad` gives a run
   that trains the embedding, reports the improved loss, and then writes a checkpoint
   without it. Evaluation would silently load the original frozen table and the whole
   experiment would read as a null result. `unfreeze` therefore wraps `save_pretrained`,
   and `restore` is its other half.

Only the action rows are trained. The remaining rows index the Florence-2 visual tokens,
whose consumption is otherwise frozen; letting those drift would change something this is
not asking about. A slice of a Parameter cannot carry its own `requires_grad`, so the
whole tensor is unfrozen and a gradient hook zeroes every row past the action segment.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)

#: Written beside the adapter, because the adapter cannot carry it.
FILENAME = "pos_emb.safetensors"
KEY = "pos_emb"


def find(policy) -> tuple[str | None, torch.nn.Parameter | None]:
    """The `pos_emb` parameter, wherever PEFT's wrappers have buried it."""
    for name, param in policy.named_parameters():
        if name.rsplit(".", 1)[-1] == KEY:
            return name, param
    return None, None


def unfreeze(policy, rows: int) -> torch.nn.Parameter:
    """Train `pos_emb[:, :rows]`, freeze the rest, and make the result persist.

    Args:
        policy: A constructed policy owning an xVLA transformer.
        rows: How many leading positions to train -- the action segment, which is the
            policy's chunk. Rows past it index visual tokens and stay exactly as
            pretrained.

    Returns:
        The parameter, now trainable.
    """
    name, param = find(policy)
    if param is None:
        raise ValueError(
            "no `pos_emb` parameter on this policy -- --method.unfreeze_pos_emb_rows "
            "only applies to xVLA, whose action expert owns one."
        )
    if rows > param.shape[1]:
        raise ValueError(
            f"asked to train {rows} positional rows but pos_emb holds {param.shape[1]}."
        )

    param.requires_grad_(True)
    mask = torch.zeros_like(param)
    mask[:, :rows, :] = 1.0
    # Built once, moved per call: the policy may be moved between construction and the
    # first backward, and a mask on the wrong device would fail there rather than here.
    param.register_hook(lambda grad: grad * mask.to(grad.device, grad.dtype))
    _persist(policy, param)

    total = param.numel()
    trained = int(mask.sum().item())
    logger.info(
        "B-spline: training %s rows 0..%d of %s (%d of %d parameters); rows %d.. stay "
        "frozen, and the tensor is saved beside the adapter as %s",
        KEY, rows - 1, name, trained, total, rows, FILENAME,
    )
    return param


def _persist(policy, param: torch.nn.Parameter) -> None:
    """Write `pos_emb` next to every checkpoint this policy saves.

    Wraps the bound method rather than the class, so nothing else in the process is
    affected and a policy without the flag saves exactly as before.
    """
    original = policy.save_pretrained

    def save_pretrained(save_directory, *args, **kwargs):
        result = original(save_directory, *args, **kwargs)
        path = Path(save_directory) / FILENAME
        save_file({KEY: param.detach().cpu().contiguous()}, str(path))
        logger.debug("wrote %s", path)
        return result

    policy.save_pretrained = save_pretrained


def restore(policy, directory) -> bool:
    """Load a trained `pos_emb` back, if the checkpoint carries one.

    Returns whether anything was restored, so a caller can say which of the two kinds of
    checkpoint it has rather than guessing. Safe to call on a checkpoint trained with the
    embedding frozen: there is no file, nothing is touched, and it returns False.
    """
    path = Path(directory) / FILENAME
    if not path.is_file():
        return False
    _, param = find(policy)
    if param is None:
        logger.warning("%s exists but this policy has no pos_emb to load it into", path)
        return False
    saved = load_file(str(path))[KEY]
    if saved.shape != param.shape:
        raise ValueError(
            f"{path} holds a {tuple(saved.shape)} pos_emb but this policy's is "
            f"{tuple(param.shape)}."
        )
    with torch.no_grad():
        param.copy_(saved.to(param.device, param.dtype))
    logger.info("restored a trained %s from %s", KEY, path)
    return True


__all__ = ["FILENAME", "find", "restore", "unfreeze"]
