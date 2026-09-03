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

The table is one tensor for the *whole* sequence, and that is a second problem. xVLA lays
its tokens out as ``[action rows | VLM tokens | auxiliary visual tokens]`` and adds
``pos_emb[:seq_len]`` across all of them, so the row a visual token lands on is
``chunk_size + its offset``. `lerobot/xvla-libero` was pretrained at chunk 30; a B-spline
matrix is 16 rows, so every one of the 250 visual and text tokens reads a row 14 below
the one it was pretrained with. The pretrained table is nowhere near smooth enough for
that to be harmless -- the cosine between a row and the row 14 above it is 0.10, against
0.007 for random pairs -- so the visual tokens are effectively handed random positional
codes, and freezing "the visual rows" freezes the wrong ones. (DemoSpeedup's chunk-15 arm
has the same shift.) :func:`realign` undoes it by moving the rows the non-action tokens
were pretrained with down to where those tokens now sit.

Two things stop the trainable slice being a config flag, and the second one is the
dangerous one.

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
   experiment would read as a null result. `unfreeze` and `realign` therefore both wrap
   `save_pretrained`, and `restore` is their other half -- a realigned table is lost at
   evaluation exactly as a trained one would be, since the policy is rebuilt from the
   base checkpoint with the shift back in place.

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
#: Set on a policy whose `save_pretrained` already writes the table, so wrapping it a
#: second time (realign, then unfreeze) does not write the file twice.
_PERSISTED = "_pos_emb_persisted"


def find(policy) -> tuple[str | None, torch.nn.Parameter | None]:
    """The `pos_emb` parameter, wherever PEFT's wrappers have buried it."""
    for name, param in policy.named_parameters():
        if name.rsplit(".", 1)[-1] == KEY:
            return name, param
    return None, None


def realign(policy, source_rows: int, rows: int) -> torch.nn.Parameter:
    """Keep every non-action token on the positional row it was pretrained with.

    Args:
        policy: A constructed policy owning an xVLA transformer.
        source_rows: The action-segment length the table was pretrained with -- the
            checkpoint's own `chunk_size`, before the method changed it.
        rows: The action-segment length this run trains with.

    The table stays a permutation of itself. Rows ``0..min(rows, source_rows)-1`` are
    untouched; the rows the non-action tokens were pretrained with are moved so they
    start at ``rows``, which is where those tokens now begin; the rows that displaces go
    to the end of the table, past anything a sequence reaches. When the segment grows
    instead, the new action rows are taken from that unused tail. Persisted beside the
    adapter like a trained table, for the reason given in the module docstring.

    Returns:
        The parameter, realigned.
    """
    name, param = find(policy)
    if param is None:
        raise ValueError(
            "no `pos_emb` parameter on this policy -- --method.realign_pos_emb only "
            "applies to xVLA, whose action expert owns one."
        )
    length = param.shape[1]
    if not (0 < rows <= length and 0 < source_rows <= length):
        raise ValueError(
            f"cannot realign pos_emb from a {source_rows}-row to a {rows}-row action "
            f"segment: the table holds {length} rows."
        )
    if rows != source_rows:
        old = param.detach().clone()
        if rows < source_rows:
            new = torch.cat([old[:, :rows], old[:, source_rows:], old[:, rows:source_rows]], dim=1)
        else:
            extra = rows - source_rows
            new = torch.cat(
                [old[:, :source_rows], old[:, length - extra :], old[:, source_rows : length - extra]],
                dim=1,
            )
        with torch.no_grad():
            param.copy_(new)
    _persist(policy, param)
    moved = length - max(rows, source_rows)
    logger.info(
        "B-spline: realigned %s (%s) for a %d-row action segment pretrained at %d rows -- "
        "%d non-action rows now sit at %d.. as they did at %d..; saved beside the adapter "
        "as %s",
        KEY, name, rows, source_rows, moved, rows, source_rows, FILENAME,
    )
    return param


def unfreeze(policy, rows: int) -> torch.nn.Parameter:
    """Train `pos_emb[:, :rows]`, freeze the rest, and make the result persist.

    Args:
        policy: A constructed policy owning an xVLA transformer.
        rows: How many leading positions to train -- the action segment, which is the
            policy's chunk. Rows past it index visual tokens and stay frozen, on whatever
            row they were given: see :func:`realign` for why that is not automatically
            the row they were pretrained with.

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
    affected and a policy without the flag saves exactly as before. Idempotent: a
    policy that is realigned and then unfrozen is wrapped once.
    """
    if getattr(policy, _PERSISTED, False):
        return
    original = policy.save_pretrained

    def save_pretrained(save_directory, *args, **kwargs):
        result = original(save_directory, *args, **kwargs)
        path = Path(save_directory) / FILENAME
        save_file({KEY: param.detach().cpu().contiguous()}, str(path))
        logger.debug("wrote %s", path)
        return result

    policy.save_pretrained = save_pretrained
    setattr(policy, _PERSISTED, True)


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


__all__ = ["FILENAME", "find", "realign", "restore", "unfreeze"]
