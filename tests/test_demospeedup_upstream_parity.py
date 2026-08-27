"""Parity with the genuine DemoSpeedup implementation.

The retiming here descends from

    https://github.com/lingxiao-guo/DemoSpeedup   (CoRL 2025)

by way of two intermediate ports -- the lerobot fork's chunk-level
``downsample_with_labels`` and crisp_gym's episode-level ``select_keep_indices``.
Checking against those ports only shows the copy is faithful to a copy. This checks
against upstream itself, by executing its real function out of a clone.

Point ``DEMOSPEEDUP_UPSTREAM`` at one; the module skips without it, so the suite
still runs on a machine that has no clone:

    git clone --depth 1 https://github.com/lingxiao-guo/DemoSpeedup /tmp/DemoSpeedup
    DEMOSPEEDUP_UPSTREAM=/tmp/DemoSpeedup pytest tests/test_demospeedup_upstream_parity.py

Upstream's modules import mujoco and hydra at module scope, so the function under
test is loaded from source text rather than imported. (crisp_gym's
``demospeedup/tests/test_upstream_parity.py`` does the same, deliberately.)

Three places where this implementation *deliberately* differs from upstream are
pinned below, so they stay decisions rather than drift.
"""

import ast
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from robot_stack.methods.demospeedup.retime import episode_keep_indices, keep_indices, retime_chunk

UPSTREAM = os.environ.get("DEMOSPEEDUP_UPSTREAM", "")
pytestmark = pytest.mark.skipif(
    not (UPSTREAM and (Path(UPSTREAM) / "aloha" / "act").is_dir()),
    reason="set DEMOSPEEDUP_UPSTREAM to a DemoSpeedup clone to run parity tests",
)


def _load(relpath: str, name: str):
    """Exec one top-level def out of an upstream file, without importing the module."""
    tree = ast.parse((Path(UPSTREAM) / relpath).read_text())
    fn = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name),
        None,
    )
    if fn is None:
        raise AssertionError(f"{relpath}: no top-level def {name}")
    ns: dict = {"np": np, "torch": torch}
    exec(compile(ast.Module([fn], []), f"<upstream:{relpath}>", "exec"), ns)  # noqa: S102
    return ns[name]


@pytest.fixture(scope="module")
def upstream_process():
    return _load("aloha/act/act_utils.py", "process_action_label")


def random_case(rng, horizon=None):
    horizon = horizon or int(rng.integers(6, 60))
    labels = torch.from_numpy(rng.integers(0, 2, horizon).astype(np.int64))
    actions = torch.randn(horizon, 7)
    is_pad = torch.zeros(horizon, dtype=torch.bool)
    is_pad[-2:] = True
    return labels, actions, is_pad


def upstream_indices(upstream_process, labels, actions, is_pad):
    """Recover the indices upstream kept, by matching the rows it emitted."""
    kept, _ = upstream_process(actions.clone(), labels.clone(), is_pad.clone())
    indices = []
    for row in kept:
        hits = (actions == row).all(-1).nonzero()
        if len(hits) == 0:
            break  # a zero-filled tail row: past the end of the kept region
        indices.append(int(hits[0]))
    return indices


def test_the_walk_matches_upstream_exactly(upstream_process):
    """The core algorithm, on 500 random label sequences."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        labels, actions, is_pad = random_case(rng)
        mine = keep_indices(labels.numpy(), 2, 4, start=-1)
        theirs = upstream_indices(upstream_process, labels, actions, is_pad)
        assert mine[: len(theirs)] == theirs, f"diverged on labels={labels.tolist()}"


@pytest.mark.parametrize(("low_v", "high_v"), [(2, 4), (1, 2), (3, 3), (2, 8)])
def test_strides_are_parameterised_the_same_way(upstream_process, low_v, high_v):
    """Upstream hard-codes 2/4; the walk must generalise without changing at 2/4."""
    rng = np.random.default_rng(1)
    labels, actions, is_pad = random_case(rng, horizon=40)
    mine = keep_indices(labels.numpy(), low_v, high_v, start=-1)
    if (low_v, high_v) == (2, 4):
        assert mine[: len(upstream_indices(upstream_process, labels, actions, is_pad))] == (
            upstream_indices(upstream_process, labels, actions, is_pad)
        )
    else:
        assert isinstance(mine, list)  # generalises; no upstream reference to compare


# --- the three deliberate deviations ------------------------------------------------


def test_deviation_1_walk_starts_at_zero_not_minus_one(upstream_process):
    """Upstream begins at ``i = -1``, so its first consulted label is ``labels[-1]``
    -- the chunk's LAST frame. Almost certainly a slip; it perturbs the first step
    only. Both intermediate ports use ``start=0`` and so does this one, but the
    upstream convention stays reachable and is what the parity test above uses.
    """
    rng = np.random.default_rng(2)
    differed = 0
    for _ in range(200):
        labels, _, _ = random_case(rng)
        if keep_indices(labels.numpy(), 2, 4, start=0) != keep_indices(labels.numpy(), 2, 4, start=-1):
            differed += 1
    assert differed > 0, "the two start conventions should not agree everywhere"


def test_deviation_2_index_zero_is_kept(upstream_process):
    """Upstream never keeps index 0; it appends only after the first jump.

    Index 0 is the action for the *current* observation, so dropping it means the
    first thing the policy is trained to output is already a step ahead. Both ports
    prepend it and so does this one.
    """
    rng = np.random.default_rng(3)
    labels, actions, is_pad = random_case(rng, horizon=40)
    theirs = upstream_indices(upstream_process, labels, actions, is_pad)
    mine = episode_keep_indices(labels.numpy(), 2, 4, keep_last=False)
    assert theirs[0] != 0, "upstream is expected to skip index 0"
    assert mine[0] == 0, "this implementation anchors on the current observation"


def test_deviation_3_the_retimed_tail_is_masked(upstream_process):
    """Upstream initialises ``new_is_pad`` to zeros, leaving the zero-filled tail
    *unmasked* -- so the loss regresses those zeros as if they were real targets.

    This implementation initialises the mask to True, so only the kept waypoints
    contribute. ``pad_mode="hold"`` addresses the same hazard from the other side,
    for policies whose loss ignores the mask entirely.
    """
    rng = np.random.default_rng(4)
    labels, actions, is_pad = random_case(rng, horizon=40)

    _, their_pad = upstream_process(actions.clone(), labels.clone(), is_pad.clone())
    _, my_pad = retime_chunk(actions.clone(), labels.numpy(), is_pad.clone(), 2, 4, "zero")

    n = len(episode_keep_indices(labels.numpy(), 2, 4, keep_last=False))
    assert not bool(their_pad[-1]), "upstream leaves the tail unmasked"
    assert bool(my_pad[-1]), "this implementation masks the tail"
    assert torch.all(my_pad[n:]), "everything past the kept region must be masked"
