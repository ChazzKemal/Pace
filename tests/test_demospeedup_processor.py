"""The retiming step, and how it finds a sample's episode tail.

The walk arithmetic is covered by test_demospeedup_retime.py. What is left is the
part that can silently mis-align: locating each sample within its own episode and
substituting the right tail's walk for its action chunk. Getting that wrong retimes
chunks against the wrong frames and still trains, so it is worth testing more
carefully than the maths.
"""

import numpy as np
import pytest
import torch
import upstream_reference
from lerobot.lerobot_types import TransitionKey
from lerobot.processor.pipeline import ProcessorStepRegistry

from robot_stack.methods.config import DemoSpeedupMethod
from robot_stack.methods.demospeedup.processor import ACTION_IS_PAD, DemoSpeedupRetimeStep
from robot_stack.methods.demospeedup.retime import keep_indices, retime_tail

CHUNK, DIM = 20, 7


def transition(actions, episode_index, *, index=None, frame_index=None, is_pad=None):
    complementary = {"episode_index": torch.tensor(episode_index)}
    if index is not None:
        complementary["index"] = torch.tensor(index)
    if frame_index is not None:
        complementary["frame_index"] = torch.tensor(frame_index)
    if is_pad is not None:
        complementary[ACTION_IS_PAD] = is_pad
    return {TransitionKey.ACTION: actions, TransitionKey.COMPLEMENTARY_DATA: complementary}


@pytest.fixture
def episodes():
    rng = np.random.default_rng(0)
    labels = {0: rng.integers(0, 2, 100).astype(np.int64), 1: rng.integers(0, 2, 80).astype(np.int64)}
    actions = {ep: rng.normal(size=(len(lab), DIM)).astype(np.float32) for ep, lab in labels.items()}
    return labels, actions


def build(labels, actions, **kwargs):
    kwargs.setdefault("low_v", 2)
    kwargs.setdefault("high_v", 4)
    return DemoSpeedupRetimeStep(labels=labels, episode_actions=actions, **kwargs)


def test_is_registered():
    assert ProcessorStepRegistry.get("demospeedup_retime") is DemoSpeedupRetimeStep


def test_substitutes_each_samples_own_tail(episodes):
    """The core correctness claim: sample i gets the walk over episode i's tail."""
    labels, ep_actions = episodes
    step = build(labels, ep_actions)
    batch_actions = torch.randn(2, CHUNK, DIM)  # loader window: values must be ignored
    out = step(transition(batch_actions.clone(), [0, 1], frame_index=[10, 30]))[TransitionKey.ACTION]

    for i, (ep, frame) in enumerate(((0, 10), (1, 30))):
        expected, _ = retime_tail(
            torch.from_numpy(ep_actions[ep][frame:]), labels[ep][frame:], CHUNK, 2, 4, "zero"
        )
        torch.testing.assert_close(out[i], expected, rtol=0, atol=0)


def test_loader_window_content_is_irrelevant(episodes):
    """The batch's own action values must not leak into the output.

    The loader's fixed window is under-supplied by construction; the step exists to
    replace it. Two different window contents must produce identical chunks.
    """
    labels, ep_actions = episodes
    step = build(labels, ep_actions)
    a = step(transition(torch.zeros(1, CHUNK, DIM), [0], frame_index=[5]))[TransitionKey.ACTION]
    b = step(transition(torch.randn(1, CHUNK, DIM), [0], frame_index=[5]))[TransitionKey.ACTION]
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_mid_episode_chunk_has_no_pad_slots(episodes):
    """Upstream's property: mid-episode, every slot is a real waypoint."""
    labels, ep_actions = episodes
    step = build(labels, ep_actions)
    out = step(transition(torch.randn(1, CHUNK, DIM), [0], frame_index=[0]))
    assert not out[TransitionKey.COMPLEMENTARY_DATA][ACTION_IS_PAD][0].any()


def test_episode_end_pads_and_masks(episodes):
    """Only where the episode itself runs out may pad slots appear -- masked."""
    labels, ep_actions = episodes
    step = build(labels, ep_actions, pad_mode="hold")
    frame = len(labels[0]) - 3
    out = step(transition(torch.randn(1, CHUNK, DIM), [0], frame_index=[frame]))
    pad = out[TransitionKey.COMPLEMENTARY_DATA][ACTION_IS_PAD][0]
    acts = out[TransitionKey.ACTION][0]
    n = int((~pad).sum())
    assert 0 < n < CHUNK
    torch.testing.assert_close(acts[n:], acts[n - 1].expand(CHUNK - n, DIM))  # hold, not zeros


def test_global_index_is_converted_to_a_within_episode_frame(episodes):
    """`frame_index` does not survive LeRobot's batch-to-transition converter.

    Only the dataset-global `index` does, so the step subtracts the episode's start
    offset. If that arithmetic is wrong, chunks are retimed against the wrong frames
    and training still succeeds -- which is why this is asserted against the
    frame_index path rather than merely exercised.
    """
    labels, ep_actions = episodes
    starts = {0: 0, 1: 100}
    step = build(labels, ep_actions, episode_starts=starts)
    direct = build(labels, ep_actions)

    actions = torch.randn(2, CHUNK, DIM)
    via_index = step(transition(actions.clone(), [0, 1], index=[10, 130]))[TransitionKey.ACTION]
    via_frame = direct(transition(actions.clone(), [0, 1], frame_index=[10, 30]))[TransitionKey.ACTION]
    torch.testing.assert_close(via_index, via_frame, rtol=0, atol=0)


def test_global_index_without_starts_is_an_error_not_a_guess(episodes):
    labels, ep_actions = episodes
    step = build(labels, ep_actions)
    with pytest.raises(ValueError, match="episode_starts"):
        step(transition(torch.randn(1, CHUNK, DIM), [0], index=[10]))


def test_unlabelled_episodes_pass_through(episodes):
    """A partially labelled dataset must degrade to ordinary training, not fail."""
    labels, ep_actions = episodes
    step = build(labels, ep_actions)
    actions = torch.randn(2, CHUNK, DIM)
    out = step(transition(actions.clone(), [0, 99], frame_index=[10, 0]))[TransitionKey.ACTION]
    assert not torch.equal(out[0], actions[0]), "labelled episode should have been substituted"
    torch.testing.assert_close(out[1], actions[1], rtol=0, atol=0)


def test_no_labels_at_all_is_a_no_op():
    """So --method.type=demospeedup stays selectable before a labelling run exists."""
    actions = torch.randn(2, CHUNK, DIM)
    out = DemoSpeedupRetimeStep()(transition(actions.clone(), [0, 1], frame_index=[0, 0]))
    torch.testing.assert_close(out[TransitionKey.ACTION], actions, rtol=0, atol=0)


def test_labels_without_actions_fail_at_construction(episodes):
    """A labelled episode the step cannot retime must not silently train as baseline."""
    labels, ep_actions = episodes
    with pytest.raises(ValueError, match="no actions"):
        DemoSpeedupRetimeStep(labels=labels, episode_actions={0: ep_actions[0]})


def test_mismatched_label_and_action_lengths_fail_at_construction(episodes):
    """Labels from a different dataset must fail loudly, not retime the wrong frames."""
    labels, ep_actions = episodes
    wrong = dict(ep_actions)
    wrong[1] = wrong[1][:-1]
    with pytest.raises(ValueError, match="do not match"):
        DemoSpeedupRetimeStep(labels=labels, episode_actions=wrong)


def test_the_input_batch_is_not_mutated(episodes):
    """Training loops reuse batches; substituting in place would corrupt later use."""
    labels, ep_actions = episodes
    step = build(labels, ep_actions)
    actions = torch.randn(2, CHUNK, DIM)
    original = actions.clone()
    step(transition(actions, [0, 1], frame_index=[10, 30]))
    torch.testing.assert_close(actions, original, rtol=0, atol=0)


def test_missing_episode_index_is_a_clear_error(episodes):
    labels, ep_actions = episodes
    step = build(labels, ep_actions)
    with pytest.raises(KeyError, match="episode_index"):
        step({TransitionKey.ACTION: torch.randn(1, CHUNK, DIM), TransitionKey.COMPLEMENTARY_DATA: {}})


def test_method_validates_its_knobs():
    with pytest.raises(ValueError, match="pad_mode"):
        DemoSpeedupMethod(pad_mode="repeat")
    with pytest.raises(ValueError, match="strides"):
        DemoSpeedupMethod(low_v=0)


def test_method_without_labels_still_builds_a_step():
    (step,) = DemoSpeedupMethod().preprocessor_steps()
    assert isinstance(step, DemoSpeedupRetimeStep)
    assert step.labels == {}


def test_method_contributes_no_postprocessor_steps():
    """DemoSpeedup acts on training targets; it has nothing to say at inference."""
    assert DemoSpeedupMethod().postprocessor_steps() == []


def test_out_len_emits_the_trained_chunk_not_the_window(episodes):
    """After halving, the policy trains a shorter chunk than the loader's window.

    The step must emit exactly the trained length: xVLA truncates over-length
    action inputs, but ACT's VAE encoder consumes the sequence at chunk_size and
    a mismatch is a hard shape error (position embedding 102 vs 52 -- the crash
    this test pins). Pass-through rows are truncated too, which is what a
    chunk-out_len window would have delivered.
    """
    labels, ep_actions = episodes
    step = build(labels, ep_actions, out_len=CHUNK // 2)
    actions = torch.randn(2, CHUNK, DIM)
    out = step(transition(actions.clone(), [0, 99], frame_index=[10, 0]))

    got = out[TransitionKey.ACTION]
    assert got.shape == (2, CHUNK // 2, DIM)
    assert out[TransitionKey.COMPLEMENTARY_DATA][ACTION_IS_PAD].shape == (2, CHUNK // 2)
    expected, _ = retime_tail(
        torch.from_numpy(ep_actions[0][10:]), labels[0][10:], CHUNK // 2, 2, 4, "zero"
    )
    torch.testing.assert_close(got[0], expected, rtol=0, atol=0)  # substituted row
    torch.testing.assert_close(got[1], actions[1, : CHUNK // 2], rtol=0, atol=0)  # pass-through row


def test_the_stride_walk_matches_upstream_exactly():
    """`keep_indices` reproduces the paper's walk on 500 random label sequences.

    Upstream returns the kept rows, not their indices, so the indices are recovered
    by matching rows back to the input. Its loop starts at ``i = -1``, which is what
    ``start=-1`` exists for; everything in this project runs the ``start=0``
    convention instead. Compared against the verbatim copy in
    `tests/upstream_reference.py` -- see that module for why it is copied.
    """
    rng = np.random.default_rng(0)
    for _ in range(500):
        horizon = int(rng.integers(6, 60))
        labels = torch.from_numpy(rng.integers(0, 2, horizon).astype(np.int64))
        actions = torch.randn(horizon, 7)
        is_pad = torch.zeros(horizon, dtype=torch.bool)
        is_pad[-2:] = True

        kept, _ = upstream_reference.process_action_label(
            actions.clone(), labels.clone(), is_pad.clone()
        )
        theirs = []
        for row in kept:
            hits = (actions == row).all(-1).nonzero()
            if len(hits) == 0:
                break  # a zero-filled tail row: past the end of the kept region
            theirs.append(int(hits[0]))

        mine = keep_indices(labels.numpy(), 2, 4, start=-1)
        assert mine[: len(theirs)] == theirs, f"diverged on labels={labels.tolist()}"
