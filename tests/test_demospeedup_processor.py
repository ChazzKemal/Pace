"""The retiming step, and how it finds a sample's labels.

The arithmetic is covered by test_demospeedup_retime.py. What is left is the part
that can silently mis-align: locating each sample within its own episode. Getting
that wrong retimes chunks against the wrong labels and still trains, so it is worth
testing more carefully than the maths.
"""

import numpy as np
import pytest
import torch
from lerobot.lerobot_types import TransitionKey
from lerobot.processor.pipeline import ProcessorStepRegistry

from robot_stack.methods.config import DemoSpeedupMethod
from robot_stack.methods.demospeedup.processor import ACTION_IS_PAD, DemoSpeedupRetimeStep
from robot_stack.methods.demospeedup.retime import retime_chunk

HORIZON, DIM = 20, 7


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
def labels():
    rng = np.random.default_rng(0)
    return {0: rng.integers(0, 2, 100).astype(np.int64), 1: rng.integers(0, 2, 80).astype(np.int64)}


def test_is_registered():
    assert ProcessorStepRegistry.get("demospeedup_retime") is DemoSpeedupRetimeStep


def test_retimes_each_sample_against_its_own_window(labels):
    """The core correctness claim: sample i uses episode i's labels at its own frame."""
    step = DemoSpeedupRetimeStep(labels=labels, low_v=2, high_v=4)
    actions = torch.randn(2, HORIZON, DIM)
    out = step(transition(actions.clone(), [0, 1], frame_index=[10, 30]))[TransitionKey.ACTION]

    for i, (ep, frame) in enumerate(((0, 10), (1, 30))):
        expected, _ = retime_chunk(actions[i].clone(), labels[ep][frame : frame + HORIZON], None, 2, 4)
        torch.testing.assert_close(out[i], expected, rtol=0, atol=0)


def test_global_index_is_converted_to_a_within_episode_frame(labels):
    """`frame_index` does not survive LeRobot's batch-to-transition converter.

    Only the dataset-global `index` does, so the step subtracts the episode's start
    offset. If that arithmetic is wrong, chunks are retimed against the wrong labels
    and training still succeeds -- which is why this is asserted against the
    frame_index path rather than merely exercised.
    """
    starts = {0: 0, 1: 100}
    step = DemoSpeedupRetimeStep(labels=labels, episode_starts=starts, low_v=2, high_v=4)
    direct = DemoSpeedupRetimeStep(labels=labels, low_v=2, high_v=4)

    actions = torch.randn(2, HORIZON, DIM)
    via_index = step(transition(actions.clone(), [0, 1], index=[10, 130]))[TransitionKey.ACTION]
    via_frame = direct(transition(actions.clone(), [0, 1], frame_index=[10, 30]))[TransitionKey.ACTION]
    torch.testing.assert_close(via_index, via_frame, rtol=0, atol=0)


def test_global_index_without_starts_is_an_error_not_a_guess(labels):
    step = DemoSpeedupRetimeStep(labels=labels, low_v=2, high_v=4)
    with pytest.raises(ValueError, match="episode_starts"):
        step(transition(torch.randn(1, HORIZON, DIM), [0], index=[10]))


def test_unlabelled_episodes_pass_through(labels):
    """A partially labelled dataset must degrade to ordinary training, not fail."""
    step = DemoSpeedupRetimeStep(labels=labels, low_v=2, high_v=4)
    actions = torch.randn(2, HORIZON, DIM)
    out = step(transition(actions.clone(), [0, 99], frame_index=[10, 0]))[TransitionKey.ACTION]
    assert not torch.equal(out[0], actions[0]), "labelled episode should have been retimed"
    torch.testing.assert_close(out[1], actions[1], rtol=0, atol=0)


def test_no_labels_at_all_is_a_no_op(labels):
    """So --method.type=demospeedup stays selectable before a labelling run exists."""
    actions = torch.randn(2, HORIZON, DIM)
    out = DemoSpeedupRetimeStep()(transition(actions.clone(), [0, 1], frame_index=[0, 0]))
    torch.testing.assert_close(out[TransitionKey.ACTION], actions, rtol=0, atol=0)


def test_chunk_running_past_the_episode_end_pads_with_the_last_label(labels):
    """Zero means *precision*, so zero-padding would brake every final chunk."""
    step = DemoSpeedupRetimeStep(labels=labels, low_v=2, high_v=4)
    near_end = len(labels[0]) - 5
    window = step._label_window(0, near_end, HORIZON)
    assert len(window) == HORIZON
    assert np.all(window[5:] == labels[0][-1])


def test_the_input_batch_is_not_mutated(labels):
    """Training loops reuse batches; retiming in place would corrupt later use."""
    step = DemoSpeedupRetimeStep(labels=labels, low_v=2, high_v=4)
    actions = torch.randn(2, HORIZON, DIM)
    original = actions.clone()
    step(transition(actions, [0, 1], frame_index=[10, 30]))
    torch.testing.assert_close(actions, original, rtol=0, atol=0)


def test_missing_episode_index_is_a_clear_error(labels):
    step = DemoSpeedupRetimeStep(labels=labels)
    with pytest.raises(KeyError, match="episode_index"):
        step({TransitionKey.ACTION: torch.randn(1, HORIZON, DIM), TransitionKey.COMPLEMENTARY_DATA: {}})


def test_pad_mask_is_subsampled_alongside_the_actions(labels):
    step = DemoSpeedupRetimeStep(labels=labels, low_v=2, high_v=4)
    is_pad = torch.zeros(1, HORIZON, dtype=torch.bool)
    is_pad[0, -4:] = True
    out = step(transition(torch.randn(1, HORIZON, DIM), [0], frame_index=[10], is_pad=is_pad.clone()))
    assert out[TransitionKey.COMPLEMENTARY_DATA][ACTION_IS_PAD].shape == is_pad.shape
    assert bool(out[TransitionKey.COMPLEMENTARY_DATA][ACTION_IS_PAD][0, -1]), "tail must stay masked"


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
