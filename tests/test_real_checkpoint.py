"""Tests for reading what a checkpoint says about how it was trained.

The parser is checked against a *real* LeRobot checkpoint as well as synthetic ones,
because the serialization detail that matters -- steps are keyed by ``registry_name``
-- is not guessable. The plausible-looking ``registered_name`` finds nothing, and the
failure is silent: a demospeedup checkpoint would be reported as an unmodified
baseline and deployed without its gripper compensation.
"""

import json
from pathlib import Path

import pytest

from pace_bench.real.checkpoint import (
    BSPLINE_DECODE_STEP,
    CheckpointFacts,
    MethodMismatch,
    read_checkpoint,
    validate_method,
    without_postprocessor_steps,
)

REAL = Path("/home/ali/Coding/Robot_Control/Yunfei/crisp_gym/outputs/train/smolvla/pretrained_model")


def write_ckpt(tmp, *, method=None, chunk_size=50, n_action_steps=50,
               policy_type="act", pre_steps=()):
    d = tmp / "pretrained_model"
    d.mkdir(parents=True, exist_ok=True)
    train = {"policy": {"type": policy_type}}
    if method is not None:
        train["method"] = method
    (d / "train_config.json").write_text(json.dumps(train))
    (d / "config.json").write_text(json.dumps(
        {"type": policy_type, "chunk_size": chunk_size, "n_action_steps": n_action_steps}))
    (d / "policy_preprocessor.json").write_text(json.dumps(
        {"name": "policy_preprocessor",
         "steps": [{"registry_name": n, "config": c} for n, c in pre_steps]}))
    return d


# --------------------------------------------------------------------------
# Against real serialization
# --------------------------------------------------------------------------

@pytest.mark.skipif(not REAL.exists(), reason="no local LeRobot checkpoint")
def test_parses_a_real_lerobot_checkpoint():
    """Guards the registry_name detail; a wrong key yields zero steps, silently."""
    f = read_checkpoint(REAL)
    assert f.policy_type == "smolvla"
    assert f.chunk_size == 50
    assert len(f.built_steps) > 0, "steps must be discoverable in real serialization"
    assert "device_processor" in f.built_steps
    assert f.method_type is None, "this one was not trained through pace_bench"


# --------------------------------------------------------------------------
# demospeedup detection
# --------------------------------------------------------------------------

def test_reads_the_method_block(tmp_path):
    d = write_ckpt(tmp_path, method={"type": "demospeedup", "low_v": 2, "high_v": 4,
                                     "halve_chunk": True, "source_chunk": 100})
    f = read_checkpoint(d)
    assert (f.method_type, f.low_v, f.high_v) == ("demospeedup", 2, 4)
    assert f.source_chunk == 100 and f.chunk_size == 50


def test_detects_that_halving_was_applied(tmp_path):
    """chunk 50 against source 100 is the evidence adjust_policy ran."""
    d = write_ckpt(tmp_path, chunk_size=50,
                   method={"type": "demospeedup", "low_v": 2, "source_chunk": 100})
    assert read_checkpoint(d).halving_applied is True


def test_halving_unknown_when_source_chunk_absent(tmp_path):
    """A pre-adjustment log has source_chunk=null; that is not evidence either way."""
    d = write_ckpt(tmp_path, chunk_size=100,
                   method={"type": "demospeedup", "low_v": 2, "source_chunk": None})
    assert read_checkpoint(d).halving_applied is None


def test_low_v_from_the_built_pipeline_wins_over_the_request(tmp_path):
    """The preprocessor records what was built; train_config what was asked for."""
    d = write_ckpt(tmp_path,
                   method={"type": "demospeedup", "low_v": 9},
                   pre_steps=[("demospeedup_retime", {"low_v": 2, "high_v": 4})])
    assert read_checkpoint(d).low_v == 2


def test_built_steps_are_discovered(tmp_path):
    d = write_ckpt(tmp_path, pre_steps=[("normalizer_processor", {}),
                                        ("demospeedup_retime", {"low_v": 2})])
    assert "demospeedup_retime" in read_checkpoint(d).built_steps


# --------------------------------------------------------------------------
# Tolerance
# --------------------------------------------------------------------------

def test_missing_files_do_not_raise(tmp_path):
    d = tmp_path / "empty"; d.mkdir()
    f = read_checkpoint(d)
    assert f.method_type is None and f.built_steps == () and f.chunk_size is None


def test_diffusion_horizon_is_read_as_chunk_size(tmp_path):
    d = tmp_path / "pretrained_model"; d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"type": "diffusion", "horizon": 32}))
    assert read_checkpoint(d).chunk_size == 32


# --------------------------------------------------------------------------
# validate_method -- conflicts are errors because both present as a bad robot
# --------------------------------------------------------------------------

def facts(mt, low_v=2):
    return CheckpointFacts(path=Path("/x"), method_type=mt, low_v=low_v)


def test_matching_method_passes():
    validate_method("demospeedup", facts("demospeedup"))
    validate_method("none", facts(None))


def test_none_on_a_demospeedup_checkpoint_is_refused():
    with pytest.raises(MethodMismatch, match="gripper compensation"):
        validate_method("none", facts("demospeedup"))


def test_pace_on_a_demospeedup_checkpoint_is_refused():
    with pytest.raises(MethodMismatch, match="already"):
        validate_method("pace", facts("demospeedup"))


def test_force_overrides():
    validate_method("none", facts("demospeedup"), force=True)


def test_pace_on_an_untrained_checkpoint_is_fine():
    """PACE is an eval-time choice; nothing in a checkpoint forbids it."""
    validate_method("pace", facts(None))


def test_demospeedup_on_a_baseline_checkpoint_is_refused():
    """The gap found while checking what the local checkpoint permits.

    Applying demospeedup's row replication to weights that were never trained on
    retimed targets slows the grasp for no reason and produces a benchmark number
    that is not comparable to a real demospeedup arm. Nothing about it is unsafe,
    which is exactly why it needs to be caught -- it would look like it worked.
    """
    with pytest.raises(MethodMismatch, match="not retimed"):
        validate_method("demospeedup", facts(None))
    with pytest.raises(MethodMismatch, match="not retimed"):
        validate_method("demospeedup", facts("none"))


def test_demospeedup_on_a_baseline_can_be_forced():
    validate_method("demospeedup", facts(None), force=True)


def test_pace_stays_exempt():
    """PACE is eval-time: no checkpoint forbids it, including a demospeedup one's peers."""
    validate_method("pace", facts(None))
    validate_method("pace", facts("none"))


class TestWithoutPostprocessorSteps:
    """The deploy pipeline owns decoding, so the checkpoint's own decode must go.

    A B-spline checkpoint ships a `bspline_decode` with `num_actions` frozen at
    training. `inference_worker` rebuilds the pipeline from the path in a spawned
    subprocess, so filtering it here -- by handing over a different path -- is the
    only filter that reaches it.
    """

    def manifest(self, tmp_path, steps):
        (tmp_path / "policy_postprocessor.json").write_text(
            json.dumps({"name": "policy_postprocessor", "steps": steps}))
        (tmp_path / "model.safetensors").write_text("weights")
        (tmp_path / "config.json").write_text("{}")
        return tmp_path

    def test_the_named_step_is_removed(self, tmp_path):
        src = self.manifest(tmp_path, [
            {"registry_name": "unnormalizer_processor"},
            {"registry_name": "bspline_decode", "config": {"num_actions": 16}},
        ])
        out = without_postprocessor_steps(src, (BSPLINE_DECODE_STEP,))
        kept = json.loads((out / "policy_postprocessor.json").read_text())["steps"]
        assert [s["registry_name"] for s in kept] == ["unnormalizer_processor"]

    def test_a_checkpoint_without_it_is_returned_untouched(self, tmp_path):
        # Inert for every other method: no temp directory, no copy, same path.
        src = self.manifest(tmp_path, [{"registry_name": "unnormalizer_processor"}])
        assert without_postprocessor_steps(src, (BSPLINE_DECODE_STEP,)) == src

    def test_a_checkpoint_with_no_manifest_is_returned_untouched(self, tmp_path):
        assert without_postprocessor_steps(tmp_path, (BSPLINE_DECODE_STEP,)) == tmp_path

    def test_every_other_file_is_reachable(self, tmp_path):
        # The subprocess loads weights and config from the view, so it must be whole.
        src = self.manifest(tmp_path, [{"registry_name": "bspline_decode"}])
        out = without_postprocessor_steps(src, (BSPLINE_DECODE_STEP,))
        assert (out / "model.safetensors").read_text() == "weights"
        assert (out / "config.json").exists()

    def test_the_weights_are_not_copied(self, tmp_path):
        # model.safetensors is gigabytes; only the JSON differs.
        src = self.manifest(tmp_path, [{"registry_name": "bspline_decode"}])
        out = without_postprocessor_steps(src, (BSPLINE_DECODE_STEP,))
        assert (out / "model.safetensors").is_symlink()
        assert not (out / "policy_postprocessor.json").is_symlink()

    def test_the_original_checkpoint_is_not_modified(self, tmp_path):
        src = self.manifest(tmp_path, [{"registry_name": "bspline_decode"}])
        before = (src / "policy_postprocessor.json").read_text()
        without_postprocessor_steps(src, (BSPLINE_DECODE_STEP,))
        assert (src / "policy_postprocessor.json").read_text() == before
