"""Every dataset an arm trains on carries exactly the features its spec names.

What is actually at stake is narrower than "a stray column reaches the policy",
and `specs.py` documents the measurement: the robot state is taken by exact name
(`observation.state`), so extra *scalar* columns are inert, while `image_features`
returns *every* VISUAL feature, so an extra camera silently adds a backbone -- and
a substituted `observation.state` of the wrong width is read as though it were the
right one. The spec asserts the whole set because the cheap check covers all three
and because a dataset that quietly differs from what a run assumes is otherwise
invisible.

Two halves. The unit tests run everywhere and check the checker. The dataset
tests run against whatever is on this machine and skip per-dataset otherwise,
matching `test_bspline_recorded_dataset.py` -- there is no point asserting about
a recording that is not here, and no point being silent about one that is.
"""

import json
import os
from pathlib import Path

import pytest

from pace_bench.data.specs import SPECS, TRAINING_SETS, DatasetSpec, check, resolve_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PACE_DATA_ROOT", REPO_ROOT.parent / "data"))
DATASETS = DATA_ROOT / "datasets"

UR10E = SPECS["ur10e_cart7"]


def write_dataset(tmp_path: Path, features: dict) -> Path:
    """The smallest thing `check` reads: a meta/info.json with a feature table."""
    (tmp_path / "meta").mkdir(parents=True, exist_ok=True)
    (tmp_path / "meta" / "info.json").write_text(json.dumps({"features": features}))
    return tmp_path


def ur10e_features(**overrides) -> dict:
    features = {
        "observation.images.camera": {"dtype": "video", "shape": [480, 640, 3]},
        "observation.images.d405": {"dtype": "video", "shape": [480, 640, 3]},
        "observation.state": {"dtype": "float32", "shape": [6]},
        "action": {"dtype": "float32", "shape": [7]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
    }
    features.update(overrides)
    return features


class TestTheChecker:
    def test_a_dataset_matching_its_spec_passes(self, tmp_path):
        check(write_dataset(tmp_path, ur10e_features()), UR10E)

    def test_an_undeclared_input_fails_and_is_named(self, tmp_path):
        """The real case: the three wall-clock columns and the duplicate state.

        These four are the inert kind -- listed in `input_features`, never read,
        because the state is taken by exact name. Flagged anyway: they are the
        signature of a merge that copied columns instead of naming them, and the
        next thing such a merge carries through might be a camera.
        """
        root = write_dataset(
            tmp_path,
            ur10e_features(
                **{
                    "observation.timestamps.wall": {"dtype": "float64", "shape": [1]},
                    "observation.timestamps.camera_header": {"dtype": "float64", "shape": [1]},
                    "observation.timestamps.d405_header": {"dtype": "float64", "shape": [1]},
                    "observation.state.cartesian": {"dtype": "float32", "shape": [6]},
                }
            ),
        )
        with pytest.raises(ValueError) as excinfo:
            check(root, UR10E)
        message = str(excinfo.value)
        assert "4 input(s)" in message
        for key in (
            "observation.timestamps.wall",
            "observation.timestamps.camera_header",
            "observation.timestamps.d405_header",
            "observation.state.cartesian",
        ):
            assert key in message, f"{key} is the problem and must be in the message"

    def test_bookkeeping_columns_are_not_inputs(self, tmp_path):
        """`timestamp` is LeRobot's own frame time, not an `observation.` key.

        Worth pinning: it is one letter away from `observation.timestamps.*` and
        is present in every dataset, so a checker that flagged it would fire on
        everything and be turned off within a day.
        """
        check(write_dataset(tmp_path, ur10e_features()), UR10E)

    def test_a_missing_input_fails(self, tmp_path):
        features = ur10e_features()
        del features["observation.images.d405"]
        with pytest.raises(ValueError, match="observation.images.d405"):
            check(write_dataset(tmp_path, features), UR10E)

    def test_a_substituted_state_width_fails(self, tmp_path):
        """The raw recordings carry a 13-dim joints+cart+gripper bundle under this
        same name; the names would match and only the width gives it away."""
        root = write_dataset(
            tmp_path, ur10e_features(**{"observation.state": {"dtype": "float32", "shape": [13]}})
        )
        with pytest.raises(ValueError, match=r"observation\.state of shape \(13,\)"):
            check(root, UR10E)

    def test_a_wrong_action_width_fails(self, tmp_path):
        root = write_dataset(
            tmp_path, ur10e_features(**{"action": {"dtype": "float32", "shape": [20]}})
        )
        with pytest.raises(ValueError, match=r"action of shape \(20,\)"):
            check(root, UR10E)

    def test_an_unknown_spec_name_fails(self):
        with pytest.raises(ValueError, match="unknown dataset spec"):
            resolve_spec("ur10e_cart6")

    def test_every_training_set_names_a_real_spec(self):
        """A typo here would otherwise only surface on the machine holding the data."""
        for dataset, spec in TRAINING_SETS.items():
            assert spec in SPECS, f"{dataset} names spec {spec!r}, which does not exist"


@pytest.mark.parametrize("relative,spec_name", sorted(TRAINING_SETS.items()))
def test_the_recorded_training_sets_match_their_specs(relative: str, spec_name: str):
    root = DATASETS / relative
    if not root.is_dir():
        pytest.skip(f"{root} is not on this machine")
    check(root, resolve_spec(spec_name))


def test_the_dataset_the_rule_came_from_still_fails():
    """`stackcups_20260829_merged` is kept, unfixed, as the archive of the recording.

    Its cleaned sibling is what trains -- for leanness, not for correctness: all
    four extra columns are scalars the policy never reads. Asserting the raw one
    *fails* keeps the example honest, and keeps the difference between the two
    names visible.
    """
    root = DATASETS / "real" / "stackcups_20260829_merged"
    if not root.is_dir():
        pytest.skip(f"{root} is not on this machine")
    with pytest.raises(ValueError, match="observation.timestamps.wall"):
        check(root, SPECS["ur10e_cart7"])


def test_the_specs_are_frozen():
    """They are shared module state; a run that mutated one would change every
    later check in the same process."""
    with pytest.raises(Exception):
        SPECS["ur10e_cart7"].state_dim = 7  # type: ignore[misc]
    assert isinstance(SPECS["ur10e_cart7"], DatasetSpec)
