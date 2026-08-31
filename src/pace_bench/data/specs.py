"""What a policy is allowed to consume from a training set, named per dataset.

LeRobot classifies **every** key beginning `observation.` as a policy input, with
no filter and no warning::

    elif key.startswith(OBS_STR):
        type = FeatureType.STATE          # utils/feature_utils.py:170

and `make_policy` builds the policy's input projections from that same set
(`policies/factory.py:333-335`), so whatever survived recording and conversion
becomes something the network reads. Nothing reports it. `DatasetConfig` has no
exclusion flag either -- `rename_map` only renames, and refuses to run without a
pretrained checkpoint -- so the dataset is the only place to decide.

That went wrong once, on `stackcups_20260829_merged`: it carried three absolute
wall-clock columns and a byte-identical duplicate of `observation.state`, so an
ACT baseline trained for 39k steps on seven inputs instead of three. The
timestamps are monotonic across the whole session, which makes them an episode
index and a within-episode clock during training -- a shortcut around the
cameras -- and their std is 6.5e3 seconds, so a deployment one day later feeds
that channel a value 13 sigma outside anything the normalizer saw.

The fix is not a filter at load time; it is naming what a training set contains,
so that a dataset can be *checked* rather than trusted. This module holds the
names. It is the same construction as `bspline.layout.resolve_layout`, one level
up: a registry entry checked against what the dataset really is, because getting
it wrong is otherwise silent.
"""

from dataclasses import dataclass
from pathlib import Path

ACTION = "action"
STATE = "observation.state"


@dataclass(frozen=True)
class DatasetSpec:
    """The exact set of features a policy trained on this data may see.

    Images are pinned by name only: a re-record at a different resolution is
    still the same training set. The vector features are pinned by width as
    well, because that is where a silent substitution actually hurts -- the raw
    UR10e recordings carry a 13-dim joints+cartesian+gripper bundle under the
    same `observation.state` name as the 6-dim cartesian pose.
    """

    #: Every `observation.*` key the dataset may carry. Exactly these, no others.
    inputs: tuple[str, ...]
    #: Width of `observation.state`.
    state_dim: int
    #: Width of `action`.
    action_dim: int


SPECS: dict[str, DatasetSpec] = {
    # UR10e + Robotiq, absolute cartesian. `action` is [x, y, z, rx, ry, rz, gripper];
    # the gripper is output-only, which is what "nogrip" means in the dataset names.
    "ur10e_cart7": DatasetSpec(
        inputs=(
            "observation.images.camera",
            "observation.images.d405",
            "observation.state",
        ),
        state_dim=6,
        action_dim=7,
    ),
    # LIBERO-10, end-effector 6D: xyz + rot6d + gripper in the first 10 columns,
    # zero-padded to 20.
    "libero_ee6d": DatasetSpec(
        inputs=(
            "observation.images.image",
            "observation.images.image2",
            "observation.state",
        ),
        state_dim=20,
        action_dim=20,
    ),
}


#: Every dataset an arm trains on, and the spec it has to satisfy. Adding a
#: dataset here is what puts it under the check -- one line, and forgetting it is
#: the only hole left. Paths are relative to `data/datasets/`.
TRAINING_SETS: dict[str, str] = {
    "real/pickplace_cart7_v2_angleaxis_nogrip": "ur10e_cart7",
    "real/stackcups_20260829_merged_clean": "ur10e_cart7",
    "sim/libero_10_ee6d": "libero_ee6d",
}


def resolve_spec(name: str) -> DatasetSpec:
    """Look a spec up by name, failing loudly on a typo rather than matching nothing."""
    if name not in SPECS:
        raise ValueError(f"unknown dataset spec {name!r}; known: {sorted(SPECS)}")
    return SPECS[name]


def _features(dataset_root: Path) -> dict:
    """The dataset's declared features, read the way LeRobot reads them."""
    import json

    info = Path(dataset_root) / "meta" / "info.json"
    if not info.is_file():
        raise FileNotFoundError(f"not a LeRobot dataset: {dataset_root}")
    return json.loads(info.read_text())["features"]


def check(dataset_root: Path, spec: DatasetSpec) -> None:
    """Raise if this dataset would feed a policy anything the spec does not name.

    Reads `meta/info.json` rather than instantiating a `LeRobotDataset`, so it
    costs milliseconds and needs no video backend -- the feature declaration is
    the whole input to LeRobot's own classification, so nothing is lost by
    reading it directly.
    """
    features = _features(dataset_root)
    actual = tuple(k for k in features if k.startswith("observation."))
    problems = []

    undeclared = [k for k in actual if k not in spec.inputs]
    if undeclared:
        problems.append(
            "would feed the policy "
            + f"{len(undeclared)} input(s) the spec does not name:\n"
            + "\n".join(
                f"      {k:<40} {features[k]['dtype']} {tuple(features[k]['shape'])}"
                for k in undeclared
            )
        )
    missing = [k for k in spec.inputs if k not in actual]
    if missing:
        problems.append("is missing spec input(s): " + ", ".join(missing))

    for key, want in ((STATE, spec.state_dim), (ACTION, spec.action_dim)):
        if key in features:
            got = tuple(features[key]["shape"])
            if got != (want,):
                problems.append(f"has {key} of shape {got}, spec says ({want},)")

    if problems:
        raise ValueError(
            f"{Path(dataset_root).name} does not match its spec -- it "
            + "\n  and it ".join(problems)
            + f"\n  spec inputs: {', '.join(spec.inputs)}"
        )
