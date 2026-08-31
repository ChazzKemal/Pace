"""Turning recordings into training sets.

The seam this package sits on: `ur10_clearpath` produces recordings, pace_bench
turns them into something a policy trains on. Nothing here touches the robot --
it reads and writes LeRobot datasets, the object the rest of this repo is built
around, which is why it lives beside the trainer and shares its pinned LeRobot.
"""

from pace_bench.data.specs import SPECS, TRAINING_SETS, DatasetSpec, check, resolve_spec

__all__ = ["SPECS", "TRAINING_SETS", "DatasetSpec", "check", "resolve_spec"]
