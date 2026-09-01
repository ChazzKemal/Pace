"""The config picker: what it offers, and when it refuses to appear.

The curses screen itself is not tested here -- it needs a terminal, and driving one
in CI buys less than it costs. What is tested is everything a wrong answer would make
dangerous: which configs are offered, which are correctly withheld, and the four cases
where the picker must not appear at all because argv already said what to run.
"""

import sys
from pathlib import Path

import pytest

from pace_bench.real.picker import (
    NO_PICKER_FLAG,
    ConfigChoice,
    TaskChoice,
    discover,
    discover_tasks,
    maybe_pick_config,
)

CONFIGS = Path("real/configs")
TASKS = Path("real/configs/tasks.yaml")


class TestDiscover:
    def test_it_offers_the_runnable_configs(self):
        names = {c.name for c in discover(CONFIGS)}
        assert {"baseline", "pace_fast", "demospeedup"} <= names

    def test_it_withholds_the_include_base_and_the_registry(self):
        # deploy_defaults.yaml is inherited by every config and tasks.yaml is the
        # checkpoint registry. Neither is launchable; offering them would be offering
        # a run that cannot start.
        names = {c.name for c in discover(CONFIGS)}
        assert "deploy_defaults" not in names
        assert "tasks" not in names

    def test_every_choice_carries_a_summary(self):
        # The summary is the file's first comment line, which is the only reason an
        # operator can tell two configs apart without opening them.
        assert all(c.summary for c in discover(CONFIGS))

    def test_a_blocked_config_is_offered_but_flagged(self):
        # bspline_fast refuses at deploy_steps until the gripper compensation is
        # stated. It stays in the list: a config that cannot run is what the operator
        # most needs telling about, and hiding it looks like a missing file.
        blocked = {c.name: c.blocked for c in discover(CONFIGS) if c.blocked}
        assert "bspline_fast" in blocked
        assert "bspline_low_v" in blocked["bspline_fast"]

    def test_the_flag_tracks_the_config_not_a_comment(self, tmp_path):
        # Same method, compensation stated -> not blocked. The check mirrors the real
        # guard, so a config that stops being blocked stops being flagged.
        (tmp_path / "ok.yaml").write_text(
            "# A bspline run with the gripper stated.\n"
            "method:\n  type: bspline\n  num_actions: 16\n"
            "gripper:\n  slowdown_frames: 5\n  bspline_low_v: 3\n"
        )
        assert discover(tmp_path)[0].blocked == ""

    def test_an_unreadable_config_is_skipped_not_fatal(self, tmp_path):
        (tmp_path / "broken.yaml").write_text("_include: nope-does-not-exist.yaml\n")
        (tmp_path / "fine.yaml").write_text("# fine\nmethod:\n  type: none\n")
        assert [c.name for c in discover(tmp_path)] == ["fine"]

    def test_no_configs_is_an_empty_list_not_an_error(self, tmp_path):
        assert discover(tmp_path) == []


class TestMaybePickConfig:
    """Every path here must return without drawing anything."""

    def test_a_pipe_passes_argv_straight_through(self):
        # The load-bearing case: scripts, CI and nohup runs must behave exactly as
        # they did before the picker existed. A menu there would hang at bring-up.
        argv = ["run_real.py", "--fps=20"]
        assert maybe_pick_config(argv) == argv

    def test_an_explicit_config_path_wins(self, monkeypatch):
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
        argv = ["run_real.py", "--config_path=real/configs/pace_fast.yaml"]
        assert maybe_pick_config(argv) == argv

    def test_a_separated_config_path_also_wins(self, monkeypatch):
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
        argv = ["run_real.py", "--config_path", "real/configs/pace_fast.yaml"]
        assert maybe_pick_config(argv) == argv

    def test_the_opt_out_flag_is_stripped_before_draccus_sees_it(self):
        # draccus would reject an unknown flag, so opting out has to remove it.
        assert maybe_pick_config(["run_real.py", NO_PICKER_FLAG]) == ["run_real.py"]

    def test_nothing_to_choose_from_passes_through(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
        argv = ["run_real.py"]
        assert maybe_pick_config(argv, tmp_path) == argv


class TestSelectionBecomesArgv:
    """What a selection turns into, without opening a terminal."""

    @pytest.fixture
    def interactive(self, monkeypatch):
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)

    def _pick(self, monkeypatch, result, seen=None):
        def fake(choices, registry=None, ask_task=True):
            if seen is not None:
                seen["ask_task"] = ask_task
            return result
        monkeypatch.setattr("pace_bench.real.picker.pick", fake)

    def test_an_explicit_task_is_not_asked_about_again(self, interactive, monkeypatch):
        # Otherwise argv would carry two --task flags saying different things.
        seen = {}
        self._pick(monkeypatch, (Path("real/configs/pace_fast.yaml"), None, False), seen)
        maybe_pick_config(["run_real.py", "--task=stackcups"])
        assert seen["ask_task"] is False

    def test_with_no_task_given_the_task_step_is_offered(self, interactive, monkeypatch):
        seen = {}
        self._pick(monkeypatch, (Path("real/configs/pace_fast.yaml"), None, False), seen)
        maybe_pick_config(["run_real.py"])
        assert seen["ask_task"] is True

    def test_a_choice_appends_the_config_path(self, interactive, monkeypatch):
        self._pick(monkeypatch, (Path("real/configs/pace_fast.yaml"), None, False))
        out = maybe_pick_config(["run_real.py"])
        assert out == ["run_real.py", "--config_path=real/configs/pace_fast.yaml"]

    def test_a_chosen_task_appends_the_task_flag(self, interactive, monkeypatch):
        # The config supplies how to deploy; the task supplies what to deploy it on.
        # Without this the run starts with policy_path empty and no checkpoint.
        task = TaskChoice("pickplace", Path("/ckpt"), "")
        self._pick(monkeypatch, (Path("real/configs/pace_fast.yaml"), task, False))
        assert "--task=pickplace" in maybe_pick_config(["run_real.py"])

    def test_the_dry_run_box_appends_the_flag(self, interactive, monkeypatch):
        self._pick(monkeypatch, (Path("real/configs/pace_fast.yaml"), None, True))
        assert maybe_pick_config(["run_real.py"])[-1] == "--dry_run=true"

    def test_cancelling_returns_none_so_the_caller_launches_nothing(
            self, interactive, monkeypatch):
        # Not "fall back to a default": pressing escape and asking for the baseline
        # are different intents, and only one of them moves an arm.
        self._pick(monkeypatch, None)
        assert maybe_pick_config(["run_real.py"]) is None

    def test_existing_args_are_preserved(self, interactive, monkeypatch):
        self._pick(monkeypatch, (Path("real/configs/baseline.yaml"), None, False))
        out = maybe_pick_config(["run_real.py", "--task=stackcups"])
        assert out[:2] == ["run_real.py", "--task=stackcups"]


class TestDiscoverTasks:
    def test_it_offers_the_tasks_trained_for_the_method(self):
        ok = {t.name for t in discover_tasks("pace", TASKS) if not t.blocked}
        assert {"pickplace", "stackcups"} <= ok

    def test_an_untrained_pair_is_listed_but_blocked(self):
        # stackcups has no bspline arm. Listing it answers the operator's question;
        # omitting it would read as a missing task rather than an untrained pair.
        by_name = {t.name: t for t in discover_tasks("bspline", TASKS)}
        assert "stackcups" in by_name
        assert by_name["stackcups"].blocked
        assert by_name["stackcups"].path is None

    def test_the_block_reason_names_what_IS_trained(self):
        by_name = {t.name: t for t in discover_tasks("bspline", TASKS)}
        assert "pace" in by_name["stackcups"].blocked

    def test_a_missing_registry_is_an_empty_list_not_an_error(self, tmp_path):
        assert discover_tasks("pace", tmp_path / "nope.yaml") == []

    def test_a_checkpoint_absent_from_this_machine_is_blocked_differently(self, tmp_path):
        # "not trained" and "trained but not here" are different problems with
        # different fixes, so they must not read the same.
        reg = tmp_path / "tasks.yaml"
        reg.write_text("root: /nonexistent\nfoo:\n  pace: some/ckpt\n")
        (task,) = discover_tasks("pace", reg)
        assert "not on this machine" in task.blocked


class TestNeedsTask:
    def test_a_config_without_a_policy_path_needs_one(self):
        assert all(c.needs_task for c in discover(CONFIGS))

    def test_an_explicit_policy_path_skips_the_task_step(self, tmp_path):
        # apply_task ignores --task when policy_path is set, so asking would be a
        # question whose answer is discarded.
        (tmp_path / "c.yaml").write_text(
            "# explicit\nmethod:\n  type: none\npolicy_path: /some/ckpt\n")
        assert discover(tmp_path)[0].needs_task is False


def test_choice_is_hashable_and_comparable():
    # frozen dataclass: the screen keeps them in a list and compares by value.
    a = ConfigChoice(Path("x"), "x", "none", "1x", "s", "")
    assert a == ConfigChoice(Path("x"), "x", "none", "1x", "s", "")
    assert {a}
