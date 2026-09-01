"""`set_flag` refuses to invent a flag crisp_gym does not define.

The bug this guards against is silent: an ``argparse.Namespace`` accepts any
attribute, so a flag renamed under a crisp_gym pin bump would be set on a dead name
while the real one kept the parser's default -- a misconfigured run on live hardware
with no error anywhere. These tests need no robot stack, which is the point of keeping
``deploy_flags`` free of crisp_gym.
"""

from argparse import Namespace
from types import SimpleNamespace

import pytest

from pace_bench.real.configs import resolve_config
from pace_bench.real.deploy_flags import DeployFlagMissing, blend_overlap_for, set_flag


def parser_like() -> Namespace:
    """A stand-in for what `crisp_gym.deploy.cli.build_parser` produces."""
    return Namespace(blend_overlap=4, invert_gripper=False, fps=20.0, stride=1)


class TestSetFlag:
    def test_it_sets_a_flag_the_parser_defines(self):
        args = parser_like()
        set_flag(args, "blend_overlap", 8)
        assert args.blend_overlap == 8

    def test_it_refuses_a_flag_the_parser_does_not_define(self):
        args = parser_like()
        with pytest.raises(DeployFlagMissing):
            set_flag(args, "blnd_overlap", 8)

    def test_the_refusal_names_the_flag_and_where_to_look(self):
        args = parser_like()
        with pytest.raises(DeployFlagMissing) as e:
            set_flag(args, "blend_overlap_frames", 8)
        msg = str(e.value)
        # The operator reads this at bring-up, so it has to say which flag and what to
        # do about it, not just that something was absent.
        assert "--blend-overlap-frames" in msg
        assert "build_parser" in msg
        assert "deploy_args" in msg

    def test_a_refused_flag_is_not_set_at_all(self):
        # The failure must not leave the namespace half-updated: a caller that catches
        # DeployFlagMissing and continues would otherwise deploy with a dead attribute.
        args = parser_like()
        with pytest.raises(DeployFlagMissing):
            set_flag(args, "nonexistent", 8)
        assert not hasattr(args, "nonexistent")

    def test_ours_allows_an_attribute_crisp_gym_never_defines(self):
        # `bspline_gripper_low_v` is this runner's own channel to `deploy_steps`.
        args = parser_like()
        set_flag(args, "bspline_gripper_low_v", 3, ours=True)
        assert args.bspline_gripper_low_v == 3

    def test_ours_is_required_rather_than_implied_by_absence(self):
        # Without the flag, the same call is exactly the failure case -- which is what
        # keeps `ours` documentation rather than a silent escape hatch.
        args = parser_like()
        with pytest.raises(DeployFlagMissing):
            set_flag(args, "bspline_gripper_low_v", 3)

    def test_it_overwrites_rather_than_appends(self):
        args = parser_like()
        set_flag(args, "fps", 30.0)
        set_flag(args, "fps", 40.0)
        assert args.fps == 40.0

    @pytest.mark.parametrize("value", [False, 0, None, "", []])
    def test_falsy_values_are_set_like_any_other(self, value):
        # `hasattr` decides, not truthiness: `--stride 0` is a real setting.
        args = parser_like()
        set_flag(args, "stride", value)
        assert args.stride == value

    def test_an_existing_attribute_set_to_none_still_counts_as_defined(self):
        # argparse leaves a flag with no default as None; that is defined, not missing.
        args = Namespace(policy_path=None)
        set_flag(args, "policy_path", "/ckpt")
        assert args.policy_path == "/ckpt"


@pytest.mark.parametrize("method_type", ["bspline", "demospeedup"])
def test_blend_is_vetoed_for_methods_that_cannot_take_it(method_type):
    assert blend_overlap_for(SimpleNamespace(type=method_type), 4) == 0


@pytest.mark.parametrize("method_type", ["none", "pace"])
def test_blend_is_passed_through_for_methods_that_can(method_type):
    assert blend_overlap_for(SimpleNamespace(type=method_type), 4) == 4


def test_the_veto_beats_the_inherited_default():
    inherited = resolve_config("real/configs/baseline.yaml")["blend"]["overlap"]
    assert inherited > 0
    assert blend_overlap_for(SimpleNamespace(type="bspline"), inherited) == 0
