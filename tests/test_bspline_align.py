"""Chunk-boundary time alignment.

The question alignment answers is "where on this new curve is the arm already?", and
every test here is a way of getting that wrong: matching the wrong point, matching a
point too far along, being led astray by the gripper, or claiming a match that is not
there. Upstream's `_align_new_plan` / `_find_closest_t_to_target`
(`scripts/policy_local_bspline.py` @ 61ed5f4) is the reference.
"""

import numpy as np
import pytest
from scipy.interpolate import BSpline

from pace_bench.methods.bspline.spline import (
    ALIGN_ERROR_THRESHOLD,
    ALIGN_WINDOW,
    align_start,
)


def straight_curve(dim=10, t0=0.0, t1=30.0, n_ctrl=12, degree=3):
    """A curve whose value at t is trivially predictable: each column ramps."""
    knots = np.concatenate([
        np.full(degree, t0),
        np.linspace(t0, t1, n_ctrl - degree + 1),
        np.full(degree, t1),
    ])
    c = np.linspace(np.zeros(dim), np.arange(1, dim + 1, dtype=float), n_ctrl)
    return BSpline(knots, c, degree, extrapolate=False), t0, t1


class TestFindsThePoint:
    def test_an_exact_match_is_found(self):
        curve, lo, hi = straight_curve()
        target_t = lo + 0.1 * (hi - lo)          # inside the 20% window
        t, error = align_start(curve, curve(target_t), lo, hi)
        assert t == pytest.approx(target_t, abs=0.2)
        assert error < 1e-3

    def test_the_start_itself_aligns_to_the_start(self):
        curve, lo, hi = straight_curve()
        t, error = align_start(curve, curve(lo), lo, hi)
        assert t == pytest.approx(lo, abs=1e-6)
        assert error < 1e-6

    def test_a_curve_that_doubles_back_picks_the_earlier_crossing(self):
        # Two points on this path have the same position. Alignment must resume at the
        # first, not the second: a bounded local method started blind can land in
        # either basin, which is what the bracketing scan exists to prevent.
        lo, hi, degree = 0.0, 40.0, 3
        n = 16
        knots = np.concatenate([np.full(degree, lo),
                                np.linspace(lo, hi, n - degree + 1),
                                np.full(degree, hi)])
        s = np.linspace(0, 1, n)
        c = np.zeros((n, 10))
        c[:, 0] = np.sin(2 * np.pi * s)          # out and back
        curve = BSpline(knots, c, degree, extrapolate=False)
        early = lo + 0.05 * (hi - lo)
        t, error = align_start(curve, curve(early), lo, hi)
        assert error < ALIGN_ERROR_THRESHOLD
        assert t < lo + 0.5 * (hi - lo)


class TestDeclinesRatherThanGuesses:
    def test_no_match_falls_back_to_the_start(self):
        curve, lo, hi = straight_curve()
        far = curve(lo) + 1000.0
        t, error = align_start(curve, far, lo, hi)
        assert t == lo
        assert error > ALIGN_ERROR_THRESHOLD

    def test_a_match_beyond_the_window_is_not_taken(self):
        # The arm is genuinely at 60% along, but resuming there would skip most of the
        # plan, so alignment declines and the chunk starts at its own beginning.
        curve, lo, hi = straight_curve()
        late = lo + 0.6 * (hi - lo)
        assert 0.6 > ALIGN_WINDOW
        t, _ = align_start(curve, curve(late), lo, hi)
        assert t == lo

    def test_a_zero_length_window_declines(self):
        curve, lo, _ = straight_curve()
        t, error = align_start(curve, curve(lo), lo, lo)
        assert t == lo
        assert error == float("inf")


class TestTheGripperIsExcluded:
    def test_a_flipped_gripper_does_not_move_the_match(self):
        # The gripper is near-binary: it says almost nothing about *where along the
        # path* the arm is, and its one big step would dominate a distance that
        # included it. Upstream's `consider_gripper_during_align` is off by default.
        curve, lo, hi = straight_curve(dim=10)
        target_t = lo + 0.1 * (hi - lo)
        anchor = np.asarray(curve(target_t)).copy()
        anchor[9] += 1.0                          # gripper disagrees by a full stroke
        t, error = align_start(curve, anchor, lo, hi, compare_dim=9)
        assert t == pytest.approx(target_t, abs=0.2)
        assert error < 1e-3

    def test_including_the_gripper_rejects_that_same_match(self):
        curve, lo, hi = straight_curve(dim=10)
        target_t = lo + 0.1 * (hi - lo)
        anchor = np.asarray(curve(target_t)).copy()
        anchor[9] += 1.0
        t, error = align_start(curve, anchor, lo, hi, compare_dim=None)
        assert error > ALIGN_ERROR_THRESHOLD
        assert t == lo


# --------------------------------------------------------------------------
# Step 2: decode_chunk uses it
# --------------------------------------------------------------------------

from pace_bench.methods.bspline.spline import DEGREE, decode_chunk  # noqa: E402


def ramp_matrix(width=16, span_per_knot=1.6, dim=10):
    """A parameter matrix whose curve moves steadily, so t is readable from position."""
    m = np.zeros((width, 1 + dim))
    m[:, 0] = np.arange(width) * span_per_knot
    m[:, 1:] = np.linspace(np.zeros(dim), np.full(dim, 2.0), width)
    return m


class TestDecodeChunkAlignment:
    def test_no_anchor_is_byte_identical_to_before(self):
        # Alignment must be inert when unused: training and LIBERO eval call this same
        # function and pass no anchor.
        m = ramp_matrix()
        assert np.array_equal(decode_chunk(m, 12), decode_chunk(m, 12, align_to=None))

    def test_an_anchor_moves_the_first_sample_onto_it(self):
        m = ramp_matrix()
        base = decode_chunk(m, 12)
        anchor = base[1]                              # arm reached the 2nd sample
        aligned = decode_chunk(m, 12, align_to=anchor, compare_dim=9)
        assert np.allclose(aligned[0][:9], anchor[:9], atol=1e-2)
        assert not np.allclose(aligned[0], base[0])

    def test_the_rate_is_preserved_and_the_row_count_absorbs_it(self):
        # The point of the design decision: speed must not become a function of
        # tracking error, so spacing stays put and the count shrinks.
        m = ramp_matrix()
        base = decode_chunk(m, 12)
        knots = m[:, 0]
        span = knots[-(DEGREE + 1)] - knots[DEGREE]
        rate = span / 11
        aligned = decode_chunk(m, 12, align_to=base[2], compare_dim=9)
        assert len(aligned) < len(base)
        # step size in curve-time is what "rate" means; recover it from the samples
        step_base = np.linalg.norm(base[1] - base[0])
        step_aligned = np.linalg.norm(aligned[1] - aligned[0])
        assert step_aligned == pytest.approx(step_base, rel=0.15)
        assert rate > 0

    def test_an_unmatchable_anchor_decodes_exactly_as_if_unaligned(self):
        m = ramp_matrix()
        far = np.full(10, 1e6)
        assert np.array_equal(decode_chunk(m, 12), decode_chunk(m, 12, align_to=far))

    def test_a_single_action_request_ignores_alignment(self):
        m = ramp_matrix()
        assert decode_chunk(m, 1, align_to=ramp_matrix()[0, 1:]).shape[0] == 1

    def test_the_end_of_the_span_is_still_never_the_zero_vector(self):
        # The nextafter guard must survive the rewrite: a tail chunk repeats its last
        # knot, and evaluating exactly at `end` there returns all zeros.
        m = ramp_matrix()
        m[-4:, 0] = m[-4, 0]                          # pad the tail, as chunking does
        out = decode_chunk(m, 8)
        assert not np.allclose(out[-1], 0.0)


# --------------------------------------------------------------------------
# Step 3: the decode step carries the anchor between chunks
# --------------------------------------------------------------------------

import torch  # noqa: E402

from pace_bench.methods.bspline.layout import resolve_layout  # noqa: E402
from pace_bench.methods.bspline.processor import BSplineDecodeStep  # noqa: E402


def pose_matrix(width=16, span_per_knot=1.6, turn=0.35):
    """A parameter matrix whose rot6d columns are a real rotation.

    `ramp_matrix` ramps every column alike, which is fine for the pure search but
    makes both rows of the 6D rotation identical -- `from_spline_actions` then
    Gram-Schmidts a degenerate frame. Anything that decodes to cart7 needs this one.
    """
    m = np.zeros((width, 11))
    m[:, 0] = np.arange(width) * span_per_knot
    m[:, 1:4] = np.linspace([0.4, -0.1, 0.35], [0.7, 0.1, 0.15], width)
    theta = np.linspace(0.0, turn, width)
    for i, th in enumerate(theta):
        r = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0]])
        m[i, 4:10] = r.reshape(6)
    m[:, 10] = 1.0
    return m


def batched(m):
    return torch.from_numpy(np.asarray(m, dtype=np.float64))[None]


class TestDecodeStepAnchor:
    def step(self, **kw):
        return BSplineDecodeStep(num_actions=12, layout=resolve_layout("cart7", 7), **kw)

    def test_the_first_chunk_has_nothing_to_align_to(self):
        s = self.step(align=True)
        plain = self.step(align=False)
        m = batched(pose_matrix())
        assert torch.allclose(s.decode_batch(m)[0], plain.decode_batch(m)[0])

    def test_the_second_chunk_aligns_to_the_first(self):
        s = self.step(align=True)
        m = batched(pose_matrix())
        s.decode_batch(m)
        assert s._anchor is not None
        second = s.decode_batch(m)[0]
        # Same matrix twice: the anchor is the previous tail, which lies past the
        # window, so alignment declines and the row count is unchanged.
        assert second.shape[1] == 12

    def test_a_partly_advanced_arm_shortens_the_next_chunk(self):
        s = self.step(align=True)
        early = pose_matrix()
        s.decode_batch(batched(early))
        # hand it a fresh curve that starts behind where the arm got to
        ahead = pose_matrix()
        ahead[:, 0] += 3.0
        n = s.decode_batch(batched(ahead))[0].shape[1]
        assert n <= 12

    def test_reset_forgets_the_anchor(self):
        s = self.step(align=True)
        s.decode_batch(batched(pose_matrix()))
        assert s._anchor is not None
        s.reset()
        assert s._anchor is None

    def test_a_real_batch_never_aligns(self):
        # Two unrelated samples: an anchor from one says nothing about the other.
        s = self.step(align=True)
        pair = torch.from_numpy(np.stack([pose_matrix(), pose_matrix()]))
        s.decode_batch(pair)
        assert s._anchor is None

    def test_align_off_never_records_an_anchor(self):
        s = self.step(align=False)
        s.decode_batch(batched(pose_matrix()))
        assert s._anchor is None

    def test_the_anchor_is_in_spline_space_not_cart7(self):
        # It must be comparable against the next curve, which lives in the 10-dim
        # spline space; a cart7 anchor would be 7 wide and wrongly parameterised.
        s = self.step(align=True)
        s.decode_batch(batched(pose_matrix()))
        assert s._anchor.shape == (10,)

    def test_align_is_serialized(self):
        assert self.step(align=True).get_config()["align"] is True


# --------------------------------------------------------------------------
# Step 4: config, deploy and eval wiring
# --------------------------------------------------------------------------

from pace_bench.methods.config import BSplineMethod  # noqa: E402


class TestWiring:
    def test_alignment_is_on_by_default(self):
        # Upstream's default too (`disable_time_align=False`). It matters here because
        # B-spline runs with seam blending disabled, so nothing else smooths a seam.
        assert BSplineMethod().align is True

    def test_the_config_field_reaches_the_decode_step(self):
        (step,) = BSplineMethod(layout="cart7", align=True).postprocessor_steps()
        assert step.align is True

    def test_it_can_be_turned_off_for_an_ablation(self):
        (step,) = BSplineMethod(layout="cart7", align=False).postprocessor_steps()
        assert step.align is False

    def test_the_gripper_column_is_excluded_for_cart7(self):
        (step,) = BSplineMethod(layout="cart7").postprocessor_steps()
        assert step.layout.spline_dim == 10
        assert step.compare_dim == 9

    def test_training_never_aligns(self):
        # preprocessor_steps builds the *chunking* step; only decoding aligns, so a
        # training run cannot pick up cross-sample state however align is set.
        steps = BSplineMethod(layout="cart7", align=True).preprocessor_steps()
        assert not any(hasattr(s, "align") for s in steps)

    def test_the_eval_wrapper_clears_the_anchor_on_reset(self):
        import types

        (step,) = BSplineMethod(layout="cart7", align=True).postprocessor_steps()
        step.decode_batch(batched(pose_matrix()))
        assert step._anchor is not None

        class FakePolicy:
            def reset(self):
                pass

            def predict_action_chunk(self, batch):
                return batched(pose_matrix())

        from pace_bench.eval.bspline_policy import attach_bspline

        policy = attach_bspline(FakePolicy(), step)
        policy.reset()
        assert step._anchor is None
        assert isinstance(policy.reset, types.MethodType)


# --------------------------------------------------------------------------
# Step 5: parity with upstream's own search
# --------------------------------------------------------------------------

from upstream_reference_bspline import find_closest_t_to_target  # noqa: E402


class TestUpstreamParity:
    """Same curve, same anchor, same window -> the same resume point.

    Upstream's `dist` is `sqrt(d**2).sum()`, i.e. an L1 sum of absolute differences,
    and its error is the max abs component. Ours must agree on both, or "ported from
    upstream" is a claim nobody checked.
    """

    @pytest.mark.parametrize("frac", [0.0, 0.03, 0.07, 0.12, 0.18])
    def test_the_same_point_is_found(self, frac):
        curve, lo, hi = straight_curve()
        window_hi = lo + ALIGN_WINDOW * (hi - lo)
        anchor = curve(lo + frac * (hi - lo))

        ours_t, ours_err = align_start(curve, anchor, lo, hi, compare_dim=9)
        theirs_t, theirs_err = find_closest_t_to_target(curve, anchor, lo, window_hi, 9)

        assert ours_t == pytest.approx(theirs_t, abs=0.05)
        assert ours_err == pytest.approx(theirs_err, abs=1e-4)

    def test_bracketing_beats_a_blind_local_search_on_a_multi_basin_curve(self):
        """The one deliberate divergence from upstream, and the evidence for it.

        `minimize_scalar(method="bounded")` is Brent: it converges to *a* local
        minimum, and on a path that revisits similar poses within the search window
        there are several. Upstream compensates by widening its bounds and retrying
        until the error passes threshold (`lam *= 1.5`), which is a globalisation
        bolted onto a local method. Scanning first brackets the true minimum outright.

        Scanning alone is not enough either: a grid can step over a sharp minimum that
        Brent would have found. So `align_start` runs both and keeps the lower.

        The comparison below is on the L1 sum, because that is the objective both
        searches minimise. The *reported* error is a max-abs -- a different norm -- so
        a point that wins on the objective can lose on the report by ~1e-7; asserting
        on the report would be testing the gap between two norms, not the search.
        """
        def objective(curve, anchor, t, dim=9):
            return float(np.abs(np.atleast_1d(curve(t)).reshape(-1)[:dim] - np.asarray(anchor).reshape(-1)[:dim]).sum())

        rng = np.random.default_rng(0)
        wins = 0
        worst_gap = 0.0
        for cycles in (3, 5, 8, 12):
            for _ in range(40):
                lo, hi, degree, n = 0.0, 40.0, 3, 64
                knots = np.concatenate([np.full(degree, lo),
                                        np.linspace(lo, hi, n - degree + 1),
                                        np.full(degree, hi)])
                s_ = np.linspace(0, 1, n)
                c = np.zeros((n, 10))
                c[:, 0] = np.sin(2 * np.pi * cycles * s_)
                c[:, 1] = np.cos(2 * np.pi * cycles * s_) * 0.5
                curve = BSpline(knots, c, degree, extrapolate=False)
                anchor = curve(lo + rng.uniform(0, ALIGN_WINDOW) * (hi - lo))

                our_t, _ = align_start(curve, anchor, lo, hi, compare_dim=9)
                their_t, _ = find_closest_t_to_target(
                    curve, anchor, lo, lo + ALIGN_WINDOW * (hi - lo), 9
                )
                ours = objective(curve, anchor, our_t)
                theirs = objective(curve, anchor, their_t)
                assert ours <= theirs + 1e-9, "the search must never do worse than upstream"
                if ours < theirs - 1e-9:
                    wins += 1
                    worst_gap = max(worst_gap, theirs - ours)

        assert wins > 40, f"expected the scan to matter often, won only {wins}/160"
        assert worst_gap > ALIGN_ERROR_THRESHOLD, (
            "if the gap never exceeds the accept threshold the scan buys nothing"
        )


# --------------------------------------------------------------------------
# n_action_steps: a B-spline chunk is one curve, not a sequence
# --------------------------------------------------------------------------

class TestActionStepsGuard:
    """crisp_gym enforces `n_act < chunk_size` because an ordinary chunk is a sequence
    you execute a prefix of. A parameter matrix is not: a prefix of its rows is a
    different, shorter curve. The only correct value is the full width, which that
    guard rejects -- so the config must leave it unset, and setting it must fail here
    rather than at robot bring-up.
    """

    def method(self):
        return BSplineMethod(layout="cart7", chunk_size=10, degree=3)

    def test_unset_is_accepted(self):
        from pace_bench.real.deploy_flags import validate_action_steps

        validate_action_steps(self.method(), None)

    def test_the_full_width_is_accepted(self):
        from pace_bench.real.deploy_flags import validate_action_steps

        m = self.method()
        assert m.width == 16
        validate_action_steps(m, m.width)

    @pytest.mark.parametrize("bad", [15, 8, 32])
    def test_any_other_value_is_refused(self, bad):
        from pace_bench.real.deploy_flags import validate_action_steps

        with pytest.raises(ValueError, match=r"parameter matrix"):
            validate_action_steps(self.method(), bad)

    def test_other_methods_are_untouched(self):
        from types import SimpleNamespace

        from pace_bench.real.deploy_flags import validate_action_steps

        validate_action_steps(SimpleNamespace(type="pace"), 32)
        validate_action_steps(SimpleNamespace(type="none"), 32)

    def test_the_shipped_config_leaves_it_unset(self):
        # deploy_defaults.yaml sets 32 and every config inherits it, so bspline_1x must
        # override it to null -- 32 would be refused, and 16 would trip crisp_gym.
        from pace_bench.real.configs import resolve_config

        assert resolve_config("real/configs/bspline_1x.yaml")["n_action_steps"] is None
        assert resolve_config("real/configs/bspline_fast.yaml")["n_action_steps"] is None


# --------------------------------------------------------------------------
# Rebuilding a step from a checkpoint's serialised config
# --------------------------------------------------------------------------

class TestConfigRoundTrip:
    """`get_config` writes layout and arrangement by NAME, so reconstruction is
    handed strings. Storing them verbatim surfaced as `'str' object has no attribute
    'recover'` inside the inference subprocess, after the robot was up -- the step is
    rebuilt by lerobot from policy_postprocessor.json, never by our own code.
    """

    SAVED = {
        "num_actions": 16, "degree": 3, "relative_knots": False,
        "layout": "cart7", "arrangement": "knot_first",
    }

    def test_names_are_resolved_to_objects(self):
        from pace_bench.methods.bspline.processor import BSplineDecodeStep

        step = BSplineDecodeStep(**self.SAVED)
        assert hasattr(step.arrangement, "recover")
        assert hasattr(step.layout, "from_spline")
        assert step.layout.name == "cart7"

    def test_a_rebuilt_step_decodes(self):
        import torch

        from pace_bench.methods.bspline.processor import BSplineDecodeStep

        step = BSplineDecodeStep(**self.SAVED)
        m = torch.zeros(1, 16, 11, dtype=torch.float64)
        m[0, :, 0] = torch.arange(16, dtype=torch.float64)      # monotonic knots
        # cart7 control points are xyz(3) + rot6d(6) + gripper(1); the rot6d block
        # must be a real rotation or the re-orthonormalisation is degenerate.
        m[0, :, 4] = 1.0
        m[0, :, 8] = 1.0
        actions, rates = step.decode_batch(m)
        assert actions.shape == (1, 16, 7)

    def test_objects_are_passed_through_unchanged(self):
        from pace_bench.methods.bspline.layout import LAYOUTS, resolve_arrangement
        from pace_bench.methods.bspline.processor import BSplineDecodeStep

        layout, arr = LAYOUTS["cart7"], resolve_arrangement("knot_first")
        step = BSplineDecodeStep(layout=layout, arrangement=arr)
        assert step.layout is layout
        assert step.arrangement is arr

    def test_the_chunk_step_coerces_too(self):
        from pace_bench.methods.bspline.processor import BSplineChunkStep

        step = BSplineChunkStep(arrangement="knot_first")
        assert hasattr(step.arrangement, "recover")

    def test_identity_is_refused_rather_than_silently_wrong(self):
        # `identity` adopts the dataset's width, which a name cannot carry.
        from pace_bench.methods.bspline.processor import BSplineDecodeStep

        with pytest.raises(ValueError, match=r"adopts the dataset"):
            BSplineDecodeStep(layout="identity")

    def test_an_unknown_name_names_the_known_ones(self):
        from pace_bench.methods.bspline.processor import BSplineDecodeStep

        with pytest.raises(ValueError, match=r"known:"):
            BSplineDecodeStep(layout="nope")
