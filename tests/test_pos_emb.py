"""Training the action slice of xVLA's positional embedding, and keeping it.

Two failures these cover, both of which are silent in the worst way -- the run reports
success and the checkpoint is wrong:

* PEFT wraps the policy *after* `make_policy` returns and freezes everything it did not
  target. An unfreeze applied to `make_policy`'s return value is undone and its object
  discarded, while the log still says the embedding is training.
* A PEFT checkpoint stores adapter tensors and `modules_to_save` weights only. A
  `pos_emb` trained but not written is lost at save time, and evaluation quietly loads
  the original frozen table.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from pace_bench.methods.bspline.pos_emb import FILENAME, find, realign, restore, unfreeze
from pace_bench.methods.config import BSplineMethod

ROWS, DIM = 32, 4


def _numbered(param: torch.nn.Parameter) -> torch.Tensor:
    """Fill the table so row r reads r everywhere: any move is then visible by value."""
    with torch.no_grad():
        param.copy_(torch.arange(ROWS, dtype=torch.float32).view(1, ROWS, 1).expand(1, ROWS, DIM))
    return param.detach().clone()


class FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Non-zero and deterministic: d(x^2)/dx is 2x, so a zero-filled table would
        # give a zero gradient everywhere and the row mask would look like it worked.
        self.pos_emb = nn.Parameter(torch.full((1, ROWS, DIM), 0.1))


class FakePolicy(nn.Module):
    """Stands in for XVLAPolicy, including the part that broke.

    `wrap_with_peft` returns a *different object* with everything frozen, which is what
    PEFT does and what makes hook ordering matter.
    """

    def __init__(self):
        super().__init__()
        self.transformer = FakeTransformer()
        # An `nn.Parameter` is trainable by construction, and so is the real pos_emb
        # (`soft_transformer.py:334` passes requires_grad=True). What freezes it in a
        # real run is PEFT, freezing everything it did not target -- so the fixture has
        # to start frozen or it would not be testing anything.
        self.transformer.pos_emb.requires_grad_(False)

    def save_pretrained(self, directory, *args, **kwargs):
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "adapter_model.safetensors").write_bytes(b"")

    def wrap_with_peft(self, **kwargs):
        wrapped = FakePolicy()
        wrapped.load_state_dict(self.state_dict())
        for param in wrapped.parameters():
            param.requires_grad_(False)
        return wrapped


class TestUnfreeze:
    def test_only_the_named_rows_get_gradient(self):
        policy = FakePolicy()
        _, param = find(policy)
        assert param.requires_grad is False, "the fixture should start frozen"

        unfreeze(policy, 8)
        assert param.requires_grad is True
        (param**2).sum().backward()
        assert (param.grad[:, :8] != 0).any(), "the action rows must train"
        assert not (param.grad[:, 8:] != 0).any(), "the visual rows must not"

    def test_a_row_count_past_the_table_is_refused(self):
        with pytest.raises(ValueError, match="pos_emb holds"):
            unfreeze(FakePolicy(), ROWS + 1)

    def test_a_policy_without_one_is_refused(self):
        with pytest.raises(ValueError, match="no `pos_emb`"):
            unfreeze(nn.Linear(2, 2), 4)


class TestRealign:
    """The table is indexed by absolute position and the action segment comes first, so
    a shorter chunk slides every non-action token down the table. Realignment keeps each
    of those tokens on the row it was pretrained with."""

    def test_non_action_rows_keep_their_pretrained_codes(self):
        policy = FakePolicy()
        _, param = find(policy)
        before = _numbered(param)
        realign(policy, source_rows=12, rows=8)
        after = param.detach()
        torch.testing.assert_close(after[:, :8], before[:, :8], msg="the action rows stay")
        # A token at offset j past the action segment read row 12 + j before and reads
        # row 8 + j now; that row must hold what 12 + j held.
        torch.testing.assert_close(after[:, 8 : 8 + (ROWS - 12)], before[:, 12:])
        assert sorted(after[0, :, 0].tolist()) == sorted(before[0, :, 0].tolist()), "a permutation"

    def test_a_wider_action_segment_moves_them_the_other_way(self):
        policy = FakePolicy()
        _, param = find(policy)
        before = _numbered(param)
        realign(policy, source_rows=8, rows=12)
        after = param.detach()
        torch.testing.assert_close(after[:, :8], before[:, :8])
        torch.testing.assert_close(after[:, 12:], before[:, 8 : 8 + (ROWS - 12)])
        assert sorted(after[0, :, 0].tolist()) == sorted(before[0, :, 0].tolist())

    def test_an_unchanged_segment_moves_nothing(self):
        policy = FakePolicy()
        _, param = find(policy)
        before = _numbered(param)
        realign(policy, source_rows=8, rows=8)
        torch.testing.assert_close(param.detach(), before)

    def test_it_is_persisted_like_a_trained_table(self, tmp_path):
        """Evaluation rebuilds the policy from the base checkpoint, shift and all, so a
        realigned table that is not written beside the adapter is a realignment that
        never happened at eval."""
        policy = FakePolicy()
        _, param = find(policy)
        _numbered(param)
        realign(policy, source_rows=12, rows=8)
        policy.save_pretrained(tmp_path)
        assert (tmp_path / FILENAME).is_file()
        fresh = FakePolicy()
        assert restore(fresh, tmp_path) is True
        torch.testing.assert_close(find(fresh)[1], param)

    def test_realigning_then_unfreezing_writes_the_file_once(self, tmp_path, monkeypatch):
        from pace_bench.methods.bspline import pos_emb

        writes = []
        monkeypatch.setattr(pos_emb, "save_file", lambda tensors, path: writes.append(path))
        policy = FakePolicy()
        realign(policy, source_rows=12, rows=8)
        unfreeze(policy, 8)
        policy.save_pretrained(tmp_path)
        assert len(writes) == 1

    def test_a_segment_past_the_table_is_refused(self):
        with pytest.raises(ValueError, match="cannot realign"):
            realign(FakePolicy(), source_rows=ROWS + 1, rows=8)


class TestRealignIsWired:
    class XVLACfg:
        """What `adjust_policy` sees on a run started from the hub checkpoint."""

        type = "xvla"
        chunk_size = 30
        n_action_steps = 30

    def test_adjust_policy_records_the_pretrained_chunk_once(self):
        method = BSplineMethod(realign_pos_emb=True)
        cfg = self.XVLACfg()
        method.adjust_policy(cfg)
        assert (method.source_chunk, cfg.chunk_size) == (30, 16)
        # A resume parses the checkpoint's config, whose chunk is already 16.
        method.adjust_policy(cfg)
        assert method.source_chunk == 30, "the recorded source must survive a re-apply"

    def test_the_built_policy_is_realigned_from_the_recorded_chunk(self):
        method = BSplineMethod(realign_pos_emb=True)
        method.adjust_policy(self.XVLACfg())
        policy = FakePolicy()
        policy.config = type("C", (), {"pretrained_path": "lerobot/xvla-libero"})()
        _, param = find(policy)
        before = _numbered(param)
        method.adjust_built_policy(policy)
        torch.testing.assert_close(param.detach()[:, 16 : 16 + (ROWS - 30)], before[:, 30:])
        assert param.requires_grad is False, "realignment alone trains nothing"

    def test_a_resumed_run_takes_the_saved_table_over_a_fresh_realignment(self, tmp_path):
        method = BSplineMethod(realign_pos_emb=True, unfreeze_pos_emb_rows=16)
        method.adjust_policy(self.XVLACfg())
        trained = FakePolicy()
        trained.config = type("C", (), {"pretrained_path": "lerobot/xvla-libero"})()
        method.adjust_built_policy(trained)
        _, param = find(trained)
        with torch.no_grad():
            param[:, :16, :] += 0.5
        trained.save_pretrained(tmp_path)

        resumed = FakePolicy()
        resumed.config = type("C", (), {"pretrained_path": str(tmp_path)})()
        method.adjust_built_policy(resumed)
        torch.testing.assert_close(find(resumed)[1], param)

    def test_realigning_without_the_recorded_chunk_is_refused(self):
        with pytest.raises(ValueError, match="source_chunk"):
            BSplineMethod(realign_pos_emb=True).adjust_built_policy(FakePolicy())

    def test_the_flag_off_leaves_the_table_alone(self):
        method = BSplineMethod()
        method.adjust_policy(self.XVLACfg())
        policy = FakePolicy()
        _, param = find(policy)
        before = _numbered(param)
        method.adjust_built_policy(policy)
        torch.testing.assert_close(param.detach(), before)


class TestPersistence:
    def test_the_trained_table_is_written_and_read_back(self, tmp_path):
        policy = FakePolicy()
        param = unfreeze(policy, 8)
        with torch.no_grad():
            param[:, :8, :] += 0.25

        policy.save_pretrained(tmp_path)
        assert (tmp_path / FILENAME).is_file(), "the adapter cannot carry it; it needs its own file"

        fresh = FakePolicy()
        _, blank = find(fresh)
        assert not torch.equal(blank, param), "without a restore the training would be lost"
        assert restore(fresh, tmp_path) is True
        _, loaded = find(fresh)
        torch.testing.assert_close(loaded, param)

    def test_a_checkpoint_without_one_is_left_alone(self, tmp_path):
        """Every arm trained so far has a frozen embedding, so this is the common case
        and must not raise or overwrite."""
        fresh = FakePolicy()
        _, before = find(fresh)
        assert restore(fresh, tmp_path) is False
        torch.testing.assert_close(find(fresh)[1], before)


class TestHookOrdering:
    def test_the_adjustment_survives_peft_wrapping(self, monkeypatch):
        """The regression. Upstream does `policy = policy.wrap_with_peft(...)` *after*
        make_policy returns, and PEFT freezes what it did not target -- so an unfreeze
        applied to make_policy's return value is both undone and thrown away."""
        from lerobot.scripts import lerobot_train

        from pace_bench.train.run_train import attach_method_steps

        # Registered so pytest restores the module-level patches attach_method_steps makes.
        for name in ("make_policy", "make_train_eval_datasets", "make_pre_post_processors"):
            monkeypatch.setattr(lerobot_train, name, getattr(lerobot_train, name), raising=False)

        built = FakePolicy()
        monkeypatch.setattr(lerobot_train, "make_policy", lambda *a, **k: built, raising=False)
        attach_method_steps(BSplineMethod(unfreeze_pos_emb_rows=8))

        policy = lerobot_train.make_policy()
        wrapped = policy.wrap_with_peft()
        _, param = find(wrapped)
        assert param.requires_grad, "PEFT froze it back -- the hook ran before the wrap"

    def test_an_unwrapped_policy_is_still_adjusted(self, monkeypatch):
        """A run without PEFT never calls wrap_with_peft, so the hook has to fire on the
        policy itself or it never fires at all."""
        from lerobot.scripts import lerobot_train

        from pace_bench.train.run_train import attach_method_steps

        for name in ("make_policy", "make_train_eval_datasets", "make_pre_post_processors"):
            monkeypatch.setattr(lerobot_train, name, getattr(lerobot_train, name), raising=False)

        plain = FakePolicy()
        plain.wrap_with_peft = None  # not a PEFT run; upstream never calls it
        monkeypatch.setattr(lerobot_train, "make_policy", lambda *a, **k: plain, raising=False)
        attach_method_steps(BSplineMethod(unfreeze_pos_emb_rows=8))

        _, param = find(lerobot_train.make_policy())
        assert param.requires_grad

    def test_the_flag_off_changes_nothing(self, monkeypatch):
        from lerobot.scripts import lerobot_train

        from pace_bench.train.run_train import attach_method_steps

        for name in ("make_policy", "make_train_eval_datasets", "make_pre_post_processors"):
            monkeypatch.setattr(lerobot_train, name, getattr(lerobot_train, name), raising=False)

        built = FakePolicy()
        monkeypatch.setattr(lerobot_train, "make_policy", lambda *a, **k: built, raising=False)
        attach_method_steps(BSplineMethod())

        _, param = find(lerobot_train.make_policy().wrap_with_peft())
        assert param.requires_grad is False


class TestRestoreIsWired:
    """Saving without loading is worse than not saving at all: the run looks right and
    the weights are gone. These pin that the load half is actually reachable."""

    def test_a_resumed_run_reloads_the_trained_table(self, tmp_path, monkeypatch):
        from pace_bench.methods.config import BSplineMethod

        trained = FakePolicy()
        param = unfreeze(trained, 8)
        with torch.no_grad():
            param[:, :8, :] += 0.5
        trained.save_pretrained(tmp_path)

        # A resume rebuilds from the checkpoint: fresh weights, `pretrained_path` set.
        resumed = FakePolicy()
        resumed.config = type("C", (), {"pretrained_path": str(tmp_path)})()
        BSplineMethod(unfreeze_pos_emb_rows=8).adjust_built_policy(resumed)

        _, restored = find(resumed)
        torch.testing.assert_close(restored, param)
        assert restored.requires_grad, "and it must still be trainable after the reload"

    def test_a_first_run_is_unaffected(self, tmp_path):
        """`pretrained_path` is a hub id on a fresh finetune, with no file beside it."""
        from pace_bench.methods.config import BSplineMethod

        fresh = FakePolicy()
        fresh.config = type("C", (), {"pretrained_path": "lerobot/xvla-libero"})()
        _, before = find(fresh)
        BSplineMethod(unfreeze_pos_emb_rows=8).adjust_built_policy(fresh)
        _, after = find(fresh)
        torch.testing.assert_close(after, before)
