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

from pace_bench.methods.bspline.pos_emb import FILENAME, find, restore, unfreeze
from pace_bench.methods.config import BSplineMethod

ROWS, DIM = 32, 4


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
