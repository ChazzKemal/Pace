"""KDE entropy: upstream's arithmetic, and the properties it has to have."""

import pytest
import torch

from robot_stack.methods.demospeedup.entropy import gaussian_kernel, kde_entropy


def upstream_kde_entropy(x: torch.Tensor, bandwidth: float = 1.0) -> torch.Tensor:
    """Transcribed from lingxiao-guo/DemoSpeedup @ 34bd43a.

    ``robobase/robobase/utils.py``: ``gaussian_kernel`` inlined into
    ``KDE.kde_entropy``, with the dead bandwidth-estimation branch dropped (upstream
    overwrites its result with ``bandwidth = 1`` on the following line) and the
    unused max-density return value omitted.
    """
    _batch, num_samples, _dim = x.size()
    x_i = x.unsqueeze(2)
    x_j = x.unsqueeze(1)
    distances = torch.sum((x_i - x_j) ** 2, dim=-1)
    kernel_values = torch.exp(-distances / (2 * bandwidth**2))
    density = kernel_values.sum(dim=2) / num_samples
    return -torch.log(density + 1e-8).mean(dim=1)


@pytest.mark.parametrize("num_samples", [2, 5, 10, 32])
@pytest.mark.parametrize("dim", [1, 7, 14])
@pytest.mark.parametrize("bandwidth", [0.5, 1.0, 2.0])
def test_matches_upstream(num_samples, dim, bandwidth):
    torch.manual_seed(num_samples * 100 + dim)
    x = torch.randn(3, num_samples, dim)
    assert torch.allclose(kde_entropy(x, bandwidth), upstream_kde_entropy(x, bandwidth), atol=1e-6)


def test_identical_samples_have_zero_entropy():
    """No disagreement means no uncertainty -- the floor of the measurement."""
    x = torch.ones(1, 8, 7)
    assert kde_entropy(x).item() == pytest.approx(0.0, abs=1e-6)


def test_entropy_rises_with_spread():
    """The whole premise: more disagreement among samples, more entropy."""
    torch.manual_seed(0)
    base = torch.randn(1, 12, 7)
    entropies = [kde_entropy(base * scale).item() for scale in (0.01, 0.1, 1.0, 10.0)]
    assert entropies == sorted(entropies), entropies


def test_entropy_saturates_at_log_num_samples():
    """Fully separated samples each see only themselves: density 1/n, entropy log n."""
    x = (torch.arange(16, dtype=torch.float32) * 1000).reshape(1, 16, 1)
    assert kde_entropy(x).item() == pytest.approx(torch.log(torch.tensor(16.0)).item(), abs=1e-4)


def test_batch_elements_are_independent():
    torch.manual_seed(1)
    a, b = torch.randn(1, 10, 7), torch.randn(1, 10, 7)
    together = kde_entropy(torch.cat([a, b], dim=0))
    assert together[0].item() == pytest.approx(kde_entropy(a).item(), abs=1e-6)
    assert together[1].item() == pytest.approx(kde_entropy(b).item(), abs=1e-6)


def test_kernel_is_one_on_the_diagonal():
    k = gaussian_kernel(torch.randn(2, 6, 3))
    assert torch.allclose(torch.diagonal(k, dim1=1, dim2=2), torch.ones(2, 6))


def test_rejects_shapes_entropy_is_undefined_on():
    with pytest.raises(ValueError, match="batch, num_samples, dim"):
        gaussian_kernel(torch.randn(10, 7))
    with pytest.raises(ValueError, match="at least 2 samples"):
        kde_entropy(torch.randn(1, 1, 7))
