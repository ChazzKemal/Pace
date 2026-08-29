"""Regenerate `segment_golden.npz` by running upstream DemoSpeedup's own function.

    git clone https://github.com/lingxiao-guo/DemoSpeedup /tmp/DemoSpeedup
    uv pip install hdbscan          # upstream's clustering backend
    DEMOSPEEDUP_UPSTREAM=/tmp/DemoSpeedup uv run python tests/assets/gen_segment_golden.py

Upstream's module imports mujoco and hydra at import time, so the two functions
under test are exec'd out of the file's source text rather than imported. The one
thing injected is a seeded IsolationForest: upstream leaves `random_state` unset, so
without this its labels are not reproducible and no golden file could exist.

The clustering backend here is upstream's own `hdbscan` package, NOT the
scikit-learn port the runtime uses -- so the golden pins the algorithm end to end,
backend included, and `test_demospeedup_segment.py` proves the port matches it.
"""

import ast
import functools
import os
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest

UPSTREAM = Path(os.environ["DEMOSPEEDUP_UPSTREAM"]) / "robobase" / "robobase" / "utils.py"
WANTED = {"hdbscan_with_custom_merge", "remove_outliers_isolation_forest"}


def load_upstream():
    import hdbscan

    tree = ast.parse(UPSTREAM.read_text())
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in WANTED]
    assert len(fns) == len(WANTED), [f.name for f in fns]
    ns = {
        "np": np,
        "hdbscan": hdbscan,
        "IsolationForest": functools.partial(IsolationForest, random_state=0),
    }
    exec(compile(ast.Module(fns, []), "<upstream>", "exec"), ns)  # noqa: S102
    return ns["hdbscan_with_custom_merge"]


def traces() -> dict[str, np.ndarray]:
    """Entropy traces spanning the shapes a real labelling run produces."""
    rng = np.random.default_rng(0)
    out = {}

    # Mostly precision, with stretches of confident transit. The shape a good
    # proxy policy gives on a manipulation demo.
    a = rng.normal(1.0, 0.25, 300)
    for s in (30, 120, 210):
        a[s : s + 45] = rng.normal(3.2, 0.4, 45)
    out["precision_majority"] = a

    # The inverse: a policy unsure almost everywhere, certain in three places.
    b = rng.normal(3.0, 0.4, 300)
    for s in (40, 140, 240):
        b[s : s + 30] = rng.normal(1.0, 0.25, 30)
    out["transit_majority"] = b

    # A long tail, which is what drives HDBSCAN to call points noise.
    out["heavy_tail"] = np.abs(rng.standard_cauchy(250)) + 0.5

    # No structure at all -- the degenerate case the clustering has to survive.
    out["unstructured"] = rng.normal(2.0, 1.0, 200)

    # A single ramp: every frame in its own neighbourhood, no plateau to cluster.
    out["monotonic_ramp"] = np.linspace(0.5, 4.0, 180)

    # Short episode, near the min_cluster_size floor.
    out["short_episode"] = rng.normal(1.5, 0.6, 40)

    return out


def main() -> None:
    upstream = load_upstream()
    data = {}
    for name, trace in traces().items():
        labels = np.abs(upstream(trace, dir=None, rollout_id=0, plot=False)).astype(np.int64)
        data[f"{name}__trace"] = trace
        data[f"{name}__labels"] = labels
        print(f"{name:20s} n={len(trace):4d}  non-precision={100 * labels.mean():5.1f}%")
    out = Path(__file__).with_name("segment_golden.npz")
    np.savez_compressed(out, **data)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
