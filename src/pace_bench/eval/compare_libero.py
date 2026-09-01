"""Compare LIBERO eval runs against each other, episode by episode.

LIBERO has no validation *set* of demonstrations -- the benchmark is online rollouts
in the simulator, and each task ships 50 fixed initial states
(``libero/init_files/<suite>/*.pruned_init``). Those initial states are the held-out
axis, and they are held out in fact and not only by convention: checked against the
raw demonstrations, none of the 500 demo start states coincides with an eval init
state, and a demo sits as far from its nearest init state as init states sit from
each other. The policy has seen the demos; it has not seen these placements.

What makes a comparison here stronger than "92.0% vs 85.5%" is that two runs of this
repo's evaluator are **paired**. ``lerobot.envs.libero`` ties sub-env *i* to init
state *i* and advances by ``n_envs`` on every reset, while upstream's rollout numbers
episodes batch-major/env-minor -- so episode *k* of any run is init state *k* of that
task -- whatever batch size it ran at. Two runs sharing (suite, tasks, n_episodes,
seed) therefore attempted the same scenes in the same order, and the
right test is McNemar on the disagreements rather than a two-sample test on the
rates. This module refuses to pair runs whose scene sets do not match, and falls
back to unpaired reporting with a warning.

    python -m pace_bench.eval.compare_libero outputs/eval/ds_libero10_base \
        outputs/eval/ds_libero10_speedup

    python -m pace_bench.eval.compare_libero --labels=baseline,demospeedup \
        --json=outputs/eval/comparison.json  outputs/eval/ds_*

The first run listed is the reference every other run is compared against; ``--baseline``
picks a different one.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

# The fields that must agree before two runs may be paired. `policy_path` and the
# whole `method` block are deliberately absent: differing in those is the point.
#
# `batch_size` is absent too, and that is a claim worth stating. Episode k of a run is
# batch k//n, env k%n. Upstream seeds it `start_seed + batch_ix*n + i` = start_seed + k
# (`lerobot_eval.py:532`), and LIBERO starts sub-env i at init state i and advances it
# by n per reset (`envs/libero.py:175`), giving init state i + (k//n)*n = k. Both land
# on k whatever n is -- so two runs with different batch sizes still attempted the same
# scene, under the same seed, at the same episode index, and pair exactly. Which
# matters in practice: an arm sharing the card with a training job has to run at a
# smaller batch than one that has the card to itself.
PAIRING_KEYS = ("task_suite", "n_episodes", "seed")

# Initial states LIBERO ships per task, in `libero/init_files/<suite>/*.pruned_init`.
# The evaluator walks them in order, so an `n_episodes` below this covers a prefix.
LIBERO_INIT_STATES = 50

# --- statistics -- each one chosen for what the pairing supports ---------------

def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a success count, in percent.

    Not the textbook normal approximation: at n=20 per task and rates near 100% that
    one produces intervals reaching above 1.0, which would misreport exactly the
    regime these runs sit in.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (100 * max(0.0, centre - half), 100 * min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts.

    ``b`` = reference succeeded and the other failed, ``c`` = the reverse. Episodes
    where both agree carry no information about the difference and are excluded --
    that is the whole point of pairing. Exact rather than chi-squared because the
    discordant counts here are small (tens of episodes, not thousands).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test -- McNemar's arithmetic, applied to durations.

    Which of two paired times is smaller is a Bernoulli question, and asking it this
    way needs no assumption about how episode durations are distributed. Ties are
    dropped, as the sign test requires.
    """
    return mcnemar_exact(wins, losses)


def paired_bootstrap_ci(
    pairs: list[tuple[float, float]], stat, *, n_resamples: int = 10000, seed: int = 0
) -> tuple[float, float]:
    """Percentile CI for a paired statistic, resampling *episodes*, not runs.

    Resampling the pair keeps each scene's two outcomes together, which is what makes
    the interval narrower than two independent intervals would be -- the same reason
    McNemar beats a two-sample test here. Seeded, so a report is reproducible.
    """
    if not pairs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(pairs)
    draws = []
    for _ in range(n_resamples):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        value = stat(sample)
        if value is not None and not math.isnan(value):
            draws.append(value)
    if not draws:
        return (float("nan"), float("nan"))
    draws.sort()
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(math.ceil(0.975 * (len(draws) - 1)))]
    return (lo, hi)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the order given.

    Every extra arm compared against one baseline is another chance to find a
    difference that is not there. Holm controls that without Bonferroni's power loss,
    and reduces to no correction at all when there is a single comparison.
    """
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted


# --- a run on disk -------------------------------------------------------------

@dataclass
class TaskResult:
    """One task's episodes from one run, in upstream's episode order."""

    task_id: int
    successes: list[bool]
    # Per-episode simulated seconds, keyed by episode index. Sparse: only episodes
    # that terminated are timed, which in practice means only successes.
    sim_times: dict[int, float] = field(default_factory=dict)
    # Durations from a run recorded before episode indices existed. They still average
    # to a valid ATR; they just cannot be matched to a scene, so no paired timing.
    unindexed_times: list[float] = field(default_factory=list)

    @property
    def n_episodes(self) -> int:
        return len(self.successes)


@dataclass
class Run:
    """One output directory: its config and one :class:`TaskResult` per task."""

    label: str
    path: Path
    config: dict
    tasks: dict[int, TaskResult]

    @property
    def episodes(self) -> list[bool]:
        return [s for t in sorted(self.tasks) for s in self.tasks[t].successes]

    @property
    def success_rate(self) -> float:
        eps = self.episodes
        return 100.0 * sum(eps) / len(eps) if eps else float("nan")

    @property
    def success_times(self) -> list[float]:
        """Simulated seconds for every successful episode this run timed."""
        out: list[float] = []
        for tid in sorted(self.tasks):
            task = self.tasks[tid]
            out.extend(task.sim_times[i] for i, ok in enumerate(task.successes) if ok and i in task.sim_times)
            out.extend(task.unindexed_times)
        return out

    @property
    def avg_success_time(self) -> float | None:
        times = self.success_times
        return sum(times) / len(times) if times else None

    def pairing_signature(self) -> tuple:
        return tuple(self.config.get(k) for k in PAIRING_KEYS) + (tuple(sorted(self.tasks)),)


def load_run(path: Path, label: str | None = None) -> Run:
    """Read one eval output directory: ``task_<id>/eval_info.json`` per task.

    Timing is taken from this repo's `episodes` block rather than from the
    `avg_success_sim_s` summary, so a run recorded before episode indices existed
    still contributes success rates -- it just cannot be paired on time.
    """
    task_dirs = sorted(path.glob("task_*"), key=lambda p: int(p.name.split("_")[1]))
    if not task_dirs:
        raise FileNotFoundError(f"{path} holds no task_* directories")

    tasks: dict[int, TaskResult] = {}
    config: dict = {}
    for task_dir in task_dirs:
        info_path = task_dir / "eval_info.json"
        if not info_path.exists():
            continue  # a task still running, or killed before it wrote
        info = json.loads(info_path.read_text())
        task_id = int(task_dir.name.split("_")[1])
        # per_task is a one-element list here: the runner evaluates one task per dir.
        successes = [bool(s) for pt in info.get("per_task", []) for s in pt["metrics"]["successes"]]
        timed = [ep for ep in info.get("episodes", []) if ep.get("sim_time") is not None and ep["success"]]
        sim_times = {ep["episode_index"]: ep["sim_time"] for ep in timed if "episode_index" in ep}
        legacy = [ep["sim_time"] for ep in timed if "episode_index" not in ep]
        tasks[task_id] = TaskResult(
            task_id=task_id, successes=successes, sim_times=sim_times, unindexed_times=legacy
        )
        config = config or info.get("config", {})

    if not tasks:
        raise FileNotFoundError(f"{path} holds no completed task_*/eval_info.json")
    return Run(label=label or path.name, path=path, config=config, tasks=tasks)


# --- one run against another ---------------------------------------------------

@dataclass
class Comparison:
    """One run measured against the reference run."""

    label: str
    reference: str
    paired: bool
    n_paired: int
    both: int
    only_reference: int  # reference succeeded, this run failed
    only_other: int
    delta_sr: float
    p_value: float
    delta_sr_ci: tuple[float, float] | None
    # Timing, over episodes both runs completed successfully -- the only ones where
    # both clocks read a comparable quantity.
    n_time_paired: int
    ref_time: float | None
    other_time: float | None
    speedup: float | None
    speedup_ci: tuple[float, float] | None = None
    # Per-episode verdict on duration: how often each run was the faster of the two.
    faster: int = 0
    slower: int = 0
    time_p: float = float("nan")
    median_ratio: float | None = None
    # Holm-adjusted McNemar p, filled in once every comparison in a report is known.
    p_adjusted: float = float("nan")


def compare(reference: Run, other: Run) -> Comparison:
    paired = reference.pairing_signature() == other.pairing_signature()
    shared = sorted(set(reference.tasks) & set(other.tasks))

    both = only_ref = only_other = neither = 0
    ref_times: list[float] = []
    other_times: list[float] = []
    for task_id in shared if paired else []:
        a, b = reference.tasks[task_id], other.tasks[task_id]
        for i, (ok_a, ok_b) in enumerate(zip(a.successes, b.successes, strict=False)):
            if ok_a and ok_b:
                both += 1
                if i in a.sim_times and i in b.sim_times:
                    ref_times.append(a.sim_times[i])
                    other_times.append(b.sim_times[i])
            elif ok_a:
                only_ref += 1
            elif ok_b:
                only_other += 1
            else:
                neither += 1

    n_paired = both + only_ref + only_other + neither
    ref_mean = sum(ref_times) / len(ref_times) if ref_times else None
    other_mean = sum(other_times) / len(other_times) if other_times else None
    if paired and ref_mean and other_mean:
        speedup = ref_mean / other_mean
    else:
        # Unpaired fallback: the two runs' own averages over their own successes.
        r, o = reference.avg_success_time, other.avg_success_time
        speedup = r / o if r and o else None

    # Interval on the difference the pairing actually supports. `success_pairs` is one
    # entry per attempted episode; `time_pairs` one per episode both runs solved.
    success_pairs = (
        [(1.0, 1.0)] * both + [(1.0, 0.0)] * only_ref + [(0.0, 1.0)] * only_other + [(0.0, 0.0)] * neither
    )
    delta_sr_ci = (
        paired_bootstrap_ci(
            success_pairs,
            lambda s: 100 * (sum(b for _, b in s) - sum(a for a, _ in s)) / len(s),
        )
        if paired and success_pairs
        else None
    )
    time_pairs = list(zip(ref_times, other_times, strict=True))
    speedup_ci = (
        paired_bootstrap_ci(time_pairs, lambda s: sum(a for a, _ in s) / sum(b for _, b in s))
        if time_pairs
        else None
    )
    faster = sum(1 for a, b in time_pairs if b < a)
    slower = sum(1 for a, b in time_pairs if b > a)
    ratios = sorted(a / b for a, b in time_pairs if b > 0)

    return Comparison(
        label=other.label,
        reference=reference.label,
        paired=paired,
        n_paired=n_paired,
        both=both,
        only_reference=only_ref,
        only_other=only_other,
        delta_sr=other.success_rate - reference.success_rate,
        p_value=mcnemar_exact(only_ref, only_other) if paired else float("nan"),
        delta_sr_ci=delta_sr_ci,
        n_time_paired=len(ref_times),
        # The paired means where there are any, and each run's own average otherwise --
        # `n_time_paired` is what says which of the two a reader is looking at.
        ref_time=ref_mean if ref_mean is not None else reference.avg_success_time,
        other_time=other_mean if other_mean is not None else other.avg_success_time,
        speedup=speedup,
        speedup_ci=speedup_ci,
        faster=faster,
        slower=slower,
        time_p=sign_test(faster, slower) if time_pairs else float("nan"),
        median_ratio=ratios[len(ratios) // 2] if ratios else None,
    )


# --- the report ----------------------------------------------------------------

def _fmt(value: float | None, spec: str = ".2f", dash: str = "n/a") -> str:
    return dash if value is None or (isinstance(value, float) and math.isnan(value)) else format(value, spec)


def _verdict(c: Comparison) -> str:
    """One sentence per arm, and no verdict the numbers do not carry.

    A comparison that is not paired gets no claim about significance at all: its
    episodes are different scenes, so the difference in rates confounds method and
    scene draw.
    """
    if not c.paired:
        return "not comparable episode by episode; treat the rates as separate measurements"
    sig = c.p_adjusted if not math.isnan(c.p_adjusted) else c.p_value
    if sig < 0.05:
        success = f"{'higher' if c.delta_sr > 0 else 'lower'} success rate (p={sig:.3f})"
    else:
        n_disagree = c.only_reference + c.only_other
        success = (
            f"no detectable success difference ({n_disagree} disagreements out of {c.n_paired}, "
            f"p={sig:.3f})"
        )
    if c.speedup is None:
        return success + "; no timing recorded"
    if c.n_time_paired:
        fast = f"{c.speedup:.2f}x faster" if c.speedup > 1 else f"{1 / c.speedup:.2f}x slower"
        strength = "consistently" if c.time_p < 0.05 else "on average but not consistently"
        return f"{success}; {strength} {fast} on the episodes both solved"
    return f"{success}; {c.speedup:.2f}x on unmatched averages, so the ratio is indicative only"


def format_summary(runs: list[Run], reference: Run) -> str:
    """The closing block: scope, whether pairing holds, and each arm's verdict.

    Everything here is stated per comparison rather than per run, because that is
    what the design supports: the runs share scenes, so the quantity with an interval
    around it is the *difference*, not either rate on its own.
    """
    lines = ["summary"]
    cfg = reference.config
    n_ep = cfg.get("n_episodes")
    per_task = sorted({t.n_episodes for run in runs for t in run.tasks.values()})
    episodes_desc = f"{per_task[0]}" if len(per_task) == 1 else f"{per_task[0]}-{per_task[-1]}"
    lines.append(
        f"  scope    {len(runs)} runs | {len(reference.tasks)} tasks | {episodes_desc} episodes per task "
        f"| {len(reference.episodes)} per run"
    )
    if n_ep and n_ep <= LIBERO_INIT_STATES:
        lines.append(
            f"           episode k is init state k, so this covers {n_ep} of the "
            f"{LIBERO_INIT_STATES} initial states LIBERO ships per task"
            + (" -- the full set" if n_ep == LIBERO_INIT_STATES else "")
        )

    comparisons = [compare(reference, r) for r in runs if r is not reference]
    if not comparisons:
        return "\n".join(lines)

    for c, adj in zip(comparisons, holm_adjust([c.p_value for c in comparisons]), strict=True):
        c.p_adjusted = adj
    unpaired = [c.label for c in comparisons if not c.paired]
    lines.append(
        f"  pairing  every run walked the same scenes in the same order"
        if not unpaired
        else f"  pairing  NOT shared by {', '.join(unpaired)} -- their configs put them on other scenes"
    )
    multiple = len(comparisons) > 1

    by_label = {r.label: r for r in runs}
    for c in comparisons:
        other = by_label[c.label]
        lines.append("")
        lines.append(f"  {c.label} vs {c.reference}")
        ci = (
            f" [{c.delta_sr_ci[0]:+.1f}, {c.delta_sr_ci[1]:+.1f}]"
            if c.delta_sr_ci and not math.isnan(c.delta_sr_ci[0])
            else ""
        )
        p_txt = "n/a (unpaired)" if math.isnan(c.p_value) else f"{c.p_value:.3f}"
        if multiple and not math.isnan(c.p_adjusted):
            p_txt += f" (Holm {c.p_adjusted:.3f})"
        lines.append(
            f"    success  {other.success_rate:.1f}% vs {reference.success_rate:.1f}%  |  "
            f"{c.delta_sr:+.1f} pp{ci}  |  McNemar p={p_txt}"
        )
        if c.ref_time and c.other_time:
            sci = (
                f" [{c.speedup_ci[0]:.2f}, {c.speedup_ci[1]:.2f}]"
                if c.speedup_ci and not math.isnan(c.speedup_ci[0])
                else ""
            )
            detail = (
                f"faster on {c.faster}/{c.faster + c.slower} shared successes, sign p={c.time_p:.3g}"
                if c.n_time_paired
                else "each run averaged over its own successes -- not matched scene by scene"
            )
            lines.append(
                f"    time     {c.other_time:.2f} s vs {c.ref_time:.2f} s  |  "
                f"{c.speedup:.2f}x{sci}  |  {detail}"
            )
        lines.append(f"    reading  {_verdict(c)}")
    return "\n".join(lines)


def format_report(runs: list[Run], reference: Run) -> str:
    """The terminal report: a summary table, a per-task grid, and the paired tests."""
    lines: list[str] = []
    width = max(len(r.label) for r in runs)

    lines.append("run".ljust(width) + "  tasks   eps       SR   95% CI            ATR    speedup")
    lines.append("-" * (width + 54))
    for run in runs:
        eps = run.episodes
        lo, hi = wilson_interval(sum(eps), len(eps))
        cmp_ = compare(reference, run) if run is not reference else None
        speedup = 1.0 if run is reference else (cmp_.speedup if cmp_ else None)
        lines.append(
            f"{run.label.ljust(width)}  {len(run.tasks):5d} {len(eps):5d}  "
            f"{run.success_rate:6.1f}%  [{lo:5.1f},{hi:6.1f}]  "
            f"{_fmt(run.avg_success_time):>6} s  {_fmt(speedup, '.2f') + 'x':>7}"
        )

    task_ids = sorted({t for run in runs for t in run.tasks})
    lines.append("")
    lines.append("success rate per task (%)")
    lines.append("run".ljust(width) + "".join(f"  t{t:<3d}" for t in task_ids))
    for run in runs:
        cells = []
        for t in task_ids:
            task = run.tasks.get(t)
            if task is None or not task.successes:
                cells.append("     -")
            else:
                cells.append(f"  {100 * sum(task.successes) / task.n_episodes:4.0f}")
        lines.append(run.label.ljust(width) + "".join(cells))

    others = [r for r in runs if r is not reference]
    if others:
        lines.append("")
        lines.append(f"paired against {reference.label!r} (same suite, tasks, seed and init states)")
        for run in others:
            c = compare(reference, run)
            if not c.paired:
                lines.append(
                    f"  {run.label}: NOT paired -- "
                    f"{dict(zip(PAIRING_KEYS, run.pairing_signature(), strict=False))} differs from the "
                    "reference; success rates are still comparable, the test is not."
                )
                continue
            lines.append(
                f"  {run.label}: dSR {c.delta_sr:+.1f} pp over {c.n_paired} paired episodes | "
                f"won {c.only_other}, lost {c.only_reference} | McNemar p={c.p_value:.3f}"
            )
            if c.n_time_paired:
                lines.append(
                    f"    time on the {c.n_time_paired} episodes both solved: "
                    f"{c.ref_time:.2f} s -> {c.other_time:.2f} s  ({c.speedup:.2f}x)"
                )
            elif c.speedup is not None:
                lines.append(
                    f"    time: {c.ref_time:.2f} s -> {c.other_time:.2f} s  ({c.speedup:.2f}x), each run "
                    "averaged over its own successes -- one of them predates per-episode indices, "
                    "so the durations are not matched scene by scene"
                )
    return "\n".join(lines) + "\n\n" + format_summary(runs, reference)


# --- cli -----------------------------------------------------------------------

def _labels_for(paths: list[Path], override: str | None) -> list[str]:
    if override:
        labels = [s.strip() for s in override.split(",")]
        if len(labels) != len(paths):
            raise SystemExit(f"--labels has {len(labels)} entries for {len(paths)} run directories")
        return labels
    names = [p.name for p in paths]
    if len(set(names)) == len(names):
        return names
    return [str(p) for p in paths]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="eval output directories (each holding task_*/)")
    parser.add_argument("--labels", help="comma-separated display names, one per run directory")
    parser.add_argument("--baseline", help="label or path of the reference run (default: the first)")
    parser.add_argument("--json", type=Path, help="also write the numbers here")
    args = parser.parse_args(argv)

    labels = _labels_for(args.runs, args.labels)
    runs = [load_run(path, label) for path, label in zip(args.runs, labels, strict=True)]

    reference = runs[0]
    if args.baseline:
        matches = [r for r in runs if r.label == args.baseline or str(r.path) == args.baseline]
        if not matches:
            raise SystemExit(f"--baseline={args.baseline!r} matches none of {[r.label for r in runs]}")
        reference = matches[0]

    print(format_report(runs, reference))

    if args.json:
        payload = {
            "reference": reference.label,
            "runs": [
                {
                    "label": r.label,
                    "path": str(r.path),
                    "n_tasks": len(r.tasks),
                    "n_episodes": len(r.episodes),
                    "success_rate": r.success_rate,
                    "avg_success_sim_s": r.avg_success_time,
                    "per_task_success_rate": {
                        str(t): 100 * sum(v.successes) / v.n_episodes if v.successes else None
                        for t, v in sorted(r.tasks.items())
                    },
                    "config": r.config,
                }
                for r in runs
            ],
            "comparisons": [vars(compare(reference, r)) for r in runs if r is not reference],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
