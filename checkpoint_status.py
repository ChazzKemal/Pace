#!/usr/bin/env python3
"""Which benchmark cells are trained, and where their last checkpoint is.

The benchmark grid is dataset x backbone x method. PACE contributes no cell of
its own -- it acts at eval time on the baseline arm's weights -- so the three
columns that need weights are `baseline` (which is also the PACE arm),
`demospeedup` and `bspline`.

A cell is filled by reading each run's own record rather than by trusting its
directory name: `train_config.json` beside the last checkpoint names the
dataset, the policy and the method it was actually launched with. A run that
died before its first checkpoint has no such file, so its wandb metadata (which
records the argv) stands in -- that is the only way an arm that started and
produced nothing can be told apart from one that was never started at all.

    python checkpoint_status.py                 # terminal table
    python checkpoint_status.py --html out.html # + the status page

Rerun after any training run; nothing here is cached.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent
TRAIN_ROOT = REPO / "outputs" / "train"
DATA_ROOT = Path(os.environ.get("PACE_DATA_ROOT", REPO.parent / "data"))

# --- the grid ---------------------------------------------------------------
# Each dataset names the `--dataset.repo_id` its arms are trained under. An
# empty repo_id means the recording does not exist yet: the row is real, the
# data is not.

METHODS = [
    ("baseline", "Baseline", "also the PACE arm"),
    ("demospeedup", "DemoSpeedup", "retimed training targets"),
    ("bspline", "B-spline", "spline action space"),
]


@dataclass
class Dataset:
    key: str
    label: str
    kind: str
    repo_id: str | None
    root: Path | None
    backbones: list[str]
    note: str = ""


DATASETS = [
    Dataset("pickplace", "pickplace", "real", "local/pickplace",
            DATA_ROOT / "datasets/real/pickplace_cart7_v2_angleaxis_nogrip",
            ["act", "diffusion"]),
    Dataset("cups_stacking", "cups_stacking", "real", "local/stack_cups_merged",
            DATA_ROOT / "datasets/real/stackcups_20260829_merged",
            ["act", "diffusion"],
            "175 merged episodes; supersedes the 12-episode local/stack_cups"),
    Dataset("table_clean", "table_clean", "real", None, None,
            ["act", "diffusion"], "not recorded yet"),
    Dataset("libero_10", "libero_10", "sim", "local/libero_10_ee6d",
            DATA_ROOT / "datasets/sim/libero_10_ee6d",
            ["xvla"], "xVLA LoRA-finetuned from lerobot/xvla-libero"),
]

BACKBONE_LABEL = {"act": "ACT", "diffusion": "Diffusion", "xvla": "xVLA"}

# repo_ids that belong to a row's history but must not fill its cells.
SUPERSEDED = {"local/stack_cups": "cups_stacking"}


# --- reading a run ----------------------------------------------------------

@dataclass
class Run:
    name: str
    dir: Path
    repo_id: str | None = None
    dataset_root: str | None = None
    policy: str | None = None
    method: str | None = None
    budget: int | None = None
    batch_size: int | None = None
    steps: int = 0
    ckpt: Path | None = None
    mtime: dt.datetime | None = None
    started: dt.datetime | None = None
    resumable: bool = False
    running: bool = False
    method_cfg: dict = field(default_factory=dict)
    source: str = ""

    @property
    def status(self) -> str:
        if self.running:
            return "running"
        if self.ckpt is None:
            return "failed"
        if self.budget and self.steps < self.budget:
            return "partial"
        return "trained"


def _parse_args(argv: list[str]) -> dict[str, str]:
    out = {}
    for a in argv:
        if a.startswith("--") and "=" in a:
            k, _, v = a[2:].partition("=")
            out[k] = v
    return out


def _weights_mtime(pretrained: Path) -> dt.datetime | None:
    # ACT/Diffusion write model.safetensors; a LoRA run writes only the adapter.
    for name in ("model.safetensors", "adapter_model.safetensors"):
        f = pretrained / name
        if f.exists():
            return dt.datetime.fromtimestamp(f.stat().st_mtime)
    return None


def _running_output_dirs() -> set[str]:
    """Output dirs of live training processes, as named on their command line."""
    try:
        ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True,
                            timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    dirs = set()
    for line in ps.splitlines():
        if "run_train" not in line:
            continue
        for tok in line.split():
            # --output_dir=outputs/train/X, or a resume's --config_path into it
            if tok.startswith("--output_dir="):
                dirs.add(Path(tok.split("=", 1)[1]).name)
            elif tok.startswith("--config_path=") and "/outputs/train/" in tok:
                tail = tok.split("/outputs/train/", 1)[1]
                dirs.add(tail.split("/", 1)[0])
    return dirs


def scan() -> list[Run]:
    live = _running_output_dirs()
    runs = []
    if not TRAIN_ROOT.is_dir():
        return runs
    for d in sorted(TRAIN_ROOT.iterdir()):
        if not d.is_dir():
            continue
        run = Run(name=d.name, dir=d, running=d.name in live)

        last = d / "checkpoints" / "last"
        if last.exists():
            real = last.resolve()
            run.ckpt = real / "pretrained_model"
            run.resumable = (real / "training_state").is_dir()
            try:
                run.steps = int(real.name)
            except ValueError:
                run.steps = 0
            run.mtime = _weights_mtime(run.ckpt)

        cfg_file = (run.ckpt / "train_config.json") if run.ckpt else None
        if cfg_file and cfg_file.is_file():
            cfg = json.loads(cfg_file.read_text())
            run.source = "train_config.json"
            run.repo_id = cfg.get("dataset", {}).get("repo_id")
            run.dataset_root = cfg.get("dataset", {}).get("root")
            run.policy = cfg.get("policy", {}).get("type")
            run.budget = cfg.get("steps")
            run.batch_size = cfg.get("batch_size")
            run.method_cfg = cfg.get("method") or {}
            run.method = run.method_cfg.get("type")
        else:
            meta = d / "wandb" / "latest-run" / "files" / "wandb-metadata.json"
            if meta.is_file():
                m = json.loads(meta.read_text())
                run.source = "wandb metadata (no checkpoint written)"
                a = _parse_args(m.get("args", []))
                run.repo_id = a.get("dataset.repo_id")
                run.dataset_root = a.get("dataset.root")
                run.policy = a.get("policy.type")
                run.budget = int(a["steps"]) if a.get("steps", "").isdigit() else None
                run.batch_size = int(a["batch_size"]) if a.get("batch_size", "").isdigit() else None
                run.method = a.get("method.type")
                started = m.get("startedAt")
                if started:
                    run.started = dt.datetime.fromisoformat(
                        started.replace("Z", "+00:00")).astimezone().replace(tzinfo=None)

        if run.method == "none":
            run.method = "baseline"
        runs.append(run)
    return runs


def assign(runs: list[Run]):
    """(dataset_key, backbone, method) -> Run, plus the runs that fill no cell."""
    cells: dict[tuple[str, str, str], Run] = {}
    orphans: list[tuple[Run, str]] = []
    by_repo = {d.repo_id: d for d in DATASETS if d.repo_id}
    for run in runs:
        ds = by_repo.get(run.repo_id or "")
        if ds is None:
            why = ("superseded predecessor of "
                   f"{SUPERSEDED[run.repo_id]}" if run.repo_id in SUPERSEDED
                   else f"no grid row for dataset {run.repo_id!r}")
            orphans.append((run, why))
            continue
        if run.policy not in ds.backbones:
            orphans.append((run, f"{run.policy} is not a {ds.key} backbone"))
            continue
        key = (ds.key, run.policy, run.method)
        prev = cells.get(key)
        # Two runs for one cell: keep the one that got further.
        if prev is None or (run.steps, run.status == "trained") > (prev.steps, prev.status == "trained"):
            if prev is not None:
                orphans.append((prev, "superseded by a further-trained run in the same cell"))
            cells[key] = run
        else:
            orphans.append((run, "superseded by a further-trained run in the same cell"))
    return cells, orphans


# --- formatting -------------------------------------------------------------

def fmt_steps(n: int | None) -> str:
    if not n:
        return "0"
    return f"{n / 1000:g}k" if n >= 1000 else str(n)


def fmt_when(when: dt.datetime | None) -> str:
    return when.strftime("%Y-%m-%d %H:%M") if when else "--"


def ago(when: dt.datetime | None, now: dt.datetime) -> str:
    if when is None:
        return ""
    s = (now - when).total_seconds()
    if s < 3600:
        return f"{int(s // 60)} min ago"
    if s < 86400:
        return f"{int(s // 3600)} h ago"
    return f"{int(s // 86400)} d ago"


STATUS_TEXT = {
    "trained": "trained",
    "running": "training now",
    "partial": "partial",
    "failed": "no checkpoint",
    "missing": "not started",
    "nodata": "no dataset",
}


def cell_rows():
    """Every grid cell, in reading order."""
    for ds in DATASETS:
        for bb in ds.backbones:
            yield ds, bb


# --- terminal rendering ------------------------------------------------------
# Colour and hyperlinks are both terminal capabilities, so both are decided once,
# from the same signal: a TTY that has not asked to be left alone. NO_COLOR is
# honoured because a status board that ignores it is not usable in a pipe.

HOST = socket.gethostname()

ANSI = {
    "trained": "\033[38;5;36m",   # teal-green
    "running": "\033[38;5;33m",   # blue
    "partial": "\033[38;5;178m",  # amber
    "failed": "\033[38;5;167m",   # brick
    "missing": "\033[38;5;244m",  # grey
    "nodata": "\033[38;5;240m",   # dimmer grey
    "head": "\033[1m",
    "rule": "\033[38;5;238m",
    "dim": "\033[38;5;245m",
    "off": "\033[0m",
}

MARK = {
    "trained": "●",
    "running": "▶",
    "partial": "◐",
    "failed": "✕",
    "missing": "○",
    "nodata": "·",
}


class Term:
    """Whether this output supports colour and OSC 8 hyperlinks, and how to emit them."""

    def __init__(self, color: bool, links: bool):
        self.color, self.links = color, links

    def paint(self, text: str, key: str) -> str:
        if not self.color or key not in ANSI:
            return text
        return f"{ANSI[key]}{text}{ANSI['off']}"

    def link(self, text: str, target: Path) -> str:
        """An OSC 8 hyperlink, which terminals render as a clickable path.

        The escape sequence carries no printable width, so anything aligning
        columns has to measure the *text*, never the returned string -- see `pad`.

        The URL carries this machine's hostname rather than being host-less. A
        bare `file:///path` claims the path is on whichever machine is drawing the
        terminal, which over SSH is the wrong one; naming the host makes the URL
        true, and a terminal that cannot act on a remote host does nothing instead
        of opening some unrelated local path. It is also why this is not the only
        way the paths are offered -- see `print_paths`.
        """
        if not self.links:
            return text
        return f"\033]8;;file://{HOST}{target}\033\\{text}\033]8;;\033\\"

    @staticmethod
    def pad(text: str, width: int) -> str:
        """Pad to a visible width, ignoring escape sequences already in `text`."""
        visible = re.sub(r"\033(?:\]8;;[^\033]*\033\\|\[[0-9;]*m)", "", text)
        return text + " " * max(0, width - len(visible))


def detect_term(force_color: bool | None, force_links: bool | None) -> Term:
    tty = sys.stdout.isatty()
    color = tty and not os.environ.get("NO_COLOR")
    if force_color is not None:
        color = force_color
    return Term(color, color if force_links is None else force_links)


def bar(done: int, total: int, width: int = 8) -> str:
    filled = max(1, round(width * done / total)) if total else 0
    return "▰" * filled + "▱" * (width - filled)


def dataset_cell(ds: Dataset, term: Term) -> tuple[str, str]:
    """(display text, rendered) for the dataset behind a group of arms.

    Carries the absolute root, not just the repo_id: the repo_id is the name the
    arms were trained under and says nothing about where the recording is, and
    "where is it" is what a band is asked. Shown in full for the same reason the
    checkpoints are -- an editor's terminal links a path it can see.

    A row whose dataset is not on this machine says so. `table_clean` has no
    recording at all, and a dataset that has merely moved is worth catching here
    rather than in a queue that stops six hours in.
    """
    if ds.repo_id is None:
        text = ds.note or "no dataset recorded yet"
        return (text, term.paint(text, "dim"))
    if ds.root is None:
        return (ds.repo_id, term.paint(ds.repo_id, "dim"))
    missing = not ds.root.exists()
    text = f"{ds.repo_id}  {ds.root}" + ("  (missing)" if missing else "")
    rendered = term.paint(ds.repo_id, "dim") + "  " + term.link(str(ds.root), ds.root)
    if missing:
        rendered += "  " + term.paint("(missing)", "failed")
    return (text, rendered)


def steps_cell(run: Run) -> str:
    """Exact when the arm is finished; against its budget, with a bar, when not.

    The two formats differ deliberately: a finished arm's step count is a fact to
    read, an unfinished one's is a position to judge, and the budget is what makes
    it judgeable. `checkpoints/last` existing is not proof an arm finished, so the
    unfinished case must never be able to look like the finished one.
    """
    if run.budget and run.steps < run.budget:
        return (f"{fmt_steps(run.steps)} / {fmt_steps(run.budget)} "
                f"{bar(run.steps, run.budget)}")
    return f"{run.steps:,}".replace(",", " ")


def ckpt_cell(run: Run, term: Term) -> tuple[str, str]:
    """(display text, rendered cell). The text is what the column has to fit."""
    if run.ckpt is None:
        return ("no checkpoint written", term.paint("no checkpoint written", "dim"))
    # The run dir and its step are the identifying part; the rest of the path is
    # the same for every arm, so it is carried by the link rather than the column.
    text = f"{run.dir.name} @ {run.ckpt.parent.name}"
    return (text, term.link(text, run.ckpt))


def print_paths(cells, term: Term, bare: bool = False):
    """Every checkpoint as a full absolute path, one per line.

    The table above shows `<run> @ <step>` because a 100-character path in a
    column makes the other four unreadable. That trade costs something real when
    the terminal is not on this machine: an editor's terminal (VS Code over
    Remote-SSH, say) finds and opens absolute paths it can *see*, and it cannot
    see one that a display string is standing in for. So the paths are printed
    once, in full, unindented and unadorned -- which is also the form that
    survives a copy-paste into a --policy_path argument.
    """
    rows = []
    for ds, bb in cell_rows():
        for mkey, mlabel, _ in METHODS:
            run = cells.get((ds.key, bb, mkey))
            if run is None or run.ckpt is None:
                continue
            rows.append((f"{ds.label}/{bb}/{mkey}", run.ckpt))
    if not rows:
        return
    if bare:
        for _, ckpt in rows:
            print(ckpt)
        return
    width = max(len(label) for label, _ in rows)
    print("  " + term.paint("checkpoints, in full", "head"))
    for label, ckpt in rows:
        print(f"  {term.paint(Term.pad(label, width), 'dim')}  {ckpt}")


def print_table(cells, orphans, now, term: Term):
    total = sum(len(ds.backbones) for ds in DATASETS) * len(METHODS)
    done = sum(1 for r in cells.values() if r.status == "trained")

    # Build every row first: the column widths are whatever the content needs.
    groups = []  # (dataset, backbone, [(arm, status, steps, when, ckpt_text, ckpt_cell)])
    counts = {k: 0 for k in ("trained", "running", "partial", "failed", "missing", "nodata")}
    for ds, bb in cell_rows():
        rows = []
        for mkey, mlabel, _ in METHODS:
            run = cells.get((ds.key, bb, mkey))
            arm = mlabel + (" / PACE" if mkey == "baseline" else "")
            if run is None:
                state = "nodata" if ds.repo_id is None else "missing"
                counts[state] += 1
                rows.append((arm, state, STATUS_TEXT[state], "", "", "-",
                             term.paint("-", "dim"), None))
                continue
            counts[run.status] += 1
            when = run.mtime or run.started
            text, cell = ckpt_cell(run, term)
            steps = steps_cell(run)
            rows.append((arm, run.status, STATUS_TEXT[run.status], steps,
                         fmt_when(when) if when else "-", text, cell, run))
        band_text, band_rendered = dataset_cell(ds, term)
        groups.append((ds, bb, rows, band_text, band_rendered))

    w_arm = max(len(r[0]) for _, _, rows, _, _ in groups for r in rows)
    w_stat = max(len(r[2]) for _, _, rows, _, _ in groups for r in rows) + 2
    w_steps = max(9, max(len(r[3]) for _, _, rows, _, _ in groups for r in rows))
    w_when = 16
    w_path = max(len(r[5]) for _, _, rows, _, _ in groups for r in rows)
    widths = [w_arm, w_stat, w_steps, w_when, w_path]
    inner = sum(widths) + 3 * (len(widths) - 1)
    # A band spans every column, so a long dataset path would print past the right
    # border. Give the slack to the last column instead: `inner` is a function of
    # the column widths (each drawn with two spaces of padding, plus one separator
    # between them), so widening a column is the only way to widen the box and keep
    # the rules, the header and the rows aligned with each other.
    widest_band = max(len(f"{ds.label} / {BACKBONE_LABEL.get(bb, bb)}  {text}")
                      for ds, bb, _, text, _ in groups)
    if widest_band > inner:
        widths[-1] += widest_band - inner
        inner = sum(widths) + 3 * (len(widths) - 1)

    def rule(left, mid, right, fill="─"):
        return term.paint(left + mid.join(fill * (w + 2) for w in widths) + right, "rule")

    def span(left, right, fill="─"):
        return term.paint(left + fill * (inner + 2) + right, "rule")

    def row(cellsr, keys=None):
        bar_ = term.paint("│", "rule")
        out = []
        for i, c in enumerate(cellsr):
            painted = term.paint(c, keys[i]) if keys and keys[i] else c
            out.append(" " + Term.pad(painted, widths[i]) + " ")
        return bar_ + bar_.join(out) + bar_

    print()
    title = f"pace_bench checkpoints — {done} of {total} cells trained"
    print("  " + term.paint(title, "head"))
    legend = "  ".join(
        term.paint(f"{MARK[k]} {n} {STATUS_TEXT[k]}", k)
        for k, n in counts.items() if n
    )
    print("  " + legend)
    print("  " + term.paint(f"{now:%Y-%m-%d %H:%M}", "dim"))
    print()

    head = ["Arm", "Status", "Steps", "Last written", "Checkpoint"]
    print(rule("┌", "┬", "┐"))
    print(row([term.paint(h, "head") for h in head]))

    for gi, (ds, bb, rows, _text, band_rendered) in enumerate(groups):
        # A spanning band names the dataset and backbone, so the arm column stays
        # narrow instead of repeating "pickplace / ACT" on every line.
        print(rule("├", "┴", "┤") if gi == 0 else span("├", "┤"))
        label = f"{ds.label} / {BACKBONE_LABEL.get(bb, bb)}"
        band = term.paint(label, "head") + "  " + band_rendered
        print(term.paint("│", "rule") + " " + Term.pad(band, inner) + " "
              + term.paint("│", "rule"))
        print(rule("├", "┬", "┤"))
        for arm, status, stext, steps, when, _t, cell, _run in rows:
            mark = f"{MARK[status]} {stext}"
            print(row([arm, mark, steps, when, cell],
                      keys=[None, status, None, "dim", None]))
    print(rule("└", "┴", "┘"))

    print()
    print_paths(cells, term)
    if orphans:
        print()
        print("  " + term.paint("runs that fill no cell", "head"))
        for run, why in orphans:
            print("  " + term.paint(f"  {run.name:<28} {why}", "dim"))
    print()


# --- the status page --------------------------------------------------------

CSS = """
:root{
  --ground:#eceef1; --surface:#ffffff; --surface-2:#f5f6f8;
  --ink:#161a20; --ink-2:#5b6573; --ink-3:#8b94a1;
  --rule:#d9dde3; --rule-2:#c3c9d2; --structural:#3f4d68;
  --ok:#1c7a58; --run:#2360c4; --part:#a4700c; --fail:#b03a34; --none:#98a0ad;
  --ok-bg:#e4f2ec; --run-bg:#e4ecfa; --part-bg:#f7eeda; --fail-bg:#f8e6e4; --none-bg:#eef0f3;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#101318; --surface:#181c22; --surface-2:#1f242c;
    --ink:#e7eaef; --ink-2:#9aa3b1; --ink-3:#6d7684;
    --rule:#2a303a; --rule-2:#3a424e; --structural:#93a4c4;
    --ok:#4dbf94; --run:#71a6f7; --part:#d7a13f; --fail:#e5786f; --none:#79828f;
    --ok-bg:#152a23; --run-bg:#16223a; --part-bg:#2c2517; --fail-bg:#2e1c1b; --none-bg:#1e232a;
  }
}
:root[data-theme="dark"]{
  --ground:#101318; --surface:#181c22; --surface-2:#1f242c;
  --ink:#e7eaef; --ink-2:#9aa3b1; --ink-3:#6d7684;
  --rule:#2a303a; --rule-2:#3a424e; --structural:#93a4c4;
  --ok:#4dbf94; --run:#71a6f7; --part:#d7a13f; --fail:#e5786f; --none:#79828f;
  --ok-bg:#152a23; --run-bg:#16223a; --part-bg:#2c2517; --fail-bg:#2e1c1b; --none-bg:#1e232a;
}

*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif;
  font-size:15px; line-height:1.5;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1080px; margin:0 auto; padding:44px 28px 80px;
      display:flex; flex-direction:column; gap:34px}

.eyebrow{
  font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;
  text-transform:uppercase; letter-spacing:.14em; font-size:11.5px;
  font-weight:600; color:var(--structural);
}
h1{font-size:30px; line-height:1.15; margin:6px 0 0; font-weight:600;
   letter-spacing:-.015em; text-wrap:balance}
.lede{margin:10px 0 0; color:var(--ink-2); max-width:62ch}
.mono{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
.tnum{font-variant-numeric:tabular-nums}

/* header + tally */
header{display:flex; flex-wrap:wrap; gap:28px; align-items:flex-end;
       justify-content:space-between; border-bottom:1px solid var(--rule);
       padding-bottom:26px}
.tally{display:flex; align-items:baseline; gap:9px}
.tally .n{font-family:"IBM Plex Mono",monospace; font-size:46px; font-weight:500;
          letter-spacing:-.03em; line-height:1; font-variant-numeric:tabular-nums}
.tally .of{font-family:"IBM Plex Mono",monospace; font-size:22px; color:var(--ink-3)}
.tally .cap{font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;
            text-transform:uppercase; letter-spacing:.12em; font-size:11px;
            color:var(--ink-2); font-weight:600}

.legend{display:flex; flex-wrap:wrap; gap:8px}
.chip{display:inline-flex; align-items:center; gap:7px; padding:3px 10px 3px 8px;
      border-radius:2px; font-size:12px; font-weight:500;
      font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;
      letter-spacing:.03em; border:1px solid var(--rule-2); background:var(--surface)}
.dot{width:8px; height:8px; border-radius:50%; flex:none}
.s-trained .dot{background:var(--ok)} .s-trained{color:var(--ok); border-color:var(--ok)}
.s-running .dot{background:var(--run)} .s-running{color:var(--run); border-color:var(--run)}
.s-partial .dot{background:var(--part)} .s-partial{color:var(--part); border-color:var(--part)}
.s-failed  .dot{background:var(--fail)} .s-failed{color:var(--fail); border-color:var(--fail)}
.s-missing .dot{background:var(--none)} .s-missing{color:var(--ink-2)}
.s-nodata  .dot{background:transparent; border:1px dashed var(--none)} .s-nodata{color:var(--ink-3)}

/* matrix */
section{display:flex; flex-direction:column; gap:14px}
.matrix{width:100%; border-collapse:collapse; background:var(--surface);
        border:1px solid var(--rule)}
.matrix th, .matrix td{padding:0; text-align:left}
.matrix thead th{
  font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;
  text-transform:uppercase; letter-spacing:.1em; font-size:11px; font-weight:600;
  color:var(--ink-2); padding:11px 14px; border-bottom:1px solid var(--rule-2);
  vertical-align:bottom;
}
.matrix thead th span{display:block; text-transform:none; letter-spacing:0;
                      font-weight:400; color:var(--ink-3); font-size:11px;
                      margin-top:2px}
.matrix tbody th{padding:11px 14px; border-bottom:1px solid var(--rule);
                 border-right:1px solid var(--rule); font-weight:500; font-size:13.5px;
                 white-space:nowrap; vertical-align:middle}
.matrix tbody th small{display:block; color:var(--ink-3); font-weight:400; font-size:11.5px}
.matrix tbody td{border-bottom:1px solid var(--rule); border-right:1px solid var(--rule)}
.matrix tbody tr:last-child th, .matrix tbody tr:last-child td{border-bottom:none}
.matrix tbody td:last-child{border-right:none}
.matrix tbody th{border-left:3px solid transparent}
tr.grp-start th{border-top:1px solid var(--rule-2)}
tr.grp-start td{border-top:1px solid var(--rule-2)}
.matrix tbody th.real{border-left-color:var(--structural)}
.matrix tbody th.sim{border-left-color:var(--rule-2)}

a.cell{display:flex; flex-direction:column; gap:2px; padding:10px 14px;
       min-height:56px; justify-content:center; text-decoration:none; color:inherit}
a.cell:hover{background:var(--surface-2)}
a.cell:focus-visible{outline:2px solid var(--structural); outline-offset:-2px}
.cell .k{display:flex; align-items:center; gap:7px; font-size:12.5px;
         font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;
         letter-spacing:.03em; font-weight:600}
.cell .v{font-family:"IBM Plex Mono",monospace; font-size:13px;
         font-variant-numeric:tabular-nums; color:var(--ink)}
.cell .v em{font-style:normal; color:var(--ink-3)}
.cell.empty{cursor:default}
.cell.empty .v{color:var(--ink-3)}

/* detail */
.card{background:var(--surface); border:1px solid var(--rule)}
.card > h3{margin:0; padding:12px 16px; font-size:14px; font-weight:600;
           border-bottom:1px solid var(--rule); display:flex; flex-wrap:wrap;
           gap:10px; align-items:baseline}
.card > h3 .note{font-weight:400; font-size:12.5px; color:var(--ink-3)}
.scroll{overflow-x:auto}
table.detail{width:100%; border-collapse:collapse; font-size:13px}
table.detail th{font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;
  text-transform:uppercase; letter-spacing:.1em; font-size:10.5px; font-weight:600;
  color:var(--ink-3); text-align:left; padding:9px 16px;
  border-bottom:1px solid var(--rule); white-space:nowrap}
table.detail td{padding:10px 16px; border-bottom:1px solid var(--rule);
                vertical-align:top}
table.detail tbody tr:last-child td{border-bottom:none}
table.detail tr:target{background:var(--surface-2)}
table.detail tr:target td:first-child{box-shadow:inset 3px 0 0 var(--structural)}
td.arm{white-space:nowrap; font-weight:500}
td.arm small{display:block; color:var(--ink-3); font-weight:400; font-size:11.5px}
td.num{font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums;
       white-space:nowrap}
td.num em{font-style:normal; color:var(--ink-3)}
td.when{white-space:nowrap; font-family:"IBM Plex Mono",monospace; font-size:12.5px}
td.when small{display:block; font-family:"IBM Plex Sans",sans-serif; color:var(--ink-3);
              font-size:11.5px}
td.path{font-size:12px; color:var(--ink-2); word-break:break-all; min-width:24ch}
td.path .dim{font-family:"IBM Plex Mono",monospace; color:var(--ink-3)}
button.copy{
  font-family:"IBM Plex Mono",monospace; font-size:12px; line-height:1.45;
  color:var(--ink); background:var(--surface-2); border:1px solid var(--rule-2);
  border-radius:2px; padding:5px 8px; text-align:left; cursor:pointer;
  display:flex; align-items:flex-start; gap:8px; width:100%; word-break:break-all;
}
button.copy:hover{border-color:var(--structural)}
button.copy:focus-visible{outline:2px solid var(--structural); outline-offset:1px}
button.copy .ico{flex:none; color:var(--ink-3);
  font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;
  font-size:10px; text-transform:uppercase; letter-spacing:.09em; font-weight:600;
  padding-top:1px}
button.copy[data-copied] .ico{color:var(--ok)}
button.copy .run{color:var(--ink)}
button.copy .rest{color:var(--ink-3)}
.dsroot{margin:0; padding:10px 16px; border-bottom:1px solid var(--rule);
        display:flex; flex-wrap:wrap; gap:10px; align-items:center}
.dsroot button.copy{width:auto; max-width:100%}
.dsroot.none{font-size:12.5px; color:var(--ink-3)}
.dsroot .warn{font-size:12px; font-weight:600; color:var(--fail);
              font-family:"IBM Plex Sans Condensed","IBM Plex Sans",sans-serif;
              text-transform:uppercase; letter-spacing:.08em}

.bar{height:4px; background:var(--none-bg); margin-top:5px; width:100%;
     max-width:120px; overflow:hidden}
.bar i{display:block; height:100%; background:var(--part)}

.footnotes{border-top:1px solid var(--rule); padding-top:22px;
           display:flex; flex-direction:column; gap:12px; font-size:13px;
           color:var(--ink-2); max-width:74ch}
.footnotes b{color:var(--ink); font-weight:600}
.footnotes code{font-family:"IBM Plex Mono",monospace; font-size:12px;
                background:var(--surface); border:1px solid var(--rule);
                padding:1px 5px}
@media (max-width:720px){
  .wrap{padding:30px 16px 60px}
  h1{font-size:24px}
  .matrix thead th, .matrix tbody th{padding:9px 10px}
  a.cell{padding:9px 10px}
}
@media (prefers-reduced-motion:reduce){*{animation:none!important; transition:none!important}}
"""


COPY_JS = """
<script>
// navigator.clipboard is unavailable in some embedding contexts, so the
// execCommand path stays as a fallback rather than being assumed dead.
(function () {
  function legacy(text) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    var ok = false;
    try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
    document.body.removeChild(ta);
    return ok;
  }
  function flash(btn, label) {
    var ico = btn.querySelector(".ico");
    if (!ico) return;
    if (btn.dataset.timer) clearTimeout(Number(btn.dataset.timer));
    ico.textContent = label;
    btn.dataset.copied = "1";
    btn.dataset.timer = String(setTimeout(function () {
      ico.textContent = "copy";
      delete btn.dataset.copied;
    }, 1400));
  }
  document.addEventListener("click", function (ev) {
    var btn = ev.target.closest ? ev.target.closest("button.copy") : null;
    if (!btn) return;
    var path = btn.dataset.path || "";
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(path).then(
        function () { flash(btn, "copied"); },
        function () { flash(btn, legacy(path) ? "copied" : "select it"); }
      );
    } else {
      flash(btn, legacy(path) ? "copied" : "select it");
    }
  });
})();
</script>
"""


def _cell_id(ds_key, bb, mkey):
    return f"{ds_key}-{bb}-{mkey}"


def render_html(cells, orphans, now) -> str:
    e = html.escape
    total = sum(len(ds.backbones) for ds in DATASETS) * len(METHODS)
    counts = {k: 0 for k in ("trained", "running", "partial", "failed", "missing", "nodata")}
    for ds, bb in cell_rows():
        for mkey, _, _ in METHODS:
            run = cells.get((ds.key, bb, mkey))
            if run is None:
                counts["nodata" if ds.repo_id is None else "missing"] += 1
            else:
                counts[run.status] += 1

    out = ['<title>pace_bench Checkpoints</title>',
           '<link rel="preconnect" href="https://fonts.googleapis.com">',
           '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
           '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
           'family=IBM+Plex+Mono:wght@400;500&'
           'family=IBM+Plex+Sans+Condensed:wght@400;600&'
           'family=IBM+Plex+Sans:wght@400;500;600&display=swap">',
           f'<style>{CSS}</style>', '<div class="wrap">']

    # header
    out.append('<header><div>')
    out.append('<p class="eyebrow">Training status</p>')
    out.append('<h1>pace_bench checkpoints</h1>')
    out.append('<p class="lede">Every cell the benchmark needs weights for, and where '
               'the last one landed. PACE has no column of its own — it runs at eval '
               'time on the baseline arm, so <b>baseline</b> is also the PACE arm.</p>')
    out.append('</div><div style="display:flex;flex-direction:column;gap:10px;align-items:flex-end">')
    out.append(f'<div class="tally"><span class="n">{counts["trained"]}</span>'
               f'<span class="of">/ {total}</span>'
               f'<span class="cap">cells<br>trained</span></div>')
    legend = []
    for key, label in (("trained", "trained"), ("running", "training now"),
                       ("partial", "partial"), ("failed", "no checkpoint"),
                       ("missing", "not started"), ("nodata", "no dataset")):
        if counts[key]:
            legend.append(f'<span class="chip s-{key}"><i class="dot"></i>'
                          f'{counts[key]} {e(label)}</span>')
    out.append(f'<div class="legend">{"".join(legend)}</div>')
    out.append('</div></header>')

    # matrix
    out.append('<section><div class="scroll"><table class="matrix">')
    out.append('<thead><tr><th>Dataset / backbone</th>')
    for mkey, mlabel, mnote in METHODS:
        out.append(f'<th>{e(mlabel)}<span>{e(mnote)}</span></th>')
    out.append('</tr></thead><tbody>')
    prev_ds = None
    for ds, bb in cell_rows():
        cls = "grp-start" if ds.key != prev_ds else ""
        prev_ds = ds.key
        out.append(f'<tr class="{cls}"><th class="{ds.kind}">{e(ds.label)}'
                   f'<small>{e(BACKBONE_LABEL.get(bb, bb))}'
                   f'{" &middot; sim" if ds.kind == "sim" else ""}</small></th>')
        for mkey, _, _ in METHODS:
            run = cells.get((ds.key, bb, mkey))
            if run is None:
                state = "nodata" if ds.repo_id is None else "missing"
                out.append(f'<td><div class="cell empty s-{state}">'
                           f'<span class="k"><i class="dot"></i>'
                           f'{e(STATUS_TEXT[state])}</span>'
                           f'<span class="v">&mdash;</span></div></td>')
                continue
            steps = fmt_steps(run.steps)
            if run.budget and run.steps < run.budget:
                steps += f' <em>/ {fmt_steps(run.budget)}</em>'
            out.append(
                f'<td><a class="cell s-{run.status}" href="#{_cell_id(ds.key, bb, mkey)}">'
                f'<span class="k"><i class="dot"></i>{e(STATUS_TEXT[run.status])}</span>'
                f'<span class="v">{steps} steps</span></a></td>')
        out.append('</tr>')
    out.append('</tbody></table></div></section>')

    # detail, one card per dataset
    for ds in DATASETS:
        out.append('<section><div class="card">')
        head = f'<h3>{e(ds.label)}'
        bits = []
        if ds.repo_id:
            bits.append(f'<span class="note mono">{e(ds.repo_id)}</span>')
        if ds.note:
            bits.append(f'<span class="note">{e(ds.note)}</span>')
        out.append(head + "".join(bits) + '</h3>')
        # The recording itself, on the same terms as a checkpoint: the absolute
        # path, copyable, and said plainly when it is not there to be copied.
        if ds.root is None:
            out.append('<p class="dsroot none">No recording yet — this row is in '
                       'the grid, its dataset is not on disk.</p>')
        else:
            missing = not ds.root.exists()
            out.append(
                '<p class="dsroot">'
                f'<button class="copy" type="button" data-path="{e(str(ds.root))}" '
                f'title="{e(str(ds.root))}"><span class="ico">copy</span>'
                f'<span class="run">{e(str(ds.root))}</span></button>'
                + ('<span class="warn">not on this machine</span>' if missing else '')
                + '</p>')
        out.append('<div class="scroll"><table class="detail"><thead><tr>'
                   '<th>Arm</th><th>Status</th><th>Steps</th><th>Last written</th>'
                   '<th>Last checkpoint</th></tr></thead><tbody>')
        for bb in ds.backbones:
            for mkey, mlabel, _ in METHODS:
                cid = _cell_id(ds.key, bb, mkey)
                run = cells.get((ds.key, bb, mkey))
                arm = (f'<td class="arm">{e(BACKBONE_LABEL.get(bb, bb))}'
                       f'<small>{e(mlabel)}'
                       f'{" / PACE" if mkey == "baseline" else ""}</small></td>')
                if run is None:
                    state = "nodata" if ds.repo_id is None else "missing"
                    out.append(
                        f'<tr id="{cid}">{arm}'
                        f'<td><span class="chip s-{state}"><i class="dot"></i>'
                        f'{e(STATUS_TEXT[state])}</span></td>'
                        f'<td class="num">&mdash;</td><td class="when">&mdash;</td>'
                        f'<td class="path dim">&mdash;</td></tr>')
                    continue
                steps = f'{run.steps:,}'.replace(",", " ")
                if run.budget and run.steps < run.budget:
                    budget = f'{run.budget:,}'.replace(",", " ")
                    pct = min(100, round(100 * run.steps / run.budget))
                    steps += (f' <em>/ {budget}</em>'
                              f'<div class="bar"><i style="width:{pct}%"></i></div>')
                when = run.mtime or run.started
                when_html = (f'{e(fmt_when(when))}<small>{e(ago(when, now))}</small>'
                             if when else '&mdash;')
                if run.ckpt:
                    # The absolute path is what a shell or a --policy_path needs, so
                    # that is what the button copies; the cell shows the identifying
                    # part in full ink and the invariant remainder dimmed.
                    rel = run.ckpt.relative_to(TRAIN_ROOT)
                    path = (
                        f'<button class="copy" type="button" '
                        f'data-path="{e(str(run.ckpt))}" '
                        f'title="{e(str(run.ckpt))}">'
                        f'<span class="ico">copy</span>'
                        f'<span><span class="run">{e(run.dir.name)}</span>'
                        f'<span class="rest">/{e(str(rel.relative_to(run.dir.name)))}'
                        f'</span></span></button>')
                else:
                    path = ('<span class="dim">no checkpoint — '
                            f'run dir outputs/train/{e(run.name)}</span>')
                out.append(
                    f'<tr id="{cid}">{arm}'
                    f'<td><span class="chip s-{run.status}"><i class="dot"></i>'
                    f'{e(STATUS_TEXT[run.status])}</span></td>'
                    f'<td class="num">{steps}</td>'
                    f'<td class="when">{when_html}</td>'
                    f'<td class="path">{path}</td></tr>')
        out.append('</tbody></table></div></div></section>')

    # footnotes
    out.append('<div class="footnotes">')
    out.append('<p><b>How a cell is filled.</b> Each run is classified by its own '
               '<code>train_config.json</code> — dataset repo_id, policy type, '
               '<code>--method.type</code> — not by its directory name. A run that '
               'died before writing a checkpoint has no config, so its wandb argv '
               'stands in; that is what separates <i>no checkpoint</i> from '
               '<i>not started</i>.</p>')
    out.append('<p><b>Steps.</b> The number is the last checkpoint actually on disk, '
               'against the budget the run was launched with. A checkpoint existing '
               'is not proof the arm finished.</p>')
    if orphans:
        items = "".join(
            f'<li><code>outputs/train/{e(r.name)}</code> — {e(why)}'
            + (f', {r.steps:,} steps'.replace(",", " ") if r.steps else '')
            + '</li>' for r, why in orphans)
        out.append(f'<p><b>Runs outside the grid.</b></p><ul>{items}</ul>')
    out.append('<p><b>Paths.</b> Click a dataset root or a checkpoint to copy its '
               'absolute path — a hosted page cannot open a <code>file://</code> '
               'link at all, so copying is the only thing a button here can '
               'usefully do. <code>python checkpoint_status.py</code> prints the '
               'same paths in the terminal, where an editor session on this host '
               'can open them.</p>')
    out.append(f'<p>Generated by <code>checkpoint_status.py</code> on '
               f'{now:%Y-%m-%d at %H:%M}. Rerun it after any training run.</p>')
    out.append('</div></div>')
    out.append(COPY_JS)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--html", metavar="FILE", help="also write the status page here")
    ap.add_argument("--quiet", action="store_true", help="skip the terminal table")
    ap.add_argument("--no-color", action="store_true", help="plain text, no ANSI colour")
    ap.add_argument("--no-links", action="store_true",
                    help="print checkpoint paths as text, without OSC 8 hyperlinks")
    ap.add_argument("--paths", action="store_true",
                    help="print only the checkpoint paths, one per line, and exit")
    args = ap.parse_args()

    now = dt.datetime.now()
    cells, orphans = assign(scan())
    if args.paths:
        print_paths(cells, Term(False, False), bare=True)
        return
    if not args.quiet:
        term = detect_term(False if args.no_color else None,
                           False if args.no_links else None)
        print_table(cells, orphans, now, term)
    if args.html:
        Path(args.html).write_text(render_html(cells, orphans, now))
        print(f"  wrote {args.html}\n")


if __name__ == "__main__":
    main()
