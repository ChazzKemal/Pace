"""Pick a run configuration interactively when the command line did not name one.

``run_real.py --config_path real/configs/pace_fast.yaml`` is the contract and stays
the contract. This exists for the other case: an operator at the rig who knows they
want "the fast PACE one" and would otherwise have to remember the filename, or open
four YAMLs to recall which is which.

Three rules follow from what this launches. Every one of them is about a real arm:

* **Interactive only.** If stdin or stdout is not a terminal the argv passes straight
  through untouched. A menu that blocks a scripted or ``nohup``-ed run would be a hang
  at bring-up, which is worse than no menu at all.
* **Cancelling means nothing happened.** ``q`` and ``Esc`` exit without launching.
  There is no fall-through to a default configuration, because "I pressed escape" and
  "run the baseline" are not the same intent.
* **Selecting is not launching.** A choice opens a confirmation showing what will
  actually run, and needs a second, different key. One stray arrow key should not
  become robot motion.

Free of crisp_gym, torch and lerobot -- it reads YAML and draws a curses screen, so it
stays importable and testable on a machine with no robot stack.
"""

from __future__ import annotations

import curses
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from pace_bench.real.configs import REPO_ROOT, load_tasks, resolve_config

logger = logging.getLogger(__name__)

#: Where run configurations live. Anchored to the repo, not the cwd -- see REPO_ROOT.
CONFIG_DIR = REPO_ROOT / "real/configs"

#: The (task, method) -> checkpoint registry. Matches `RealEvalConfig.tasks_file`.
TASKS_FILE = REPO_ROOT / "real/configs/tasks.yaml"

#: Opt out explicitly, for an interactive terminal that still wants the old behaviour
#: (draccus defaults, no menu). Stripped from argv before draccus sees it.
NO_PICKER_FLAG = "--no-picker"


@dataclass(frozen=True)
class ConfigChoice:
    """One selectable run configuration, already resolved through its ``_include``."""

    path: Path
    name: str
    method: str
    #: The one knob that distinguishes this config from its siblings, pre-formatted.
    detail: str
    #: First line of the file's leading comment block -- every config has one.
    summary: str
    #: Non-empty when the config is known to refuse at build time, saying why. Shown
    #: rather than hidden: a config that cannot run is exactly what an operator needs
    #: to be told about, and hiding it would look like the file had gone missing.
    blocked: str
    #: An explicit ``policy_path`` from the config. When set it wins over the registry,
    #: exactly as ``apply_task`` treats it, so the task step is skipped entirely.
    policy_path: str = ""

    @property
    def needs_task(self) -> bool:
        return not self.policy_path


def _summary(path: Path) -> str:
    """The file's first comment line, which every run config already carries."""
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
        if line.strip():
            break
    return ""


def _detail(method: dict) -> str:
    """The knob worth showing next to a method name."""
    kind = method.get("type", "none")
    if kind == "pace":
        return f"{method.get('max_speed', 1.0):g}x"
    if kind == "bspline":
        return f"{method.get('num_actions', '?')} samples"
    if kind == "demospeedup":
        return f"low_v {method.get('low_v', '?')}"
    return "1x"


def _blocked(cfg: dict) -> str:
    """Why this config will refuse at ``deploy_steps``, or "" if it will not.

    Mirrors the ``GripperCompensationUndecided`` guard rather than reading the file's
    prose, so a config that stops being blocked stops being flagged without anyone
    having to remember to edit a comment.
    """
    method, gripper = cfg.get("method", {}), cfg.get("gripper", {})
    if (method.get("type") == "bspline"
            and gripper.get("slowdown_frames")
            and not gripper.get("bspline_low_v")):
        return "needs --gripper.bspline_low_v, or slowdown_frames=0"
    return ""


def discover(config_dir: Path = CONFIG_DIR) -> list[ConfigChoice]:
    """Every runnable configuration in ``config_dir``, by name.

    A run configuration is one with a top-level ``method:`` block. That is what
    separates them from ``deploy_defaults.yaml``, which is the include base, and
    ``tasks.yaml``, which is the checkpoint registry -- neither is launchable, and
    neither should appear in a menu. Testing the key rather than the filename means a
    new support file does not have to be added to an exclusion list to stay out.
    """
    out = []
    for path in sorted(config_dir.glob("*.yaml")):
        try:
            cfg = resolve_config(path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            # A config that will not load cannot be offered, but a menu is the wrong
            # place to report it: the operator wants the list, and the run that names
            # this file explicitly will fail with the real error.
            logger.debug("skipping unreadable config %s: %s", path, exc)
            continue
        if "method" not in cfg:
            continue
        out.append(ConfigChoice(
            path=path,
            name=path.stem,
            method=cfg["method"].get("type", "none"),
            detail=_detail(cfg["method"]),
            summary=_summary(path),
            blocked=_blocked(cfg),
            policy_path=str(cfg.get("policy_path") or ""),
        ))
    return out


@dataclass(frozen=True)
class TaskChoice:
    """One task offered for the method the chosen config runs."""

    name: str
    #: Absolute checkpoint path, or None when there is nothing to run.
    path: Path | None
    #: Non-empty when this task cannot run under this method, saying why. Listed
    #: anyway: "stackcups has no bspline checkpoint" is the answer to the operator's
    #: question, and an absent row would read as a missing task rather than an
    #: untrained pair.
    blocked: str


def discover_tasks(method: str, registry: Path = TASKS_FILE) -> list[TaskChoice]:
    """Every task in the registry, marked for whether it can run under ``method``.

    Mirrors :func:`pace_bench.real.configs.resolve_policy_path` -- both the "not
    trained" case and the "trained but not on this machine" case, which are different
    problems with different fixes and so must not read the same. Nothing here falls
    back to another arm's checkpoint; that is the registry's whole point.
    """
    try:
        tasks = load_tasks(registry)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        logger.debug("no task registry at %s: %s", registry, exc)
        return []

    out = []
    for name, entry in tasks.items():
        entry = dict(entry)
        root = entry.pop("root", None)
        raw = entry.get(method)
        if not raw:
            trained = sorted(m for m, v in entry.items() if v)
            out.append(TaskChoice(name, None,
                                  f"no {method} checkpoint (trained: {', '.join(trained) or 'none'})"))
            continue
        path = Path(raw)
        if not path.is_absolute():
            if not root:
                out.append(TaskChoice(name, None, "relative path with no `root` to resolve it"))
                continue
            path = Path(root) / path
        blocked = "" if path.exists() else "checkpoint is not on this machine"
        out.append(TaskChoice(name, path, blocked))
    return out


def _init_colors() -> dict[str, int]:
    """Colour pairs, degrading to plain attributes on a terminal without colour."""
    if not curses.has_colors():
        return {k: 0 for k in ("head", "dim", "warn", "ok", "key")}
    curses.start_color()
    curses.use_default_colors()
    for i, fg in enumerate((curses.COLOR_CYAN, curses.COLOR_WHITE, curses.COLOR_YELLOW,
                            curses.COLOR_GREEN, curses.COLOR_MAGENTA), start=1):
        curses.init_pair(i, fg, -1)
    return {"head": curses.color_pair(1) | curses.A_BOLD, "dim": curses.A_DIM,
            "warn": curses.color_pair(3), "ok": curses.color_pair(4),
            "key": curses.color_pair(5)}


def _draw(scr, choices: list[ConfigChoice], cursor: int, dry_run: bool, col) -> None:
    scr.erase()
    h, w = scr.getmaxyx()

    def put(y, x, text, attr=0):
        if 0 <= y < h and x < w:
            scr.addnstr(y, x, text, max(0, w - x - 1), attr)

    put(0, 0, "  pace_bench — real robot deploy", col["head"])
    put(1, 0, "  no --config_path given; choose a run configuration", col["dim"])

    name_w = max((len(c.name) for c in choices), default=10)
    for i, c in enumerate(choices):
        y = 3 + i
        selected = i == cursor
        marker = "▸" if selected else " "
        row = f" {marker} {c.name:<{name_w}}  {c.method:<12} {c.detail:<12} "
        put(y, 0, row, curses.A_REVERSE if selected else 0)
        tail = f"⚠ {c.blocked}" if c.blocked else c.summary
        put(y, len(row), tail, col["warn"] if c.blocked else col["dim"])

    box = 3 + len(choices) + 1
    on = cursor == len(choices)
    put(box, 0, f" {'▸' if on else ' '} [{'x' if dry_run else ' '}] dry run",
        curses.A_REVERSE if on else (col["ok"] if dry_run else 0))
    put(box, 24, "no motion; the loop runs and nothing is published", col["dim"])

    put(box + 2, 0, "  ↑↓ move   space toggle   enter select   q cancel", col["key"])
    scr.refresh()


def _task_screen(scr, choice: ConfigChoice, tasks: list[TaskChoice], col) -> TaskChoice | None:
    """Choose what to deploy the config on. None to go back to the config list."""
    cursor = 0
    while True:
        scr.erase()
        h, w = scr.getmaxyx()

        # h/w are rebound every iteration (the terminal can be resized mid-screen),
        # so bind them as defaults rather than closing over the loop variables.
        def put(y, x, text, attr=0, *, h=h, w=w):
            if 0 <= y < h and x < w:
                scr.addnstr(y, x, text, max(0, w - x - 1), attr)

        put(0, 0, f"  {choice.name} — which task?", col["head"])
        put(1, 0, f"  checkpoints trained for method '{choice.method}'", col["dim"])

        if not tasks:
            put(3, 2, "no task registry, or it lists no tasks.", col["warn"])
            put(5, 2, "  esc back    q cancel", col["key"])
            scr.refresh()
            if scr.getch() in (27, ord("q"), ord("Q")):
                return None
            continue

        name_w = max(len(t.name) for t in tasks)
        for i, t in enumerate(tasks):
            selected = i == cursor
            row = f" {'▸' if selected else ' '} {t.name:<{name_w}}  "
            put(3 + i, 0, row, curses.A_REVERSE if selected else 0)
            tail = f"⚠ {t.blocked}" if t.blocked else str(t.path)
            put(3 + i, len(row), tail, col["warn"] if t.blocked else col["dim"])

        put(3 + len(tasks) + 1, 0, "  ↑↓ move   enter select   esc back   q cancel", col["key"])
        scr.refresh()

        k = scr.getch()
        if k in (curses.KEY_UP, ord("k")):
            cursor = (cursor - 1) % len(tasks)
        elif k in (curses.KEY_DOWN, ord("j")):
            cursor = (cursor + 1) % len(tasks)
        elif k in (curses.KEY_ENTER, 10, 13):
            return tasks[cursor]
        elif k in (27, ord("q"), ord("Q")):
            return None


def _confirm(scr, choice: ConfigChoice, task, dry_run: bool, col) -> bool:
    """Show what will actually run. True to launch, False to go back."""
    scr.erase()
    h, w = scr.getmaxyx()

    def put(y, x, text, attr=0):
        if 0 <= y < h and x < w:
            scr.addnstr(y, x, text, max(0, w - x - 1), attr)

    cfg = resolve_config(choice.path)
    put(0, 0, "  about to launch", col["head"])
    rows = [
        ("config", str(choice.path)),
        ("method", f"{choice.method}  ({choice.detail})"),
        ("task", task.name if task else "(config names the checkpoint)"),
        ("checkpoint", choice.policy_path or (str(task.path) if task and task.path else "NONE")),
        ("fps", str(cfg.get("fps", ""))),
        ("n_action_steps", str(cfg.get("n_action_steps", ""))),
        ("dry run", "YES — nothing is published" if dry_run else "no — THE ARM WILL MOVE"),
    ]
    for i, (k, v) in enumerate(rows):
        put(2 + i, 2, f"{k:<16}", col["dim"])
        put(2 + i, 20, v, col["ok"] if (k == "dry run" and dry_run) else 0)

    y = 2 + len(rows) + 1
    if task is not None and task.blocked:
        put(y, 2, f"⚠ this task cannot run: {task.blocked}", col["warn"])
        y += 2
    if choice.blocked:
        put(y, 2, f"⚠ this config refuses to run: {choice.blocked}", col["warn"])
        y += 2
    if not dry_run:
        put(y, 2, "⚠ not a dry run.", col["warn"])
        y += 2
    put(y, 2, "  enter launch    esc back    q cancel", col["key"])
    scr.refresh()

    while True:
        k = scr.getch()
        if k in (curses.KEY_ENTER, 10, 13):
            return True
        if k in (27, ord("q"), ord("Q")):
            return False


def _screen(scr, choices: list[ConfigChoice], registry: Path, ask_task: bool):
    curses.curs_set(0)
    col = _init_colors()
    cursor, dry_run = 0, False
    last = len(choices)          # the dry-run checkbox sits one past the last config

    while True:
        _draw(scr, choices, cursor, dry_run, col)
        k = scr.getch()
        if k in (curses.KEY_UP, ord("k")):
            cursor = (cursor - 1) % (last + 1)
        elif k in (curses.KEY_DOWN, ord("j")):
            cursor = (cursor + 1) % (last + 1)
        elif k == ord(" "):
            if cursor == last:
                dry_run = not dry_run
        elif k in (curses.KEY_ENTER, 10, 13):
            if cursor == last:
                dry_run = not dry_run
                continue
            choice = choices[cursor]
            task = None
            if choice.needs_task and ask_task:
                # Escape here returns to the config list rather than cancelling: a
                # wrong config is the likeliest reason the task list looks wrong.
                task = _task_screen(scr, choice, discover_tasks(choice.method, registry), col)
                if task is None:
                    continue
            if _confirm(scr, choice, task, dry_run, col):
                return choice.path, task, dry_run
        elif k in (27, ord("q"), ord("Q")):
            return None


def pick(choices: list[ConfigChoice], registry: Path = TASKS_FILE, ask_task: bool = True):
    """Run the chooser. ``(config_path, task_or_None, dry_run)``, or None if cancelled."""
    return curses.wrapper(_screen, choices, registry, ask_task)


def maybe_pick_config(argv: list[str], config_dir: Path = CONFIG_DIR) -> list[str] | None:
    """``argv`` with a chosen ``--config_path`` appended, or ``None`` if cancelled.

    Returns ``argv`` untouched whenever the picker should not appear at all: a config
    was already named, the opt-out flag was passed, this is not a terminal, or there is
    nothing to choose from.
    """
    if NO_PICKER_FLAG in argv:
        return [a for a in argv if a != NO_PICKER_FLAG]
    if any(a == "--config_path" or a.startswith("--config_path=") for a in argv):
        return argv
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return argv

    choices = discover(config_dir)
    if not choices:
        return argv

    # An explicit --task already answered the second question; asking again would
    # append a contradicting flag.
    ask_task = not any(a == "--task" or a.startswith("--task=") for a in argv)

    try:
        picked = pick(choices, TASKS_FILE, ask_task)
    except curses.error:
        # A terminal curses cannot drive (no TERM, a pipe that lied about isatty).
        # Falling back to the old behaviour would launch something unchosen, so don't.
        print("could not open the config picker; pass --config_path explicitly",
              file=sys.stderr)
        return None
    if picked is None:
        return None

    path, task, dry_run = picked
    out = [*argv, f"--config_path={path}"]
    if task is not None:
        out.append(f"--task={task.name}")
    if dry_run:
        out.append("--dry_run=true")
    return out
