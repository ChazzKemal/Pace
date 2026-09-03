"""An interactive report for one deploy run: what the arm was told, what it did.

The run folder already answers "was inference keeping up?" -- ``summary.json`` has five
stages of producer timing and per-frame sender slack, and the answer on this rig is a
comfortable yes. The question it cannot answer alone is the one that decides how fast
the policy may be driven: *was the arm keeping up*, and if not, what is stopping it.

That answer needs three things put on one clock:

* ``poses.csv``    -- where the arm actually was, sampled at 50 Hz on its own thread.
* ``commands.csv`` -- what it was told to be. ``cmd_xyz`` is the published target
  verbatim: ``_pre_compute_chunk_arrays`` sets ``target_xyz = actions[:, :3]``
  (``timing.py:414``), so the commanded and achieved columns are the same quantity in
  the same frame and may be compared directly.
* ``manifest.json`` -- the controller stiffness, which converts a tracking error in
  millimetres into a force in Newtons.

The commanded rows carry a *push* timestamp rather than a publish one, so their
execution timeline is reconstructed as ``cumsum(cycles x control_dt)`` -- the integer
controller-cycle dwell the sender actually used, not an approximation. (It agrees with
``dt_base / s_eff`` to the digit; both are kept only because the equality is worth
seeing rather than assuming.)

What the report is careful *not* to do is subtract the two blindly. A published target
is not yet a response: the arm moves toward it over the following control cycles, so a
raw difference is dominated by the servo's own lag and reads as failure when nothing is
wrong. The lag is recovered first, by scanning the offset that minimises RMS per axis,
and reported as a headline number in its own right -- it is the phase margin, and it
says more about achievable speed than the residual does.

Two signatures separate the possible diagnoses, which is why both are shown:

* **Lag with amplitude preserved** -- the arm goes everywhere it was told, just late.
  Pure delay; usually benign, and raising speed eats into the margin.
* **Amplitude loss** -- the arm never reaches the commanded extremes. It is being asked
  to move faster than its stiffness can drive it, and the fix is gain, not scheduling.
  The amplitude ratio needs no time alignment at all (it is a min/max over the run), so
  it is the most robust number here and the one to trust when the two disagree.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pace_bench.real import timeline as tlmod
from pace_bench.real.timeline import Timeline

logger = logging.getLogger(__name__)

AXES = ("x", "y", "z")
#: Cartesian stiffness parameter names, in the order :data:`AXES` expects.
KP_KEYS = ("task.k_pos_x", "task.k_pos_y", "task.k_pos_z")


@dataclass
class AxisFit:
    """How the achieved trajectory relates to the commanded one, for one axis."""

    axis: str
    lag_ms: float
    rms_aligned_mm: float
    rms_naive_mm: float
    amplitude_ratio: float
    correlation: float

    @property
    def lag_improvement(self) -> float:
        """Fraction of the naive error that was just delay. 0 when alignment buys
        nothing, which means the mismatch is shape, not timing."""
        if self.rms_naive_mm <= 0:
            return 0.0
        return 1.0 - self.rms_aligned_mm / self.rms_naive_mm


@dataclass
class Analysis:
    fits: list[AxisFit] = field(default_factory=list)
    realized_speed: float = 1.0
    configured_speed: float | None = None
    kp: list[float] | None = None
    #: True when the run scaled stiffness with speed, so `kp` is only the BASE.
    kp_scaled: bool = False
    kp_exp: float = 2.0
    wrench_mean_n: list[float] | None = None
    wrench_p99_n: list[float] | None = None
    duration_s: float = 0.0
    n_chunks: int = 0

    @property
    def median_lag_ms(self) -> float:
        return float(np.median([f.lag_ms for f in self.fits])) if self.fits else 0.0

    @property
    def worst_amplitude(self) -> AxisFit | None:
        return min(self.fits, key=lambda f: f.amplitude_ratio) if self.fits else None


def _read_csv(path: Path) -> dict[str, np.ndarray] | None:
    """Minimal CSV reader. numpy only -- pandas is not a dependency of this repo's
    robot environment, and the files are a few thousand rows of plain floats."""
    if not path.exists():
        return None
    with open(path) as f:
        header = f.readline().strip().split(",")
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return {name: data[:, i] for i, name in enumerate(header)}


def command_timeline(cmd: dict, control_dt: float) -> np.ndarray:
    """Absolute execution time for each commanded row.

    ``cycles`` is the integer number of controller cycles the sender dwelt on each
    target, so ``cycles x control_dt`` is the exact commanded duration -- the sender
    quantises to whole cycles, and reconstructing from ``dt_base / s_eff`` would
    reintroduce the rounding the sender already removed. Anchored on the first row's
    wall clock, which is the closest timestamp the replay log carries to the moment
    execution began.
    """
    dwell = cmd["cycles"] * control_dt
    return cmd["t_wall"][0] + np.concatenate([[0.0], np.cumsum(dwell)[:-1]])


def fit_axis(axis: str, grid: np.ndarray, t_cmd: np.ndarray, cmd_v: np.ndarray,
             t_ach: np.ndarray, ach_v: np.ndarray,
             lag_range_ms: tuple[int, int] = (-300, 1500),
             step_ms: int = 10) -> AxisFit:
    """Recover the delay that best explains one axis, and what is left after it.

    The scan is a plain RMS minimisation rather than a correlation peak: correlation
    is blind to amplitude, and amplitude is exactly the failure mode worth catching.
    """
    ci = np.interp(grid, t_cmd, cmd_v)
    lags = range(lag_range_ms[0], lag_range_ms[1] + 1, step_ms)
    scored = [(float(np.sqrt(np.mean(
        (np.interp(grid + L / 1000.0, t_ach, ach_v) - ci) ** 2))), L) for L in lags]
    rms, lag = min(scored)
    rms0 = {L: r for r, L in scored}[0]
    pi = np.interp(grid, t_ach, ach_v)
    cmd_span = float(ci.max() - ci.min())
    return AxisFit(
        axis=axis,
        lag_ms=float(lag),
        rms_aligned_mm=rms * 1000.0,
        rms_naive_mm=rms0 * 1000.0,
        amplitude_ratio=(float(pi.max() - pi.min()) / cmd_span) if cmd_span > 0 else 1.0,
        correlation=float(np.corrcoef(ci, pi)[0, 1]),
    )


def kp_series(a: Analysis, grid: np.ndarray, t_cmd: np.ndarray,
              cmd: dict) -> np.ndarray:
    """Stiffness at each grid sample, (N, 3).

    Constant while ``scale_kp`` is off. With it on, ``ReplayScaler.step_to`` pushes
    ``kp_base * s_eff**kp_exp`` at every integer-cycle segment boundary, so treating kp
    as fixed would understate the commanded force by up to the peak factor -- 4x at
    ``max_speed: 2.0`` with the default exponent.

    Caveat that cannot be resolved from the recording: those writes are
    *fire-and-forget* (``gains.py:527`` -- ``call_async`` on a cached client, never
    waited on), so this is the stiffness that was **requested**, not confirmed applied.
    A dropped request is superseded by the next segment, so the average tracks the
    schedule even when an individual write does not land.
    """
    base = np.asarray(a.kp, dtype=float)
    if not a.kp_scaled:
        return np.broadcast_to(base, (len(grid), 3))
    s_eff = np.interp(grid, t_cmd, cmd["s_eff"])
    return (s_eff ** a.kp_exp)[:, None] * base[None, :]


def analyse(poses: dict, cmd: dict, summary: dict, manifest: dict) -> Analysis:
    """Everything the report states, computed once."""
    control_dt = float(summary.get("control_dt_ms", 2.0)) / 1000.0
    fps = float(summary.get("fps_baseline", 20.0))
    t_cmd = command_timeline(cmd, control_dt)
    t_ach = poses["t_wall"]

    lo, hi = max(t_ach[0], t_cmd[0]), min(t_ach[-1], t_cmd[-1])
    grid = np.arange(lo, hi, 1.0 / fps / 2.0)

    a = Analysis(
        duration_s=float(summary.get("duration_s", 0.0)),
        n_chunks=int(summary.get("chunks_run", 0)),
        configured_speed=(summary.get("args") or {}).get("max_speed"),
    )
    dwell = cmd["cycles"] * control_dt
    a.realized_speed = float((1.0 / fps) / dwell.mean()) if dwell.mean() > 0 else 1.0

    for ax in AXES:
        a.fits.append(fit_axis(ax, grid, t_cmd, cmd[f"cmd_{ax}"],
                               t_ach, poses[f"ach_{ax}"]))

    gcfg = (manifest or {}).get("gains") or {}
    a.kp_scaled = bool(gcfg.get("scale_kp", False))
    a.kp_exp = float(gcfg.get("kp_exp", 2.0))

    gains = (manifest or {}).get("controller_gains_at_startup") or {}
    if all(k in gains and gains[k] is not None for k in KP_KEYS):
        a.kp = [float(gains[k]) for k in KP_KEYS]
        lag = a.median_lag_ms / 1000.0
        err = np.column_stack([
            np.interp(grid + lag, t_ach, poses[f"ach_{ax}"])
            - np.interp(grid, t_cmd, cmd[f"cmd_{ax}"]) for ax in AXES])
        f = np.abs(err) * kp_series(a, grid, t_cmd, cmd)
        a.wrench_mean_n = f.mean(axis=0).tolist()
        a.wrench_p99_n = np.percentile(f, 99, axis=0).tolist()
    return a


def verdict(a: Analysis, manifest: dict) -> list[tuple[str, str]]:
    """The report's headline: what this run measured, and what it does not settle.

    Deliberately measurement-led. An earlier version of this function blamed speed for
    amplitude loss; a 1.00x baseline on the same task then showed the *same* ratio
    (x: 0.62 both times), which that story cannot explain. The honest output names the
    number, states what a single run can and cannot attribute it to, and says which
    experiment would separate the candidates -- an opinionated wrong answer is worse
    than a plot, because it gets acted on.
    """
    out: list[tuple[str, str]] = []
    worst = a.worst_amplitude
    scale_kp = bool(((manifest or {}).get("gains") or {}).get("scale_kp", False))

    if a.configured_speed and a.realized_speed < 0.85 * float(a.configured_speed):
        out.append(("Realized speed is below the configured peak", (
            f"{a.realized_speed:.2f}x realized against max_speed "
            f"{float(a.configured_speed):.2f}. Expected for a modulating method -- "
            "peak speed is reached only where the path is straight -- but it means "
            "the headline speed is not what ran.")))

    if worst is not None and worst.amplitude_ratio < 0.9:
        msg = (f"The arm covers {worst.amplitude_ratio:.0%} of the commanded "
               f"{worst.axis}-range. This is a min/max over the whole run, so it needs "
               "no time alignment and is the most robust number here. What it does "
               "NOT tell you on its own is the cause: run the same task at 1.0x and "
               "compare. If the ratio improves at 1.0x, speed is the limit and the "
               "lever is stiffness. If it is unchanged, the deficit is present at "
               "nominal speed too and speed is not what is costing you reach.")
        if not scale_kp:
            msg += (" Note gains.scale_kp is OFF, so kp stayed at its 1x value for "
                    "the whole run regardless of how fast the method drove the arm.")
        out.append(("Amplitude loss -- the arm does not reach the commanded extremes",
                    msg))

    if a.median_lag_ms > 250:
        detail = f"Median best-fit delay {a.median_lag_ms:.0f} ms. "
        if a.kp:
            fn = np.sqrt(a.kp[0] / 15.0) / (2 * np.pi)
            detail += (f"With kp={a.kp[0]:.0f} N/m and a ~15 kg effective mass the "
                       f"impedance law's natural frequency is about {fn:.2f} Hz, so a "
                       "lag of this order is roughly what the gains predict rather "
                       "than a fault. Treat it as the phase margin: it caps how much "
                       "faster the arm can be driven before it chases a target it "
                       "never reaches.")
        out.append(("Servo lag", detail))

    if not out:
        out.append(("Tracking looks healthy", (
            "No large lag and no amplitude loss. If you want more speed, this run "
            "is not the thing stopping you.")))
    return out


# --------------------------------------------------------------------------- HTML

_CSS = """
:root{--bg:#fbfaf8;--fg:#1a1a1a;--mut:#6b6b6b;--line:#e3e0da;--card:#fff;
      --accent:#b5533a;--ok:#2f7d5d;--warn:#b5533a}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
  --bg:#16161a;--fg:#eceae6;--mut:#9a978f;--line:#2c2c33;--card:#1d1d22;
  --accent:#e0805f;--ok:#5cbf92;--warn:#e0805f}}
:root[data-theme=dark]{--bg:#16161a;--fg:#eceae6;--mut:#9a978f;--line:#2c2c33;
  --card:#1d1d22;--accent:#e0805f;--ok:#5cbf92;--warn:#e0805f}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;
 font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:32px 24px 72px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.02em}
h2{font-size:17px;margin:36px 0 12px;letter-spacing:-.01em}
.sub{color:var(--mut);font-size:13px;margin-bottom:26px;
 font-family:ui-monospace,SFMono-Regular,Menlo,monospace;word-break:break-all}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:13px 15px}
.card .k{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.07em}
.card .v{font-size:23px;font-weight:600;margin-top:3px;letter-spacing:-.02em}
.card .n{color:var(--mut);font-size:11.5px;margin-top:2px}
.v.warn{color:var(--warn)}.v.ok{color:var(--ok)}
.finding{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--accent);
 border-radius:0 8px 8px 0;padding:13px 17px;margin-bottom:10px}
.finding b{display:block;margin-bottom:4px;font-size:14.5px}
.finding span{color:var(--mut);font-size:13.5px}
table{border-collapse:collapse;width:100%;font-size:13.5px;
 font-variant-numeric:tabular-nums;background:var(--card);
 border:1px solid var(--line);border-radius:8px;overflow:hidden}
th,td{padding:8px 13px;text-align:right;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{background:color-mix(in srgb,var(--card) 80%,var(--fg) 6%);font-size:11px;
 text-transform:uppercase;letter-spacing:.06em;color:var(--mut);font-weight:600}
tr:last-child td{border-bottom:0}
.plot{background:var(--card);border:1px solid var(--line);border-radius:9px;
 padding:6px;margin-bottom:14px;overflow-x:auto}
.note{color:var(--mut);font-size:12.5px;margin:-4px 0 14px}
"""

_JS_THEME = """
function paceTheme(){
  const cs=getComputedStyle(document.documentElement);
  const fg=cs.getPropertyValue('--fg').trim(), mut=cs.getPropertyValue('--mut').trim(),
        line=cs.getPropertyValue('--line').trim();
  return {paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
    font:{color:fg,size:12,family:'-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif'},
    xaxis:{gridcolor:line,zerolinecolor:line,linecolor:line,tickfont:{color:mut}},
    yaxis:{gridcolor:line,zerolinecolor:line,linecolor:line,tickfont:{color:mut}},
    legend:{orientation:'h',y:1.13,x:0,font:{size:11.5}},
    margin:{l:58,r:16,t:26,b:40},hovermode:'x unified'};
}
function draw(){
  const t=paceTheme();
  PACE_PLOTS.forEach(p=>{
    const lay=Object.assign({},t,p.layout||{});
    lay.xaxis=Object.assign({},t.xaxis,(p.layout||{}).xaxis||{});
    lay.yaxis=Object.assign({},t.yaxis,(p.layout||{}).yaxis||{});
    Plotly.newPlot(p.id,p.data,lay,{displaylogo:false,responsive:true,
      modeBarButtonsToRemove:['lasso2d','select2d']});
  });
  // `xaxis.matches` only links axes within one figure; these are separate figures,
  // so x-zoom is mirrored by hand across every time-axis panel.
  const linked=PACE_PLOTS.filter(p=>((p.layout||{}).xaxis||{}).matches==='x').map(p=>p.id);
  let busy=false;
  linked.forEach(id=>document.getElementById(id).on('plotly_relayout',ev=>{
    if(busy)return;const upd={};
    if('xaxis.range[0]' in ev){upd['xaxis.range']=[ev['xaxis.range[0]'],ev['xaxis.range[1]']];}
    else if(ev['xaxis.autorange']){upd['xaxis.autorange']=true;}else return;
    busy=true;linked.filter(o=>o!==id).forEach(o=>Plotly.relayout(o,upd));busy=false;}));
}
if(window.matchMedia) matchMedia('(prefers-color-scheme:dark)')
  .addEventListener('change',()=>{PACE_PLOTS.forEach(p=>Plotly.purge(p.id));draw();});
draw();
"""


def _card(k: str, v: str, note: str = "", cls: str = "") -> str:
    n = f'<div class="n">{note}</div>' if note else ""
    return (f'<div class="card"><div class="k">{k}</div>'
            f'<div class="v {cls}">{v}</div>{n}</div>')


def chunk_timing_plot(tl: Timeline, t0: float) -> dict:
    """Gantt of execution against inference, one row per chunk.

    Inference is drawn on the row of the chunk it *produced*, at the time it actually
    ran -- which is over the previous chunk's bar. That overlap is the point of the
    figure: it shows inference finishing well before the queue in front of the new
    chunk has drained, and the new chunk waiting on that queue, not on the model.
    """
    ys, ex_x, ex_b, ex_t = [], [], [], []
    br_x, br_b = [], []
    in_y, in_x, in_b, in_t = [], [], [], []
    ob_x, ob_y = [], []
    for c in tl.chunks:
        ys.append(c.idx)
        ex_b.append(c.t_first - t0)
        ex_x.append(max(c.exec_s, 0.01))
        ex_t.append(f"chunk {c.idx}: {c.n_rows} rows, {c.n_published} published"
                    + (f", {c.n_late} late" if c.n_late else "")
                    + (f"<br>obs→first row {c.latency_first_ms:.0f} ms"
                       if c.latency_first_ms is not None else ""))
        if c.n_bridge and c.t_first_policy is not None:
            br_b.append(c.t_first - t0)
            br_x.append(max(c.t_first_policy - c.t_first, 0.01))
        if c.t_req is not None and c.t_ret is not None:
            in_y.append(c.idx)
            in_b.append(c.t_req - t0)
            in_x.append(max(c.t_ret - c.t_req, 0.008))
            in_t.append(f"inference for chunk {c.idx}: {c.inference_ms:.0f} ms"
                        + (f", {c.q_logged} queued" if c.q_logged is not None else ""))
        if c.t_obs is not None:
            ob_x.append(c.t_obs - t0)
            ob_y.append(c.idx)
    bar = {"type": "bar", "orientation": "h", "hoverinfo": "text"}
    data = [
        {**bar, "y": ys, "x": ex_x, "base": ex_b, "text": ex_t, "name": "executing",
         "marker": {"color": "rgba(42,120,214,0.45)"}, "width": 0.62},
        {**bar, "y": [c.idx for c in tl.chunks if c.n_bridge and c.t_first_policy is not None],
         "x": br_x, "base": br_b, "name": "seam bridge rows", "hoverinfo": "name",
         "marker": {"color": "rgba(237,161,0,0.85)"}, "width": 0.62},
        {**bar, "y": in_y, "x": in_x, "base": in_b, "text": in_t, "name": "inference",
         "marker": {"color": "#b5533a"}, "width": 0.62},
        {"type": "scatter", "mode": "markers", "x": ob_x, "y": ob_y, "name": "observation",
         "marker": {"symbol": "line-ns-open", "size": 13, "color": "#b5533a",
                    "line": {"width": 1.6}}, "hoverinfo": "x+name"},
    ]
    n = max(len(tl.chunks), 1)
    return {"id": "chunks", "data": data,
            "layout": {"height": 90 + 24 * n, "barmode": "overlay",
                       "xaxis": {"title": "s", "matches": "x"},
                       "yaxis": {"title": "chunk", "autorange": "reversed",
                                 "dtick": 1 if n <= 20 else 5}}}


def timing_shapes(tl: Timeline, t0: float) -> list[dict]:
    """Inference windows and chunk seams, to overlay on any time-axis panel."""
    shapes: list[dict] = []
    for c in tl.chunks:
        if c.t_req is not None and c.t_ret is not None:
            shapes.append({"type": "rect", "xref": "x", "yref": "paper",
                           "x0": c.t_req - t0, "x1": c.t_ret - t0, "y0": 0, "y1": 1,
                           "fillcolor": "rgba(181,83,58,0.22)", "line": {"width": 0},
                           "layer": "below"})
        if c.idx > 0:
            shapes.append({"type": "line", "xref": "x", "yref": "paper",
                           "x0": c.t_first - t0, "x1": c.t_first - t0, "y0": 0, "y1": 1,
                           "line": {"color": "rgba(42,120,214,0.5)", "width": 1,
                                    "dash": "dot"}})
    return shapes


def timing_table(tl: Timeline) -> str:
    def _ms(v):
        return "–" if v is None else f"{v:.0f}"
    rows = []
    for c in tl.chunks:
        q = ("–" if c.q_logged is None else str(c.q_logged)) + (
            f" / {c.q_measured}" if c.q_measured is not None else "")
        rows.append(
            f"<tr><td>{c.idx}</td><td>{_ms(c.latency_first_ms)}</td>"
            f"<td>{_ms(c.latency_policy_ms)}</td><td>{_ms(c.inference_ms)}</td>"
            f"<td>{q}</td><td>{c.n_bridge}</td>"
            f"<td>{c.n_published}/{c.n_rows}</td><td>{c.n_late}</td></tr>")
    return ("<table><tr><th>chunk</th><th>obs → first row (ms)</th>"
            "<th>obs → first policy row (ms)</th><th>inference (ms)</th>"
            "<th>queued at request (logged / measured)</th><th>bridge rows</th>"
            "<th>published</th><th>late</th></tr>" + "".join(rows) + "</table>")


def render_html(run_dir: Path, poses: dict, cmd: dict, summary: dict,
                manifest: dict, a: Analysis, chunks: dict | None,
                tl: Timeline | None = None) -> str:
    control_dt = float(summary.get("control_dt_ms", 2.0)) / 1000.0
    t_cmd = command_timeline(cmd, control_dt) - poses["t_wall"][0]
    t_ach = poses["t_wall"] - poses["t_wall"][0]
    plots: list[dict] = []
    shapes = timing_shapes(tl, poses["t_wall"][0]) if tl is not None else []

    # Tracking, one row per axis, x-axes matched so zooming one zooms all.
    for i, ax in enumerate(AXES):
        fit = a.fits[i]
        plots.append({
            "id": f"trk_{ax}",
            "data": [
                {"x": t_cmd.tolist(), "y": cmd[f"cmd_{ax}"].tolist(), "name": "commanded",
                 "type": "scatter", "mode": "lines", "line": {"width": 1.6}},
                {"x": t_ach.tolist(), "y": poses[f"ach_{ax}"].tolist(), "name": "achieved",
                 "type": "scatter", "mode": "lines", "line": {"width": 1.6}},
                {"x": (t_ach - fit.lag_ms / 1000.0).tolist(),
                 "y": poses[f"ach_{ax}"].tolist(),
                 "name": f"achieved, shifted −{fit.lag_ms:.0f} ms",
                 "type": "scatter", "mode": "lines",
                 "line": {"width": 1.1, "dash": "dot"}, "visible": "legendonly"},
            ],
            "layout": {"height": 210, "xaxis": {"title": "s", "matches": "x"},
                       "yaxis": {"title": f"{ax} (m)"}, "shapes": shapes},
        })

    # Commanded speed over time.
    plots.append({
        "id": "speed",
        "data": [{"x": t_cmd.tolist(), "y": cmd["s_eff"].tolist(), "name": "s_eff",
                  "type": "scatter", "mode": "lines", "line": {"width": 1.4}}],
        "layout": {"height": 200, "xaxis": {"title": "s", "matches": "x"},
                   "yaxis": {"title": "speed factor"}, "shapes": shapes},
    })

    # When each chunk was inferred, against what was executing at the time.
    if tl is not None and tl.chunks:
        plots.append(chunk_timing_plot(tl, poses["t_wall"][0]))

    # Estimated wrench.
    if a.kp:
        lag = a.median_lag_ms / 1000.0
        grid = np.arange(max(t_ach[0], t_cmd[0]), min(t_ach[-1], t_cmd[-1]), 0.02)
        traces = []
        kps = kp_series(a, grid + poses["t_wall"][0], t_cmd + poses["t_wall"][0], cmd)
        for j, ax in enumerate(AXES):
            e = (np.interp(grid + lag, t_ach, poses[f"ach_{ax}"])
                 - np.interp(grid, t_cmd, cmd[f"cmd_{ax}"]))
            traces.append({"x": grid.tolist(), "y": (np.abs(e) * kps[:, j]).tolist(),
                           "name": f"|F{ax}|", "type": "scatter", "mode": "lines",
                           "line": {"width": 1.2}})
        plots.append({"id": "wrench", "data": traces,
                      "layout": {"height": 210, "xaxis": {"title": "s", "matches": "x"},
                                 "yaxis": {"title": "N"}}})

    # Inference against the budget.
    if chunks is not None and "synth_ms" in chunks:
        budget = float(summary.get("overlap_budget_ms", 0.0))
        n = len(np.atleast_1d(chunks["synth_ms"]))
        plots.append({
            "id": "infer",
            "data": [
                {"x": list(range(n)), "y": np.atleast_1d(chunks["synth_ms"]).tolist(),
                 "name": "inference", "type": "bar"},
                {"x": [0, n - 1], "y": [budget, budget], "name": f"budget {budget:.0f} ms",
                 "type": "scatter", "mode": "lines", "line": {"dash": "dash", "width": 1.4}},
            ],
            "layout": {"height": 210, "xaxis": {"title": "chunk"},
                       "yaxis": {"title": "ms"}},
        })

    cards = [
        _card("realized speed", f"{a.realized_speed:.2f}×",
              f"configured max {float(a.configured_speed):.2f}×" if a.configured_speed else "",
              "warn" if (a.configured_speed and a.realized_speed
                         < 0.85 * float(a.configured_speed)) else ""),
        _card("median servo lag", f"{a.median_lag_ms:.0f} ms",
              "best-fit delay, per axis below",
              "warn" if a.median_lag_ms > 250 else "ok"),
    ]
    if a.worst_amplitude:
        w = a.worst_amplitude
        cards.append(_card("worst amplitude", f"{w.amplitude_ratio:.0%}",
                           f"{w.axis}-axis reach vs commanded",
                           "warn" if w.amplitude_ratio < 0.9 else "ok"))
    if a.wrench_p99_n:
        cards.append(_card("peak est. force", f"{max(a.wrench_p99_n):.0f} N",
                           (f"p99, kp {a.kp[0]:.0f}\u2192{a.kp[0] * a.realized_speed ** a.kp_exp:.0f} N/m (scaled)" if a.kp_scaled else f"p99, kp={a.kp[0]:.0f} N/m")))
    if tl is not None and tl.median_latency_first_ms is not None:
        lat = tl.median_latency_first_ms
        latp = tl.median_latency_policy_ms
        cards.append(_card("obs → execution", f"{lat:.0f} ms",
                           f"{lat / tl.median_frame_ms:.1f} frames"
                           + (f"; first policy row {latp:.0f} ms" if latp else ""),
                           "warn" if lat / tl.median_frame_ms > 4 else "ok"))
    cards.append(_card("duration", f"{a.duration_s:.0f} s", f"{a.n_chunks} chunks"))

    rows = "".join(
        f"<tr><td>{f.axis}</td><td>{f.lag_ms:.0f}</td><td>{f.rms_naive_mm:.1f}</td>"
        f"<td>{f.rms_aligned_mm:.1f}</td><td>{f.lag_improvement:.0%}</td>"
        f"<td>{f.amplitude_ratio:.2f}</td><td>{f.correlation:.3f}</td></tr>"
        for f in a.fits)

    found = verdict(a, manifest) + (tlmod.describe(tl) if tl is not None else [])
    findings = "".join(f'<div class="finding"><b>{t}</b><span>{d}</span></div>'
                       for t, d in found)
    timing_html = ""
    if tl is not None and tl.chunks:
        src = ("measured by ChunkClock" if tl.inference_source == "measured"
               else "reconstructed from trace.npz and chunks.csv")
        timing_html = f"""
<h2>Chunk timing</h2>
<div class="note">Each row is one chunk. The blue bar is when its rows were published;
the orange head is the seam bridge (rows the blend wrote, not the policy); the red
bar is the inference that produced it, drawn where it actually ran — over the
previous chunk. The tick is the observation it was computed from. The same red
windows are shaded on the tracking panels above, and the dotted lines there are chunk
seams. Inference times {src}; publish times are the anchored deadlines corrected by
the sender's recorded slack.</div>
<div class="plot" id="chunks"></div>
{timing_table(tl)}
<div class="note">"queued at request" is the producer's own count (q_before_inf) and,
after the slash, the number of the previous chunk's rows that were published after
the request was issued — the same thing measured from the timeline; they differ by
the row the sender was already holding.</div>"""

    dep = (manifest or {}).get("deployed_policy_path", "?")
    method = (manifest or {}).get("method", "?")
    task = (manifest or {}).get("task", "?")

    return f"""<title>Deploy Run {run_dir.name}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.3/plotly.min.js"></script>
<style>{_CSS}</style>
<div class="wrap">
<h1>{task} · {method} · {a.realized_speed:.2f}×</h1>
<div class="sub">{run_dir.name}<br>{dep}</div>

<div class="cards">{"".join(cards)}</div>

<h2>What is limiting this run</h2>
{findings}

<h2>Commanded vs achieved</h2>
<div class="note">Zoom any panel — all three x-axes are linked. Toggle the dotted
trace in the legend to see the achieved path shifted back by its best-fit lag: if it
lands on the commanded line, the mismatch was pure delay; if a gap remains, the arm
is not reaching where it was told.</div>
<div class="plot" id="trk_x"></div>
<div class="plot" id="trk_y"></div>
<div class="plot" id="trk_z"></div>

<h2>Per-axis fit</h2>
<table><tr><th>axis</th><th>lag (ms)</th><th>RMS naive (mm)</th>
<th>RMS aligned (mm)</th><th>explained by delay</th>
<th>amplitude ratio</th><th>corr</th></tr>{rows}</table>
<div class="note">Amplitude ratio needs no time alignment — it is a min/max over the
whole run — so trust it over the RMS columns when they disagree.</div>

<h2>Commanded speed</h2>
<div class="plot" id="speed"></div>

{'<h2>Estimated wrench</h2><div class="note">kp × aligned tracking error. Joint efforts are not readable on this rig (crisp_py drops msg.effort), but the controller is an impedance law, so this is the force it was commanding. With scale_kp ON, kp varies as kp_base x s_eff**kp_exp and is tracked per sample here \u2014 but those writes are fire-and-forget, so this is the stiffness requested, not confirmed applied.</div><div class="plot" id="wrench"></div>' if a.kp else ''}

{timing_html}

{'<h2>Inference against budget</h2><div class="plot" id="infer"></div>' if chunks is not None and "synth_ms" in chunks else ''}
</div>
<script>const PACE_PLOTS={json.dumps(plots)};{_JS_THEME}</script>"""


def build(run_dir: Path, out: Path | None = None) -> Path:
    """Read a deploy run folder and write its report beside it."""
    run_dir = Path(run_dir)
    poses = _read_csv(run_dir / "poses.csv")
    cmd = _read_csv(run_dir / "commands.csv")
    if poses is None or cmd is None:
        raise FileNotFoundError(
            f"{run_dir} has no poses.csv/commands.csv -- it predates the recorder. "
            "Runs from before that only carry timing, so tracking cannot be shown.")
    summary = json.loads((run_dir / "summary.json").read_text())
    mpath = run_dir / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    chunks = _read_csv(run_dir / "chunks.csv")

    a = analyse(poses, cmd, summary, manifest)
    # The timeline is a second view of the same files; a run whose trace is missing
    # or malformed still gets the tracking report, with the timing section left out.
    tl = None
    try:
        tl = tlmod.reconstruct(tlmod.load_run(run_dir))
    except Exception:
        logger.exception("chunk timeline could not be reconstructed; section omitted")
    out = Path(out) if out else run_dir / "report.html"
    out.write_text(render_html(run_dir, poses, cmd, summary, manifest, a, chunks, tl))
    logger.info("report written to %s", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", nargs="?", help="deploy_runs/<ts>; default: newest")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.run_dir:
        run = Path(args.run_dir)
    else:
        root = Path.home() / ".cache/huggingface/lerobot/deploy_runs"
        cands = [d for d in root.iterdir() if (d / "poses.csv").exists()]
        if not cands:
            raise SystemExit(f"no run under {root} has poses.csv yet")
        run = max(cands, key=lambda d: d.stat().st_mtime)
        logger.info("newest run with tracking data: %s", run.name)
    print(build(run, args.out))


if __name__ == "__main__":
    main()
