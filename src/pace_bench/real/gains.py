"""Damping that rises with speed, on top of crisp_gym's stiffness scaler.

On this rig every cartesian kd reads 0, which the C++ controller treats as *auto*:
``d = 2 * sqrt(k)`` recomputed every cycle (``cartesian_controller.cpp:441``), and
``ReplayScaler`` therefore never pushes a kd for those axes -- scaling kp by
``s**kp_exp`` makes the controller's own formula scale d by ``s**(kp_exp/2)``.

That formula is critical damping for a unit mass. With the arm's ~15 kg effective
mass the damping ratio is nearer 0.25, and the 17:39 run on 2026-09-02 showed the
matching symptom: the achieved z ran 35 mm past the commanded maximum after a fast
seam. So this pushes an explicit kd of ``kd_ratio * 2 * sqrt(kp)`` instead, and sets
the scaler's ``kd_exp`` to ``kp_exp / 2`` so the pushed value keeps tracking
``sqrt(kp_eff)`` exactly as auto would -- the ratio is the only change.

Done by editing the scaler's cached originals rather than patching crisp_gym: the
scaler pushes ``_original_kd * s**kd_exp`` for every non-auto axis at each segment
boundary (``gains.py:516-523``), so marking the axes non-auto with a synthetic base
is enough. ``restore()`` pushes the cached originals back, so the true ones are
kept aside and put back just before it runs -- the controller ends the run on auto
damping, as it started.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


def raise_damping(scaler, *, kd_ratio: float, kp_exp: float) -> dict | None:
    """Make the scaler push ``kd_ratio x 2 sqrt(kp_eff)`` on every axis it scales.

    Returns what will be pushed at s = 1 (for the manifest), or None when nothing
    changes: no scaler, ratio 1, or a scaler whose ``apply()`` cached nothing.
    Never raises -- a damping tweak must not stop a run.
    """
    if scaler is None or kd_ratio <= 0 or abs(kd_ratio - 1.0) < 1e-9:
        return None
    try:
        kp = dict(getattr(scaler, "_original_kp", None) or {})
        if not kp:
            logger.warning("kd_ratio=%.2f requested but the scaler cached no kp; "
                           "damping left on auto", kd_ratio)
            return None
        true_kd = dict(getattr(scaler, "_original_kd", {}) or {})
        true_auto = dict(getattr(scaler, "_kd_is_auto", {}) or {})
        by_suffix = {name.rsplit(".", 1)[-1]: v for name, v in kp.items()}
        pushed: dict[str, float] = {}
        for kd_name in sorted(set(true_kd) | set(true_auto)):
            suffix = kd_name.rsplit(".", 1)[-1]           # d_pos_x -> k_pos_x
            kp_name = suffix.replace("d_", "k_", 1)
            if kp_name not in by_suffix or by_suffix[kp_name] is None:
                continue
            auto = true_auto.get(kd_name, True) or (true_kd.get(kd_name) or 0.0) <= 0.0
            base = 2.0 * math.sqrt(float(by_suffix[kp_name])) if auto else float(true_kd[kd_name])
            scaler._original_kd[kd_name] = kd_ratio * base
            scaler._kd_is_auto[kd_name] = False
            pushed[kd_name] = kd_ratio * base
        if not pushed:
            return None
        scaler.kd_exp = float(kp_exp) / 2.0

        orig_restore = scaler.restore

        def restore():
            scaler._original_kd.clear()
            scaler._original_kd.update(true_kd)
            scaler._kd_is_auto.clear()
            scaler._kd_is_auto.update(true_auto)
            logger.info("damping: restoring the controller's original kd (auto) before restore()")
            orig_restore()

        scaler.restore = restore
        logger.info("damping: kd_ratio=%.2f -> pushing kd = %.2f x 2sqrt(kp), scaling as "
                    "s^%.2f; at s=1: %s", kd_ratio, kd_ratio, scaler.kd_exp,
                    {k.rsplit(".", 1)[-1]: round(v, 1) for k, v in pushed.items()})
        return {"kd_ratio": float(kd_ratio), "kd_exp": scaler.kd_exp, "kd_at_s1": pushed}
    except Exception:
        logger.exception("raise_damping failed; damping left as the scaler had it")
        return None
