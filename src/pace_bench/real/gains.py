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


def raise_damping(scaler, *, kp_exp: float, kd_ratio: float = 1.0,
                  kd_base: float = 0.0, kd_base_rot: float = 0.0,
                  kd_exp: float = 1.0) -> dict | None:
    """Give every scaled axis an explicit kd instead of the controller's auto one.

    Two ways to say what you want, ``kd_base`` winning when both are set:

    * ``kd_base`` > 0: push ``kd_base`` at s = 1 and ``kd_base * s**kd_exp`` above
      it -- an absolute number, e.g. 100 N s/m against kp 400 (auto would be 40),
      growing as ``s**1.5``. This is the scaler's own law with a base it could not
      read from an auto axis. **Translation axes only**: the rotation axes have
      their own stiffness (k_rot 100 on this rig, auto d_rot = 20) and their own
      units, and take ``kd_base_rot`` -- left at 0 they stay on auto. Pushing the
      translational number onto them (the 18:43 run on 2026-09-02) put 5-9x the
      auto damping on rotation and made the arm vibrate.
    * ``kd_ratio`` != 1: push ``kd_ratio * 2 sqrt(kp)`` and keep it tracking
      ``sqrt(kp_eff)`` (``kd_exp`` becomes ``kp_exp / 2``) -- auto, times a factor.

    Returns what will be pushed at s = 1 (for the manifest), or None when nothing
    changes: no scaler, nothing requested, or a scaler whose ``apply()`` cached
    nothing. Never raises -- a damping tweak must not stop a run.
    """
    use_base = kd_base is not None and kd_base > 0
    if scaler is None or (not use_base and (kd_ratio <= 0 or abs(kd_ratio - 1.0) < 1e-9)):
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
            is_rot = "rot" in suffix
            if use_base:
                if is_rot and not (kd_base_rot and kd_base_rot > 0):
                    continue                                  # rotation stays on auto
                val = float(kd_base_rot if is_rot else kd_base)
            else:
                auto = true_auto.get(kd_name, True) or (true_kd.get(kd_name) or 0.0) <= 0.0
                base = (2.0 * math.sqrt(float(by_suffix[kp_name])) if auto
                        else float(true_kd[kd_name]))
                val = kd_ratio * base
            scaler._original_kd[kd_name] = val
            scaler._kd_is_auto[kd_name] = False
            pushed[kd_name] = val
        if not pushed:
            return None
        scaler.kd_exp = float(kd_exp) if use_base else float(kp_exp) / 2.0

        orig_restore = scaler.restore

        def restore():
            scaler._original_kd.clear()
            scaler._original_kd.update(true_kd)
            scaler._kd_is_auto.clear()
            scaler._kd_is_auto.update(true_auto)
            logger.info("damping: restoring the controller's original kd (auto) before restore()")
            orig_restore()

        scaler.restore = restore
        logger.info("damping: %s, scaling as s^%.2f; at s=1: %s",
                    (f"kd_base={kd_base:g} (rot: {kd_base_rot or 'auto'})" if use_base
                     else f"kd_ratio={kd_ratio:.2f} x 2sqrt(kp)"),
                    scaler.kd_exp, {k.rsplit(".", 1)[-1]: round(v, 1) for k, v in pushed.items()})
        return {"kd_base": float(kd_base) if use_base else None,
                "kd_base_rot": (float(kd_base_rot) if use_base and kd_base_rot else None),
                "kd_ratio": None if use_base else float(kd_ratio),
                "kd_exp": scaler.kd_exp, "kd_at_s1": pushed}
    except Exception:
        logger.exception("raise_damping failed; damping left as the scaler had it")
        return None
