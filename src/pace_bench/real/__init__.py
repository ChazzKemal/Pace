"""Real-robot deployment: the half of PACE that drives hardware.

Everything here depends on ``crisp_gym``, which the sim environment does not install.
Nothing outside this subpackage imports it, so ``pace_bench`` stays usable for
training, labelling and LIBERO evaluation on a machine with no robot stack.
"""
