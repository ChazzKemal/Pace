"""B-spline action representation (Han et al., arXiv:2607.09648)."""

from pace_bench.methods.bspline.spline import (
    DEGREE,
    MAX_ERROR,
    RAW_DIM,
    SPLINE_DIM,
    assign_chunks_to_frames,
    chunk_parameters,
    decode_chunk,
    decode_relative_knots,
    encode_relative_knots,
    episode_parameter_chunks,
    fit_episode,
    from_spline_actions,
    to_spline_actions,
)

__all__ = [
    "DEGREE",
    "MAX_ERROR",
    "RAW_DIM",
    "SPLINE_DIM",
    "assign_chunks_to_frames",
    "chunk_parameters",
    "decode_chunk",
    "decode_relative_knots",
    "encode_relative_knots",
    "episode_parameter_chunks",
    "fit_episode",
    "from_spline_actions",
    "to_spline_actions",
]
