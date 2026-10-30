"""Visual features for deciding whether one strip belongs to the left of another."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image

FEATURE_NAMES = (
    "seam_mae",
    "seam_rmse",
    "linear_prediction_mae",
    "gradient_gap",
    "edge_profile_mae",
    "edge_profile_correlation_loss",
    "band_mean_gap",
    "dark_pixel_disagreement",
    "relative_seam_roughness",
)


def _gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def compatibility_features(left: Image.Image, right: Image.Image, band: int = 8) -> np.ndarray:
    """Describe the directed seam ``left -> right`` without using strip labels."""
    if left.height != right.height:
        raise ValueError("Strip heights must match")
    a, b = _gray(left), _gray(right)
    if a.shape[1] < 2 or b.shape[1] < 2:
        raise ValueError("Strips must be at least two pixels wide")
    width = max(2, min(band, a.shape[1], b.shape[1]))
    a_band, b_band = a[:, -width:], b[:, :width]
    last, first = a[:, -1], b[:, 0]
    seam = last - first
    seam_mae = float(np.mean(np.abs(seam)))
    seam_rmse = float(np.sqrt(np.mean(seam**2)))
    predicted = np.clip(2.0 * a[:, -1] - a[:, -2], 0.0, 1.0)
    prediction_mae = float(np.mean(np.abs(predicted - first)))
    left_gradient = a[:, -1] - a[:, -2]
    right_gradient = b[:, 1] - b[:, 0]
    gradient_gap = float(np.mean(np.abs(left_gradient - right_gradient)))
    left_profile = np.mean(1.0 - a_band, axis=1)
    right_profile = np.mean(1.0 - b_band, axis=1)
    profile_mae = float(np.mean(np.abs(left_profile - right_profile)))
    if np.std(left_profile) < 1e-6 or np.std(right_profile) < 1e-6:
        correlation_loss = 1.0
    else:
        correlation_loss = float(1.0 - np.clip(np.corrcoef(left_profile, right_profile)[0, 1], -1.0, 1.0))
    band_mean_gap = float(abs(np.mean(a_band) - np.mean(b_band)))
    dark_disagreement = float(np.mean((last < 0.82) != (first < 0.82)))
    internal = 0.5 * (
        float(np.mean(np.abs(np.diff(a_band, axis=1))))
        + float(np.mean(np.abs(np.diff(b_band, axis=1))))
    )
    relative_roughness = seam_mae / (internal + 1e-4)
    return np.asarray(
        [
            seam_mae,
            seam_rmse,
            prediction_mae,
            gradient_gap,
            profile_mae,
            correlation_loss,
            band_mean_gap,
            dark_disagreement,
            relative_roughness,
        ],
        dtype=np.float64,
    )


def pairwise_feature_tensor(strips: Sequence[Image.Image]) -> np.ndarray:
    count = len(strips)
    result = np.zeros((count, count, len(FEATURE_NAMES)), dtype=np.float64)
    for left in range(count):
        for right in range(count):
            if left != right:
                result[left, right] = compatibility_features(strips[left], strips[right])
    return result


def endpoint_costs(strips: Sequence[Image.Image]) -> tuple[np.ndarray, np.ndarray]:
    """Estimate page endpoints from visual border/margin evidence.

    Synthetic pages contain a thin page frame near each outer boundary. This is
    still image evidence: no original strip index is supplied to this function.
    """
    starts, ends = [], []
    for strip in strips:
        gray = _gray(strip)
        probe = max(4, min(20, strip.width // 3))
        left_columns = np.mean(gray[:, :probe], axis=0)
        right_columns = np.mean(gray[:, -probe:], axis=0)
        # A long dark line near the page edge is useful endpoint evidence.
        left_line = float(np.min(left_columns[2:])) if probe > 2 else float(np.min(left_columns))
        right_line = float(np.min(right_columns[:-2])) if probe > 2 else float(np.min(right_columns))
        left_edge_white = float(abs(np.mean(gray[:, 0]) - 0.97))
        right_edge_white = float(abs(np.mean(gray[:, -1]) - 0.97))
        starts.append(0.75 * left_line + 0.25 * left_edge_white)
        ends.append(0.75 * right_line + 0.25 * right_edge_white)
    starts_arr = np.asarray(starts)
    ends_arr = np.asarray(ends)
    # Normalize so endpoint terms influence but do not dominate seam evidence.
    starts_arr = (starts_arr - starts_arr.min()) / (np.ptp(starts_arr) + 1e-9)
    ends_arr = (ends_arr - ends_arr.min()) / (np.ptp(ends_arr) + 1e-9)
    # Breaking the page at an outer boundary can otherwise look deceptively
    # smooth (white paper next to white paper), so endpoints need meaningful
    # weight in the global path objective.
    return 2.0 * starts_arr, 2.0 * ends_arr
