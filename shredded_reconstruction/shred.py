"""Cut, shuffle, and compose vertical document strips."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class ShreddedDocument:
    """A shuffled observation plus labels kept only for evaluation."""

    strips: tuple[Image.Image, ...]
    original_indices: tuple[int, ...]
    strip_count: int


@dataclass(frozen=True)
class DisturbanceConfig:
    """Small, reproducible visual perturbations used only to form observations.

    Document-level effects happen before cutting. Strip-level effects happen
    independently after cutting, so there is no hidden positional marker.
    """

    document_brightness: float = 1.0
    blur_radius: float = 0.0
    strip_brightness_jitter: float = 0.0
    edge_loss: int = 0
    vertical_jitter: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.document_brightness <= 0 or self.blur_radius < 0:
            raise ValueError("brightness must be positive and blur_radius non-negative")
        if self.strip_brightness_jitter < 0 or self.edge_loss < 0 or self.vertical_jitter < 0:
            raise ValueError("disturbance magnitudes must be non-negative")


def disturb_document(image: Image.Image, config: DisturbanceConfig | None = None) -> Image.Image:
    """Apply document-wide scan effects before shredding."""
    config = config or DisturbanceConfig()
    result = image.copy()
    if config.document_brightness != 1.0:
        result = ImageEnhance.Brightness(result).enhance(config.document_brightness)
    if config.blur_radius:
        result = result.filter(ImageFilter.GaussianBlur(radius=config.blur_radius))
    return result


def _paper_fill(image: Image.Image) -> tuple[int, ...] | int:
    array = np.asarray(image)
    if array.ndim == 2:
        return int(np.median(array))
    return tuple(int(value) for value in np.median(array.reshape(-1, array.shape[-1]), axis=0))


def disturb_strips(strips: Sequence[Image.Image], config: DisturbanceConfig | None = None) -> list[Image.Image]:
    """Independently perturb strips without using their true positions.

    Vertical translations paste onto a fresh canvas rather than using array
    rolling, therefore pixels that leave at one end are genuinely lost.
    """
    config = config or DisturbanceConfig()
    rng = np.random.default_rng(config.seed)
    result: list[Image.Image] = []
    for strip in strips:
        item = strip.copy()
        if config.strip_brightness_jitter:
            factor = float(rng.uniform(1.0 - config.strip_brightness_jitter, 1.0 + config.strip_brightness_jitter))
            item = ImageEnhance.Brightness(item).enhance(factor)
        if config.edge_loss:
            left_loss = int(rng.integers(0, config.edge_loss + 1))
            right_loss = int(rng.integers(0, config.edge_loss + 1))
            if left_loss + right_loss >= item.width - 1:
                raise ValueError("edge_loss is too large for strip width")
            item = item.crop((left_loss, 0, item.width - right_loss, item.height)).copy()
        if config.vertical_jitter:
            offset = int(rng.integers(-config.vertical_jitter, config.vertical_jitter + 1))
            if offset:
                shifted = Image.new(item.mode, item.size, color=_paper_fill(item))
                shifted.paste(item, (0, offset))
                item = shifted
        result.append(item)
    return result


def cut_vertical_strips(image: Image.Image, strip_count: int) -> list[Image.Image]:
    if not 2 <= strip_count <= image.width:
        raise ValueError("strip_count must be between 2 and the image width")
    boundaries = [round(i * image.width / strip_count) for i in range(strip_count + 1)]
    return [image.crop((boundaries[i], 0, boundaries[i + 1], image.height)).copy() for i in range(strip_count)]


def shred_document(
    image: Image.Image,
    strip_count: int,
    shuffle_seed: int,
    disturbance: DisturbanceConfig | None = None,
) -> ShreddedDocument:
    """Return shuffled strips; original_indices are not used by reconstruction."""
    config = disturbance or DisturbanceConfig()
    strips = disturb_strips(cut_vertical_strips(disturb_document(image, config), strip_count), config)
    indices = list(range(strip_count))
    random.Random(shuffle_seed).shuffle(indices)
    return ShreddedDocument(tuple(strips[index] for index in indices), tuple(indices), strip_count)


def compose_strips(strips: Sequence[Image.Image], order: Sequence[int] | None = None) -> Image.Image:
    if not strips:
        raise ValueError("At least one strip is required")
    order = list(range(len(strips))) if order is None else list(order)
    if sorted(order) != list(range(len(strips))):
        raise ValueError("order must be a permutation of strip positions")
    heights = {strips[index].height for index in order}
    modes = {strips[index].mode for index in order}
    if len(heights) != 1 or len(modes) != 1:
        raise ValueError("All strips must have the same height and mode")
    output = Image.new(strips[order[0]].mode, (sum(strips[index].width for index in order), heights.pop()))
    x = 0
    for index in order:
        output.paste(strips[index], (x, 0))
        x += strips[index].width
    return output
