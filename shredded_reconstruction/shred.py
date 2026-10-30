"""Cut, shuffle, and compose vertical document strips."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

from PIL import Image


@dataclass(frozen=True)
class ShreddedDocument:
    """A shuffled observation plus labels kept only for evaluation."""

    strips: tuple[Image.Image, ...]
    original_indices: tuple[int, ...]
    strip_count: int


def cut_vertical_strips(image: Image.Image, strip_count: int) -> list[Image.Image]:
    if not 2 <= strip_count <= image.width:
        raise ValueError("strip_count must be between 2 and the image width")
    boundaries = [round(i * image.width / strip_count) for i in range(strip_count + 1)]
    return [image.crop((boundaries[i], 0, boundaries[i + 1], image.height)).copy() for i in range(strip_count)]


def shred_document(image: Image.Image, strip_count: int, shuffle_seed: int) -> ShreddedDocument:
    """Return shuffled strips; original_indices are not used by reconstruction."""
    strips = cut_vertical_strips(image, strip_count)
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
