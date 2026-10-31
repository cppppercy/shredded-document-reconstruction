"""Generate deterministic synthetic documents for training and evaluation."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DOCUMENT_VARIANTS = ("report", "table", "mixed", "sparse")

WORDS = (
    "analysis model feature learning document reconstruction global optimization "
    "dataset experiment method result precision recall image boundary compatibility "
    "training evaluation algorithm robust baseline synthetic visual evidence system"
).split()


def _font(size: int, *, mono: bool = False, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if mono:
        candidates.extend(
            [
                "C:/Windows/Fonts/consola.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                "DejaVuSansMono.ttf",
            ]
        )
    elif bold:
        candidates.extend(
            [
                "C:/Windows/Fonts/arialbd.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "DejaVuSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "C:/Windows/Fonts/arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "DejaVuSans.ttf",
            ]
        )
    for candidate in candidates:
        if candidate.startswith(("C:/", "/")) and not Path(candidate).exists():
            continue
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _paper(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    """Create lightly textured paper whose continuity is visible at a true seam."""
    coarse = rng.normal(0.0, 1.0, size=(max(3, height // 35), max(3, width // 35)))
    low = Image.fromarray(np.uint8(np.clip((coarse + 3.0) * 36, 0, 255)), mode="L")
    low = low.resize((width, height), Image.Resampling.BICUBIC)
    texture = (np.asarray(low, dtype=np.float32) - 108.0) * 0.035
    yy, xx = np.mgrid[:height, :width]
    illumination = 1.2 * np.sin(xx / 87.0) + 0.8 * np.cos(yy / 113.0)
    base = 247.0 + texture + illumination
    rgb = np.stack((base + 2.0, base + 1.0, base - 1.0), axis=-1)
    return Image.fromarray(np.uint8(np.clip(rgb, 0, 255)), mode="RGB")


def _sentence(rng: random.Random, min_words: int = 9, max_words: int = 16) -> str:
    count = rng.randint(min_words, max_words)
    words = [rng.choice(WORDS) for _ in range(count)]
    text = " ".join(words)
    return text[0].upper() + text[1:] + "."


def _draw_wrapped_lines(
    draw: ImageDraw.ImageDraw,
    rng: random.Random,
    top: int,
    count: int,
    width: int,
    *,
    left: int = 54,
    line_height: int = 29,
) -> int:
    font = _font(18)
    y = top
    for _ in range(count):
        text = _sentence(rng)
        while draw.textlength(text, font=font) > width - left - 52:
            text = text.rsplit(" ", 1)[0]
        draw.text((left, y), text, fill=(42, 47, 53), font=font)
        y += line_height
    return y


def _draw_report(draw: ImageDraw.ImageDraw, rng: random.Random, width: int, height: int) -> None:
    draw.text((54, 45), "RECONSTRUCTION EXPERIMENT", fill=(24, 45, 74), font=_font(31, bold=True))
    draw.text((56, 88), f"Technical note  |  sample {rng.randint(1000, 9999)}", fill=(74, 83, 95), font=_font(16))
    draw.line((54, 118, width - 54, 118), fill=(47, 91, 132), width=3)
    y = _draw_wrapped_lines(draw, rng, 145, 6, width)
    draw.rounded_rectangle((55, y + 12, width - 55, y + 125), radius=9, outline=(91, 109, 128), width=2)
    draw.text((75, y + 30), "Observation", fill=(35, 69, 103), font=_font(21, bold=True))
    _draw_wrapped_lines(draw, rng, y + 66, 2, width, left=75, line_height=25)
    y += 165
    _draw_wrapped_lines(draw, rng, y, 5, width)
    draw.text((width - 170, height - 54), "Page 1 / 1", fill=(91, 96, 104), font=_font(15))


def _draw_table(draw: ImageDraw.ImageDraw, rng: random.Random, width: int, height: int) -> None:
    draw.text((54, 42), "MODEL COMPARISON TABLE", fill=(38, 57, 81), font=_font(30, bold=True))
    draw.text((55, 84), _sentence(rng, 8, 11), fill=(68, 74, 82), font=_font(16))
    left, right, top, bottom = 55, width - 55, 130, height - 92
    rows, cols = 9, 5
    row_h = (bottom - top) // rows
    col_w = (right - left) // cols
    draw.rectangle((left, top, right, bottom), outline=(43, 62, 82), width=3)
    draw.rectangle((left, top, right, top + row_h), fill=(214, 226, 239))
    for row in range(1, rows):
        y = top + row * row_h
        draw.line((left, y, right, y), fill=(95, 111, 126), width=2 if row == 1 else 1)
    for col in range(1, cols):
        x = left + col * col_w
        draw.line((x, top, x, bottom), fill=(95, 111, 126), width=1)
    headers = ("Method", "Pairs", "Score", "Time", "Status")
    for col, value in enumerate(headers):
        draw.text((left + col * col_w + 10, top + 14), value, fill=(27, 45, 63), font=_font(16, bold=True))
    methods = ("Edge", "Gradient", "Profile", "Linear", "Hybrid", "DP", "Ablation", "Final")
    for row, method in enumerate(methods, 1):
        values = (method, str(rng.randint(54, 132)), f"{rng.uniform(0.45, 0.98):.3f}", f"{rng.uniform(0.2, 4.8):.1f}s", rng.choice(("pass", "pass", "review")))
        for col, value in enumerate(values):
            draw.text((left + col * col_w + 10, top + row * row_h + 13), value, fill=(42, 48, 55), font=_font(15, mono=col in (1, 2, 3)))
    draw.text((55, height - 63), "Table 1. Results are generated from a held-out synthetic document.", fill=(83, 89, 97), font=_font(15))


def _draw_mixed(draw: ImageDraw.ImageDraw, rng: random.Random, width: int, height: int) -> None:
    draw.text((54, 42), "DOCUMENT ANALYSIS", fill=(34, 59, 87), font=_font(31, bold=True))
    draw.line((54, 86, width - 54, 86), fill=(57, 99, 139), width=2)
    y = _draw_wrapped_lines(draw, rng, 112, 4, width)
    draw.text((55, y + 10), "Key findings", fill=(35, 68, 102), font=_font(22, bold=True))
    for index in range(4):
        yy = y + 48 + index * 31
        draw.ellipse((58, yy + 6, 66, yy + 14), fill=(47, 92, 134))
        draw.text((78, yy), _sentence(rng, 6, 10), fill=(47, 51, 57), font=_font(17))
    chart_left, chart_top = 55, y + 194
    chart_right, chart_bottom = width - 55, height - 68
    draw.rectangle((chart_left, chart_top, chart_right, chart_bottom), outline=(118, 127, 137), width=1)
    colors = ((67, 126, 177), (91, 160, 132), (223, 142, 76), (174, 94, 103), (128, 110, 166), (89, 153, 170))
    usable = chart_right - chart_left - 70
    bar_w = usable // len(colors)
    for idx, color in enumerate(colors):
        bar_h = rng.randint(65, max(70, chart_bottom - chart_top - 45))
        x0 = chart_left + 35 + idx * bar_w
        draw.rectangle((x0, chart_bottom - bar_h, x0 + bar_w - 22, chart_bottom - 1), fill=color)
        draw.text((x0 + 2, chart_bottom - bar_h - 23), f"{bar_h}", fill=(58, 62, 68), font=_font(13))


def _draw_sparse(draw: ImageDraw.ImageDraw, rng: random.Random, width: int, height: int) -> None:
    """Deliberately ambiguous layout used to expose V0.1 failure cases."""
    draw.text((width // 2 - 160, 72), "SHORT MEMO", fill=(48, 58, 70), font=_font(30, bold=True))
    draw.line((180, 121, width - 180, 121), fill=(115, 123, 132), width=2)
    _draw_wrapped_lines(draw, rng, 180, 3, width, left=190, line_height=34)
    draw.rectangle((width // 2 - 120, 380, width // 2 + 120, 505), outline=(80, 100, 120), width=3)
    draw.ellipse((width // 2 - 45, 405, width // 2 + 45, 495), outline=(67, 124, 156), width=5)
    draw.text((width // 2 - 104, height - 88), "Sparse content stress test", fill=(91, 98, 106), font=_font(16))


def generate_document(
    seed: int,
    variant: str = "report",
    size: tuple[int, int] = (960, 720),
    *,
    with_border: bool = True,
    with_texture: bool | None = None,
) -> Image.Image:
    """Generate a deterministic document image without external data.

    ``with_texture=None`` preserves the V0.1 behaviour: sparse documents have
    no continuous texture while the other variants do.  Passing ``False`` is
    useful for a harder, less synthetic-looking ablation.
    """
    if variant not in DOCUMENT_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {DOCUMENT_VARIANTS}")
    width, height = size
    if width < 480 or height < 360:
        raise ValueError("Document size must be at least 480x360")
    np_rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    use_texture = variant != "sparse" if with_texture is None else with_texture
    image = _paper(width, height, np_rng) if use_texture else Image.new("RGB", (width, height), (248, 248, 246))
    draw = ImageDraw.Draw(image)
    if with_border:
        draw.rectangle((12, 12, width - 13, height - 13), outline=(92, 99, 107), width=2)
    {
        "report": _draw_report,
        "table": _draw_table,
        "mixed": _draw_mixed,
        "sparse": _draw_sparse,
    }[variant](draw, py_rng, width, height)
    return image
