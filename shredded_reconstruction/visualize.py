"""Create clear, offline visualizations of reconstruction results."""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image, ImageDraw, ImageOps

from .document import _font


def shuffled_montage(strips: Sequence[Image.Image]) -> Image.Image:
    label_height = 38
    width = sum(strip.width for strip in strips)
    height = strips[0].height + label_height
    output = Image.new("RGB", (width, height), (229, 234, 240))
    draw = ImageDraw.Draw(output)
    x = 0
    for input_position, strip in enumerate(strips):
        output.paste(strip.convert("RGB"), (x, label_height))
        draw.text((x + max(4, strip.width // 2 - 9), 8), str(input_position), fill=(28, 43, 60), font=_font(16, bold=True))
        draw.line((x, 0, x, height), fill=(183, 78, 68), width=2)
        x += strip.width
    draw.line((width - 1, 0, width - 1, height), fill=(183, 78, 68), width=2)
    return output


def _panel(image: Image.Image, title: str, subtitle: str, size: tuple[int, int]) -> Image.Image:
    width, height = size
    canvas = Image.new("RGB", size, (244, 247, 251))
    draw = ImageDraw.Draw(canvas)
    draw.text((22, 14), title, fill=(28, 45, 66), font=_font(24, bold=True))
    draw.text((22, 48), subtitle, fill=(76, 89, 105), font=_font(14))
    area = (width - 40, height - 92)
    fitted = ImageOps.contain(image.convert("RGB"), area, Image.Resampling.LANCZOS)
    x = (width - fitted.width) // 2
    y = 78 + (area[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    draw.rectangle((x - 1, y - 1, x + fitted.width, y + fitted.height), outline=(188, 198, 210), width=1)
    return canvas


def comparison_figure(
    original: Image.Image,
    shuffled: Image.Image,
    baseline: Image.Image,
    learned: Image.Image | None,
    baseline_metrics: dict,
    learned_metrics: dict | None,
    *,
    panel_size: tuple[int, int] = (700, 560),
) -> Image.Image:
    baseline_order = baseline_metrics["predicted_original_order"]
    learned_order = learned_metrics["predicted_original_order"] if learned_metrics else None
    panels = [
        _panel(original, "1. Original document", "Ground truth used only for final evaluation", panel_size),
        _panel(shuffled, "2. Shuffled input strips", "Numbers are input positions, not correct labels", panel_size),
        _panel(
            baseline,
            "3. Hand-crafted baseline",
            f"adjacency={baseline_metrics['adjacency_accuracy']:.1%}  predicted={baseline_order}",
            panel_size,
        ),
        _panel(
            learned if learned is not None else baseline,
            "4. Learned compatibility + global DP" if learned is not None else "4. Learned method not run",
            (
                f"adjacency={learned_metrics['adjacency_accuracy']:.1%}  predicted={learned_order}"
                if learned_metrics
                else "Train a model to enable this panel"
            ),
            panel_size,
        ),
    ]
    gap = 16
    output = Image.new("RGB", (panel_size[0] * 2 + gap, panel_size[1] * 2 + gap), (221, 228, 237))
    for index, panel in enumerate(panels):
        output.paste(panel, ((index % 2) * (panel_size[0] + gap), (index // 2) * (panel_size[1] + gap)))
    return output


def evaluation_grid(images: Sequence[Image.Image], captions: Sequence[str], columns: int = 3) -> Image.Image:
    if len(images) != len(captions) or not images:
        raise ValueError("images and captions must have the same non-zero length")
    cell_w, cell_h = 430, 355
    rows = (len(images) + columns - 1) // columns
    grid = Image.new("RGB", (columns * cell_w, rows * cell_h), (231, 236, 243))
    for index, (image, caption) in enumerate(zip(images, captions, strict=True)):
        cell = Image.new("RGB", (cell_w - 12, cell_h - 12), (250, 251, 253))
        draw = ImageDraw.Draw(cell)
        draw.text((14, 12), caption, fill=(36, 51, 70), font=_font(16, bold=True))
        fitted = ImageOps.contain(image.convert("RGB"), (cell_w - 38, cell_h - 64), Image.Resampling.LANCZOS)
        cell.paste(fitted, ((cell.width - fitted.width) // 2, 48))
        grid.paste(cell, ((index % columns) * cell_w + 6, (index // columns) * cell_h + 6))
    return grid
