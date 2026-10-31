"""Named, deterministic observation conditions for V0.2 experiments."""

from __future__ import annotations

from dataclasses import dataclass

from .shred import DisturbanceConfig


CONDITION_NAMES = ("clean", "no_assists", "scan", "edge_loss", "vertical_shift")


@dataclass(frozen=True)
class ObservationCondition:
    name: str
    with_border: bool = True
    with_texture: bool | None = None
    disturbance: DisturbanceConfig = DisturbanceConfig()


def observation_condition(name: str, *, seed: int = 0) -> ObservationCondition:
    """Return a fixed condition; seed only controls visual nuisance draws."""
    if name == "clean":
        return ObservationCondition(name)
    if name == "no_assists":
        return ObservationCondition(name, with_border=False, with_texture=False)
    if name == "scan":
        brightness = 0.96 if seed % 2 else 1.04
        return ObservationCondition(
            name,
            disturbance=DisturbanceConfig(
                document_brightness=brightness,
                blur_radius=0.65,
                strip_brightness_jitter=0.035,
                seed=seed + 17,
            ),
        )
    if name == "edge_loss":
        return ObservationCondition(name, disturbance=DisturbanceConfig(edge_loss=2, seed=seed + 29))
    if name == "vertical_shift":
        return ObservationCondition(name, disturbance=DisturbanceConfig(vertical_jitter=3, seed=seed + 43))
    raise ValueError(f"Unknown observation condition {name!r}; expected one of {CONDITION_NAMES}")


def condition_description(name: str) -> str:
    return {
        "clean": "V0.1 原始干净条件",
        "no_assists": "关闭页面边框与连续纸张纹理",
        "scan": "整页亮度/轻微模糊 + 每条轻微亮度差",
        "edge_loss": "每条左右边缘随机永久缺失 0–2 像素",
        "vertical_shift": "每条上下随机错位 ±3 像素（不循环回绕）",
    }[name]
