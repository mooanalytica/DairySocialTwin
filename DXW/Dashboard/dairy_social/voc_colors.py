from __future__ import annotations


VOC_CLS_COLORS_RGB: tuple[tuple[int, int, int], ...] = (
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128),
    (128, 128, 128),
    (64, 0, 0),
    (192, 0, 0),
    (64, 128, 0),
    (192, 128, 0),
    (64, 0, 128),
    (192, 0, 128),
    (64, 128, 128),
    (192, 128, 128),
    (0, 64, 0),
    (128, 64, 0),
    (0, 192, 0),
    (128, 192, 0),
    (0, 64, 128),
)


def voc_cls_color() -> tuple[tuple[int, int, int], ...]:
    return VOC_CLS_COLORS_RGB


def voc_cls_color_hexes() -> tuple[str, ...]:
    return tuple(
        f"#{red:02x}{green:02x}{blue:02x}"
        for red, green, blue in VOC_CLS_COLORS_RGB
    )
