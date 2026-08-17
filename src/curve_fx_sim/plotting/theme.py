"""Shared Matplotlib design tokens for all orchestrator plots."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlotTheme:
    canvas: str = "#f4f1ea"
    surface: str = "#ebe6db"
    ink: str = "#202a2c"
    muted_ink: str = "#5f6968"
    grid: str = "#c8c1b4"
    accent: str = "#176b64"
    accent_secondary: str = "#b45f3a"
    danger: str = "#a33d32"
    sequential_cmap: str = "viridis"
    diverging_cmap: str = "RdBu_r"
    font_family: str = "STIXGeneral"
    mono_family: str = "DejaVu Sans Mono"
    title_size: float = 15.0
    label_size: float = 11.0
    body_size: float = 10.0
    detail_size: float = 9.0
    line_width: float = 1.6
    grid_width: float = 0.7
    figure_width: float = 11.0
    panel_height: float = 2.35


DEFAULT_THEME = PlotTheme()


def apply_theme(figure: object, axes: object, theme: PlotTheme = DEFAULT_THEME) -> None:
    figure.patch.set_facecolor(theme.canvas)
    axis_list = axes.ravel() if hasattr(axes, "ravel") else (axes,)
    for axis in axis_list:
        if axis is None:
            continue
        axis.set_facecolor(theme.surface)
        axis.tick_params(colors=theme.muted_ink, labelsize=theme.detail_size)
        axis.xaxis.label.set_color(theme.ink)
        axis.yaxis.label.set_color(theme.ink)
        axis.title.set_color(theme.ink)
        for spine in axis.spines.values():
            spine.set_color(theme.grid)
        axis.grid(color=theme.grid, linewidth=theme.grid_width, alpha=0.45)


__all__ = ["DEFAULT_THEME", "PlotTheme", "apply_theme"]
