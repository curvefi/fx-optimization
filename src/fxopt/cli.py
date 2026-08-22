"""One small CLI for grids, optimization, heatmaps, and Shift-click replay."""

from pathlib import Path

import click

from .config import ConfigError
from .optimize import OptimizationError, optimize_config
from .results import read_result_columns
from .run import run_config
from .shiftclick import trace_candidate


@click.group()
def main() -> None:
    """Run and inspect local-or-SSH Curve FX simulations."""


@main.command("run")
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output",
    "output_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for run.json and results.npz.",
)
def run_command(config: Path, output_dir: Path) -> None:
    """Run CONFIG as bounded local-or-SSH evaluator batches."""
    try:
        paths = run_config(config, output_dir)
    except (ConfigError, OSError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"wrote {paths.run_json} and {paths.results_npz}")


@main.command("optimize")
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "output_dir", required=True,
              type=click.Path(file_okay=False, path_type=Path))
def optimize_command(config: Path, output_dir: Path) -> None:
    """Run Nevergrad from CONFIG through the same evaluator fleet."""
    try:
        paths = optimize_config(config, output_dir)
    except (ConfigError, OptimizationError, OSError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"wrote {paths.run_json} and {paths.results_npz}")


@main.command("heatmap")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--x", "x_axis")
@click.option("--y", "y_axis")
@click.option("--metric", "metrics", multiple=True,
              default=("apy_net", "max_7d_rel_price_diff", "tw_real_slippage_1pct", "max_7d_skew"),
              show_default=True)
@click.option("--max-price-diff-bps", type=float, default=100.0, show_default=True)
@click.option("--max-skew-percent", type=float)
@click.option("--slippage-bps", type=float, default=20.0, show_default=True)
@click.option("--final-price-diff-bps", type=float)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--show/--no-show", default=True, show_default=True)
def heatmap_command(
    run_dir: Path, x_axis: str | None, y_axis: str | None,
    metrics: tuple[str, ...], max_price_diff_bps: float,
    max_skew_percent: float | None, slippage_bps: float,
    final_price_diff_bps: float | None, output: Path | None, show: bool,
) -> None:
    """Open the interactive filtered heatmap explorer."""
    try:
        from .explorer import open_fxopt_explorer
        explorer = open_fxopt_explorer(
            run_dir, metrics=metrics, x_axis=x_axis, y_axis=y_axis,
            max_price_diff_bps=max_price_diff_bps,
            max_skew_percent=max_skew_percent,
            slippage_bps=slippage_bps,
            final_price_diff_bps=final_price_diff_bps,
        )
        if output is not None:
            image, state = explorer.save(output)
            click.echo(f"wrote {image} and {state}")
        if show:
            explorer.show()
        elif output is None:
            raise ValueError("--no-show requires --output")
    except (OSError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("shiftclick")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--ordinal", required=True, type=click.IntRange(min=0))
@click.option("--output", "output_dir", required=True,
              type=click.Path(file_okay=False, path_type=Path))
@click.option("--trace-interval", default=200, show_default=True, type=click.IntRange(min=1))
@click.option("--actions/--no-actions", default=False, show_default=True)
def shiftclick_command(
    run_dir: Path, ordinal: int, output_dir: Path,
    trace_interval: int, actions: bool,
) -> None:
    """Replay one exact stored RUN_DIR candidate with a full trace."""
    try:
        columns = read_result_columns(run_dir, metrics=())
        config = columns.metadata.get("config")
        if not isinstance(config, str):
            raise ValueError("run metadata has no source config for Shift-click replay")
        candidate = columns.candidate_at(ordinal)
        path = trace_candidate(config, candidate=candidate, ordinal=ordinal, output_dir=output_dir,
                               trace_interval=trace_interval, trace_actions=actions)
    except (ConfigError, OSError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"wrote {path}")


if __name__ == "__main__":
    main()
