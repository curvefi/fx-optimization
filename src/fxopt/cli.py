"""The small public command surface for the new fxopt run path."""

from pathlib import Path

import click

from .config import ConfigError
from .run import run_config


@click.group()
def main() -> None:
    """Evaluate a local-or-SSH candidate grid through one persistent harness session."""


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


if __name__ == "__main__":
    main()
