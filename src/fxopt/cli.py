"""One small CLI for grids, heatmaps, and Shift-click replay."""

from pathlib import Path
import json
import platform
import shutil
import subprocess
import sys
import threading
import time

import click

from .config import ConfigError, RunConfig
from .remote_jobs import (
    REMOTE_JOB_FILENAME, follow_remote_run, remote_run_status,
    retrieve_remote_run, run_remote_config, stop_remote_run,
)
from .run import (
    grid_summary,
    run_config,
    run_distributed_config,
    run_leased_worker,
)


def _overwrite_output(config: Path, output: Path) -> None:
    destination = output.expanduser().resolve()
    if not destination.exists():
        return
    if not destination.is_dir():
        raise ConfigError("--overwrite output must be a directory")
    if destination in config.resolve().parents:
        raise ConfigError("--overwrite refuses to remove a configuration ancestor")
    if (destination / REMOTE_JOB_FILENAME).exists():
        raise ConfigError(
            "--overwrite refuses a detached remote job; retrieve it or use a different output"
        )
    entries = tuple(destination.iterdir())
    markers = {"run.json", "results.npz", ".results.npz.tmp"}
    if entries and not any((destination / name).exists() for name in markers):
        raise ConfigError("--overwrite target is not an fxopt run directory")
    shutil.rmtree(destination)


class _ProgressReporter:
    def __init__(
        self,
        label: str,
        *,
        _interval: float = 10.0,
    ) -> None:
        self.label = label
        self.interval = _interval
        self.started_at = time.monotonic()
        self.latest: tuple[int, int] | None = None
        self.last_reported_completed: int | None = None
        self.initial_printed = False
        self.finished = False
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.started = False
        self.baseline_completed: int | None = None

    def start(self) -> None:
        if self.started:
            return
        self.started = True
        self.started_at = time.monotonic()
        self.thread.start()

    def __call__(self, completed: int, total: int) -> None:
        self.start()
        with self.lock:
            if self.baseline_completed is None:
                self.baseline_completed = completed
            self.latest = (completed, total)
            final = completed >= total
            if self.finished:
                return
            if not self.initial_printed or final:
                self._write(completed, total)
                self.last_reported_completed = completed
                self.initial_printed = True
            self.finished = final

    def _heartbeat(self) -> None:
        while not self.stop_event.wait(self.interval):
            with self.lock:
                if self.latest is not None and not self.finished:
                    completed, total = self.latest
                    stale = completed == self.last_reported_completed
                    self._write(completed, total, working=stale)
                    if not stale:
                        self.last_reported_completed = completed

    def _write(self, completed: int, total: int, *, working: bool = False) -> None:
        now = time.monotonic()
        elapsed = max(0.0, now - self.started_at)
        percent = min(100, int(completed * 100 / total)) if total else 100
        if working:
            click.echo(
                f"{self.label}: waiting... {completed}/{total} saved ({percent}%) "
                f"elapsed {elapsed:.1f}s",
                err=True,
            )
            return
        baseline = self.baseline_completed or 0
        newly_completed = max(0, completed - baseline)
        rate = newly_completed / elapsed if newly_completed > 0 and elapsed > 0 else 0.0
        eta = max(0, total - completed) / rate if rate > 0 else None
        eta_text = "--" if eta is None else f"{eta:.1f}s"
        click.echo(
            f"{self.label}: saved {completed}/{total} ({percent}%) elapsed {elapsed:.1f}s "
            f"pools/s {rate:.1f} ETA {eta_text}",
            err=True,
        )

    def close(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join()


class _WorkerReporter:
    def __init__(self, worker_index: int, interval: float = 2.0) -> None:
        self.worker_index = worker_index
        self.interval = interval
        self.latest: tuple[int, int, float] | None = None
        self.lock = threading.Lock()
        self.output_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.started = False

    def __call__(self, completed: int, total: int, calculation_s: float) -> None:
        with self.lock:
            self.latest = (completed, total, calculation_s)
            if not self.started:
                self.started = True
                self.thread.start()

    def _heartbeat(self) -> None:
        while not self.stop_event.wait(self.interval):
            with self.lock:
                latest = self.latest
            if latest is None:
                continue
            completed, total, calculation_s = latest
            self.emit({
                "type": "progress",
                "worker_index": self.worker_index,
                "completed": completed,
                "total": total,
                "calculation_s": calculation_s,
            })

    def emit(self, message: dict[str, object]) -> None:
        with self.output_lock:
            click.echo(json.dumps(
                message,
                sort_keys=True,
                separators=(",", ":"),
            ))

    def close(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join()


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
@click.option("--transfer", is_flag=True, help="Rsync workspace sources once into shared home.")
@click.option("--rebuild", is_flag=True, help="Transfer and rebuild the shared evaluator once.")
@click.option("--status", "status_only", is_flag=True, help="Check detached remote state once.")
@click.option("--follow", is_flag=True, help="Follow a detached job, then retrieve it.")
@click.option("--retrieve", is_flag=True, help="Retrieve an already-complete remote job.")
@click.option("--stop", "stop_only", is_flag=True, help="Stop a detached remote job.")
@click.option("--overwrite", is_flag=True, help="Replace an existing completed run directory.")
def run_command(
    config: Path,
    output_dir: Path,
    transfer: bool,
    rebuild: bool,
    status_only: bool,
    follow: bool,
    retrieve: bool,
    stop_only: bool,
    overwrite: bool,
) -> None:
    """Run CONFIG or manage its detached remote coordinator."""
    reporter: _ProgressReporter | None = None
    try:
        config_value = RunConfig.from_toml(config)
        modes = sum((status_only, follow, retrieve, stop_only))
        if modes > 1:
            raise ConfigError(
                "--status, --follow, --retrieve, and --stop are mutually exclusive"
            )
        if (transfer or rebuild) and not config_value.hosts:
            raise ConfigError("--transfer and --rebuild require remote placement hosts")
        if overwrite and modes:
            raise ConfigError("--overwrite cannot be combined with remote job controls")
        if modes and not config_value.hosts:
            raise ConfigError("remote job controls require placement hosts")
        if overwrite:
            _overwrite_output(config, output_dir)
        if status_only:
            status = remote_run_status(config, output_dir)
            location = (
                f" at {status.coordinator}:{status.remote_output}"
                if status.remote_output is not None
                else ""
            )
            click.echo(f"remote: {status.state}{location}")
            if status.detail:
                click.echo(status.detail)
            return
        if stop_only:
            status = stop_remote_run(config, output_dir)
            location = (
                f" at {status.coordinator}:{status.remote_output}"
                if status.remote_output is not None
                else ""
            )
            click.echo(f"remote: {status.state}{location}")
            if status.detail:
                click.echo(status.detail)
            return
        if follow:
            paths = follow_remote_run(config, output_dir)
        elif retrieve:
            paths = retrieve_remote_run(config, output_dir)
        else:
            click.echo(grid_summary(config_value), err=True)
        if not modes and config_value.hosts:
            paths = run_remote_config(
                config_value,
                output_dir,
                transfer=transfer,
                rebuild=rebuild,
            )
        elif not modes:
            reporter = _ProgressReporter("run")
            paths = run_config(
                config_value,
                output_dir,
                progress_callback=reporter,
            )
    except (
        ConfigError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        TypeError,
        ValueError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        if reporter is not None:
            reporter.close()
    click.echo(f"wrote {paths.run_json} and {paths.results_npz}")


@main.command("_cluster-worker", hidden=True)
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "output_dir", required=True,
              type=click.Path(file_okay=False, path_type=Path))
@click.option("--origin-workspace", required=True, type=click.Path(path_type=Path))
@click.option("--origin-config", required=True, type=click.Path(path_type=Path))
def cluster_worker_command(
    config: Path,
    output_dir: Path,
    origin_workspace: Path,
    origin_config: Path,
) -> None:
    """Coordinate one portable worker process per configured placement."""
    try:
        paths = run_distributed_config(
            config,
            output_dir,
            origin_workspace=origin_workspace,
            origin_config=origin_config,
        )
    except (ConfigError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    payload = json.loads(paths.run_json.read_text())
    counts = payload.get("status_counts", {})
    click.echo(
        f"coordinator: wrote {paths.run_json} and {paths.results_npz} "
        f"({counts.get('ok', 0)} ok, {counts.get('failed', 0)} failed)",
        err=True,
    )


@main.command("_worker", hidden=True)
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "output_dir", required=True,
              type=click.Path(file_okay=False, path_type=Path))
@click.option("--worker-index", required=True, type=click.IntRange(min=0))
def worker_command(
    config: Path,
    output_dir: Path,
    worker_index: int,
) -> None:
    """Evaluate coordinator-assigned deterministic grid leases."""
    reporter = _WorkerReporter(worker_index)

    def commands():
        for line in sys.stdin:
            if line.strip():
                yield json.loads(line)

    def ready(
        lease_id: int | None,
        completed: int,
        total: int,
        calculation_s: float,
    ) -> None:
        message: dict[str, object] = {
            "type": "ready",
            "worker_index": worker_index,
            "completed": completed,
            "total": total,
            "calculation_s": calculation_s,
        }
        if lease_id is not None:
            message["completed_lease_id"] = lease_id
        reporter.emit(message)

    try:
        receipt = run_leased_worker(
            config,
            output_dir,
            worker_index=worker_index,
            commands=commands(),
            progress_callback=reporter,
            ready_callback=ready,
        )
    except (ConfigError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        reporter.close()
    reporter.emit(receipt)


@main.command("heatmap")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--x", "x_axis")
@click.option("--y", "y_axis")
@click.option("--log-axis", "log_axes", multiple=True,
              help="Render this numeric grid axis logarithmically; repeat as needed.")
@click.option("--columns", type=click.IntRange(min=1), default=3, show_default=True)
@click.option("--metric", "metrics", multiple=True,
              default=("apy_net_robust_90d", "detach_energy_ungated", "max_7d_rel_price_diff"),
              show_default=True)
@click.option("--max-price-diff-bps", type=float, default=100.0, show_default=True)
@click.option("--max-detach-energy", type=click.FloatRange(min=0.0))
@click.option("--slippage-bps", type=float)
@click.option("--final-price-diff-bps", type=float)
@click.option(
    "--shiftclick-yb-mode",
    type=click.Choice(("off", "active_2l", "reference_2l")),
    default="active_2l",
    show_default=True,
    help="YB mode for local Shift-click replay; right-click stays off.",
)
@click.option(
    "--shiftclick-yb-cash-multiplier",
    type=click.FloatRange(min=0.0, min_open=True),
    default=3.0,
    show_default=True,
)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--show/--no-show", default=True, show_default=True)
def heatmap_command(
    run_dir: Path, x_axis: str | None, y_axis: str | None,
    log_axes: tuple[str, ...], columns: int, metrics: tuple[str, ...],
    max_price_diff_bps: float,
    max_detach_energy: float | None,
    slippage_bps: float | None,
    final_price_diff_bps: float | None,
    shiftclick_yb_mode: str,
    shiftclick_yb_cash_multiplier: float,
    output: Path | None, show: bool,
) -> None:
    """Open the interactive filtered heatmap explorer."""
    explorer = None
    try:
        from .explorer import open_fxopt_explorer
        explorer = open_fxopt_explorer(
            run_dir, metrics=metrics, x_axis=x_axis, y_axis=y_axis,
            log_axes=log_axes, columns=columns,
            max_price_diff_bps=max_price_diff_bps,
            max_detach_energy=max_detach_energy,
            slippage_bps=slippage_bps,
            final_price_diff_bps=final_price_diff_bps,
            shiftclick_yb_mode=shiftclick_yb_mode,
            shiftclick_yb_cash_multiplier=shiftclick_yb_cash_multiplier,
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
    finally:
        if explorer is not None:
            explorer.close()


@main.command("shiftclick")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--ordinal", required=True, type=click.IntRange(min=0))
@click.option("--output", "output_dir", required=True,
              type=click.Path(file_okay=False, path_type=Path))
@click.option("--trace-interval", default=200, show_default=True, type=click.IntRange(min=1))
@click.option("--actions/--no-actions", default=False, show_default=True)
@click.option(
    "--yb-mode",
    type=click.Choice(("off", "active_2l", "reference_2l")),
    default="active_2l",
    show_default=True,
)
@click.option(
    "--yb-cash-multiplier",
    type=click.FloatRange(min=0.0, min_open=True),
    default=3.0,
    show_default=True,
)
def shiftclick_command(
    run_dir: Path, ordinal: int, output_dir: Path,
    trace_interval: int, actions: bool, yb_mode: str, yb_cash_multiplier: float,
) -> None:
    """Replay one exact stored RUN_DIR candidate with a full trace."""
    from .results import read_result_columns
    from .shiftclick import save_shiftclick_plot, trace_stored_candidate

    try:
        columns = read_result_columns(run_dir, metrics=())
        candidate = columns.candidate_at(ordinal)
        path = trace_stored_candidate(
            columns.run_id,
            columns.metadata,
            candidate=candidate,
            ordinal=ordinal,
            output_dir=output_dir,
            trace_interval=trace_interval,
            trace_actions=actions,
            yb_mode=yb_mode,
            yb_cash_multiplier=yb_cash_multiplier,
        )
        local_platform = " ".join(value for value in (platform.system(), platform.machine()) if value)
        plot = save_shiftclick_plot(
            path,
            output_dir / "shiftclick.png",
            title=f"{columns.run_id}: {ordinal} | local {local_platform}",
        )
    except (ConfigError, OSError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"wrote {path} and {plot}")


if __name__ == "__main__":
    main()
