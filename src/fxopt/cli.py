"""One small CLI for grids, optimization, heatmaps, and Shift-click replay."""

from pathlib import Path
import json
import platform
import subprocess
import threading
import time

import click

from .config import ConfigError
from .run import (
    RunConfig,
    follow_remote_run,
    grid_summary,
    remote_run_status,
    retrieve_remote_run,
    run_config,
    run_remote_config,
    stop_remote_run,
)


class _ProgressReporter:
    def __init__(
        self,
        label: str,
        *,
        stream_blade: str | None = None,
        blade_index: int = 0,
        blade_count: int = 1,
        lanes_per_blade: int = 1,
        _interval: float = 10.0,
    ) -> None:
        self.label = label
        self.stream_blade = stream_blade
        self.blade_index = blade_index
        self.blade_count = blade_count
        self.lanes_per_blade = lanes_per_blade
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
        self.blade_completed = 0
        self.blade_total = 0
        self.blade_lane_started_at: dict[str, float] = {}
        self.blade_lane_completed_at: dict[str, float] = {}
        self.blade_lane_completed: dict[str, int] = {}

    def start(self) -> None:
        if self.started:
            return
        self.started = True
        self.started_at = time.monotonic()
        if self.stream_blade is None:
            self.thread.start()

    def __call__(self, completed: int, total: int) -> None:
        self.start()
        with self.lock:
            if self.baseline_completed is None:
                self.baseline_completed = completed
            self.latest = (completed, total)
            if self.stream_blade is not None:
                self.blade_total = self._blade_share(total)
            final = completed >= total
            if self.finished:
                return
            if self.stream_blade is not None:
                if not self.initial_printed:
                    self._write_blade()
                    self.initial_printed = True
                self.finished = final
                return
            if not self.initial_printed or final:
                self._write(completed, total)
                self.last_reported_completed = completed
                self.initial_printed = True
            self.finished = final

    def lane(self, name: str, count: int, elapsed: float) -> None:
        """Aggregate completed NUMA batches for one representative blade."""
        with self.lock:
            if self.stream_blade is None or name.split(":", 1)[0] != self.stream_blade:
                return
            now = time.monotonic()
            if elapsed > 0.0:
                self.blade_lane_started_at.setdefault(name, now - elapsed)
                self.blade_lane_completed_at[name] = now
                self.blade_lane_completed[name] = (
                    self.blade_lane_completed.get(name, 0) + count
                )
            self.blade_completed = min(self.blade_total, self.blade_completed + count)
            self._write_blade(now=now)

    def _blade_share(self, total: int) -> int:
        lane_count = self.blade_count * self.lanes_per_blade
        per_lane, extra = divmod(total, lane_count)
        first_lane = self.blade_index * self.lanes_per_blade
        blade_extra = min(self.lanes_per_blade, max(0, extra - first_lane))
        return per_lane * self.lanes_per_blade + blade_extra

    def _heartbeat(self) -> None:
        while not self.stop_event.wait(self.interval):
            with self.lock:
                if self.latest is not None and not self.finished:
                    completed, total = self.latest
                    stale = completed == self.last_reported_completed
                    self._write(completed, total, working=stale)
                    if not stale:
                        self.last_reported_completed = completed

    def _write_blade(self, *, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if not self.blade_lane_started_at:
            elapsed = max(0.0, now - self.started_at)
            click.echo(
                f"{self.stream_blade}: waiting for first batch... elapsed {elapsed:.1f}s",
                err=True,
            )
            return
        rate = self._blade_rate()
        remaining = max(0, self.blade_total - self.blade_completed)
        eta = remaining / rate if rate > 0 else None
        eta_text = "--" if eta is None else f"{eta:.1f}s"
        percent = (
            min(100, int(self.blade_completed * 100 / self.blade_total))
            if self.blade_total
            else 100
        )
        click.echo(
            f"{self.stream_blade}: {self.blade_completed}/{self.blade_total} "
            f"({percent}%) {rate:.1f} pools/s ETA {eta_text}",
            err=True,
        )

    def _blade_rate(self) -> float:
        rate = 0.0
        for name, completed in self.blade_lane_completed.items():
            elapsed = (
                self.blade_lane_completed_at[name]
                - self.blade_lane_started_at[name]
            )
            if elapsed > 0.0:
                rate += completed / elapsed
        return rate

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

    def completion_summary(self, counts: dict[str, int]) -> str:
        elapsed = max(0.0, time.monotonic() - self.started_at)
        total = sum(counts.values())
        cluster_rate = total / elapsed if elapsed > 0 else 0.0
        blade_rate = self._blade_rate()
        summary = (
            f"complete {total} pools in {elapsed:.1f}s, "
            f"{cluster_rate:.1f} pools/s cluster wall"
        )
        if self.stream_blade is not None:
            summary += f", {blade_rate:.1f} pools/s {self.stream_blade} calc"
        return (
            summary
            + f" ({counts.get('ok', 0)} ok, {counts.get('failed', 0)} failed)"
        )


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
@click.option("--stream-blade", help="Show progress and ETA for one representative blade.")
@click.option("--status", "status_only", is_flag=True, help="Check detached remote state once.")
@click.option("--follow", is_flag=True, help="Follow a detached job, then retrieve it.")
@click.option("--retrieve", is_flag=True, help="Retrieve an already-complete remote job.")
@click.option("--stop", "stop_only", is_flag=True, help="Stop a detached remote job.")
def run_command(
    config: Path,
    output_dir: Path,
    transfer: bool,
    rebuild: bool,
    stream_blade: str | None,
    status_only: bool,
    follow: bool,
    retrieve: bool,
    stop_only: bool,
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
        if modes and not config_value.hosts:
            raise ConfigError("remote job controls require placement hosts")
        if stream_blade is not None and stream_blade not in config_value.hosts:
            raise ConfigError(f"--stream-blade is not a placement host: {stream_blade}")
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
            click.echo(grid_summary(config), err=True)
        if not modes and config_value.hosts:
            paths = run_remote_config(
                config,
                output_dir,
                transfer=transfer,
                rebuild=rebuild,
                stream_blade=stream_blade,
            )
        elif not modes:
            reporter = _ProgressReporter("run")
            paths = run_config(
                config,
                output_dir,
                progress_callback=reporter,
                lane_callback=reporter.lane,
                transfer=transfer,
                rebuild=rebuild,
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
@click.option("--stream-blade")
@click.option("--origin-workspace", required=True, type=click.Path(path_type=Path))
@click.option("--origin-config", required=True, type=click.Path(path_type=Path))
def cluster_worker_command(
    config: Path,
    output_dir: Path,
    stream_blade: str | None,
    origin_workspace: Path,
    origin_config: Path,
) -> None:
    """Run one already-staged grid from a coordinator blade."""
    config_value = RunConfig.from_toml(config)
    if stream_blade is not None and stream_blade not in config_value.hosts:
        raise click.ClickException(
            f"--stream-blade is not a placement host: {stream_blade}"
        )
    reporter = _ProgressReporter(
        "run",
        stream_blade=stream_blade,
        blade_index=(config_value.hosts.index(stream_blade) if stream_blade else 0),
        blade_count=max(1, len(config_value.hosts)),
        lanes_per_blade=max(1, len(config_value.numa_nodes)),
    )
    try:
        paths = run_config(
            config,
            output_dir,
            progress_callback=reporter,
            lane_callback=reporter.lane,
            prepared=True,
            origin_workspace=origin_workspace,
            origin_config=origin_config,
        )
    except (ConfigError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        reporter.close()
    payload = json.loads(paths.run_json.read_text())
    counts = payload.get("status_counts", {})
    click.echo(f"coordinator: {reporter.completion_summary(counts)}", err=True)
    click.echo(
        f"coordinator: wrote {paths.run_json} and {paths.results_npz}",
        err=True,
    )


@main.command("optimize")
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "output_dir", required=True,
              type=click.Path(file_okay=False, path_type=Path))
def optimize_command(config: Path, output_dir: Path) -> None:
    """Run Nevergrad from CONFIG through the same evaluator fleet."""
    from .optimize import OptimizationError, optimize_config

    reporter = _ProgressReporter("optimize")
    try:
        paths = optimize_config(config, output_dir, progress_callback=reporter)
    except (ConfigError, OptimizationError, OSError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        reporter.close()
    click.echo(f"wrote {paths.run_json} and {paths.results_npz}")


@main.command("heatmap")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--x", "x_axis")
@click.option("--y", "y_axis")
@click.option("--metric", "metrics", multiple=True,
              default=("apy_net_consistency_90d", "detach_energy_ungated", "max_7d_rel_price_diff"),
              show_default=True)
@click.option("--max-price-diff-bps", type=float, default=100.0, show_default=True)
@click.option("--max-skew-percent", type=float)
@click.option("--slippage-bps", type=float)
@click.option("--final-price-diff-bps", type=float)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--show/--no-show", default=True, show_default=True)
def heatmap_command(
    run_dir: Path, x_axis: str | None, y_axis: str | None,
    metrics: tuple[str, ...], max_price_diff_bps: float,
    max_skew_percent: float | None, slippage_bps: float | None,
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
