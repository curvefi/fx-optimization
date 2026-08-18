"""One coherent Click CLI for the curve-fx-sim orchestrator (`fxsim`).

Provides data verification, harness identity inspection, grid generation/execution/collection,
optimization preflight/run/status/collect, heatmap rendering, shiftclick replay,
trajectory plotting, and repository auditing.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence

import click


def _jsonable(value: Any) -> Any:
    """Recursively convert custom objects to JSON-serializable primitives."""
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return str(value.as_posix())
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _emit(value: Any) -> None:
    """Print structured JSON output."""
    click.echo(json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False))


def _fail(error: BaseException) -> None:
    """Format and raise a ClickException from any caught exception."""
    if isinstance(error, click.ClickException):
        raise error
    raise click.ClickException(str(error)) from error


def _metric_list(raw: str) -> list[str]:
    """Split a comma-separated metric list, dropping empty entries."""
    return [name.strip() for name in raw.split(",") if name.strip()]


def _pair_map(raw: str) -> dict[str, float]:
    """Parse comma-separated ``metric=value`` pairs."""
    pairs: dict[str, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise click.ClickException(f"invalid metric=value pair {part!r}")
        name, value = part.split("=", 1)
        name = name.strip()
        if not name:
            raise click.ClickException(f"invalid metric=value pair {part!r}")
        try:
            pairs[name] = float(value)
        except ValueError:
            raise click.ClickException(f"invalid numeric value in pair {part!r}")
    return pairs


# ---------------------------------------------------------------------------
# Top-level main CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="curve-fx-sim")
def main() -> None:
    """Reproducible Curve FX simulation, grid search, and optimization CLI."""


# ---------------------------------------------------------------------------
# Data commands
# ---------------------------------------------------------------------------


@main.group("data")
def data_group() -> None:
    """Verify checked-in market inputs and deterministic fixtures."""


@data_group.command("verify")
@click.option("--root", type=click.Path(path_type=Path, file_okay=False), default=None, help="Repository root.")
@click.option("--manifest", "manifest_path", type=click.Path(path_type=Path, dir_okay=False), default=None, help="Manifest path.")
def data_verify(root: Path | None, manifest_path: Path | None) -> None:
    """Verify all datasets declared by data/manifest.toml."""
    try:
        from .data import verify_data
        verified = verify_data(root=root, manifest_path=manifest_path)
        _emit({
            "status": "ok",
            "verified_datasets_count": len(verified),
            "datasets": [v.to_dict() for v in verified],
        })
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


# ---------------------------------------------------------------------------
# Harness commands
# ---------------------------------------------------------------------------


@main.group("harness")
def harness_group() -> None:
    """Inspect and verify evaluator binary identity and build capabilities."""


@harness_group.command("identity")
@click.argument("binary_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def harness_identity(binary_path: Path) -> None:
    """Query an evaluator binary for its attested build identity."""
    try:
        from .evaluation.identity import inspect_binary_identity
        identity = inspect_binary_identity(binary_path)
        _emit(identity.to_dict())
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


# ---------------------------------------------------------------------------
# Grid commands
# ---------------------------------------------------------------------------


@main.group("grid")
def grid_group() -> None:
    """Generate, execute, and collect finite parameter grid searches."""


@grid_group.command("generate")
@click.option("--pair", "pair_id", required=True, help="Pair specification path or ID.")
@click.option("--grid", "grid_id", required=True, help="Grid specification path or ID.")
@click.option("--scenario", "scenario_id", required=True, help="Scenario specification.")
@click.option("--policy", "policy_id", default=None, help="Optional policy specification.")
@click.option("--harness", "harness_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="Attested evaluator binary.")
@click.option("--run-id", default=None, help="Immutable run ID (generated if omitted).")
@click.option("--output-root", type=click.Path(path_type=Path, file_okay=False), default=Path("."), help="Repository root.")
def grid_generate(
    pair_id: str,
    grid_id: str,
    scenario_id: str,
    policy_id: str | None,
    harness_path: Path,
    run_id: str | None,
    output_root: Path,
) -> None:
    """Generate an immutable Cartesian grid run manifest."""
    try:
        from .artifacts.store import RunStore
        from .evaluation.identity import inspect_binary_identity
        from .grids.runner import compile_grid_run
        from .specs.grid import load_grid_spec
        from .specs.pair import load_pair_spec
        from .specs.scenario import load_scenario_spec
        from .specs.policy import load_policy_spec
        from .evaluation.identity import validate_evaluator_identity
        from .artifacts.tables import MetricProjection

        store = RunStore(root=output_root)
        pair_spec = load_pair_spec(pair_id, repository=store.root_dir)
        grid_spec = load_grid_spec(grid_id, repository=store.root_dir)
        scenario_spec = load_scenario_spec(scenario_id, repository=store.root_dir)
        identity = inspect_binary_identity(harness_path)
        effective_policy_id = policy_id or grid_spec.policy_id
        if grid_spec.policy_id and policy_id and grid_spec.policy_id != policy_id:
            raise click.ClickException("--policy does not match the grid policy_id")
        policy_spec = (
            load_policy_spec(effective_policy_id, repository=store.root_dir)
            if effective_policy_id
            else None
        )
        if policy_spec is None:
            validate_evaluator_identity(
                identity,
                expected_policy_id="twocrypto_native",
                expected_policy_source_sha256="none",
                expected_policy_parameter_count=0,
            )
        else:
            validate_evaluator_identity(
                identity,
                expected_policy_id=policy_spec.id,
                expected_policy_source_sha256=policy_spec.source_sha256,
                expected_policy_abi=policy_spec.policy_abi,
                expected_policy_parameter_count=len(policy_spec.parameters),
            )
        fields = identity.metric_fields
        compilation = compile_grid_run(
            grid_spec,
            run_id=run_id or f"grid_{pair_spec.id}_{grid_spec.id}",
            pair_spec=pair_spec,
            scenario_spec=scenario_spec,
            store=store,
            metric_projection=MetricProjection.from_fields(fields, projection_id="grid"),
            evaluator_identity=identity,
            policy_spec=policy_spec,
        )

        _emit({
            "status": "ok",
            "run_id": compilation.manifest["run_id"],
            "manifest_path": compilation.manifest_path.as_posix(),
            "run_dir": compilation.run_dir.as_posix(),
            "pool_count": len(compilation.points),
        })
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


@grid_group.command("run")
@click.argument("manifest_or_run_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--site", default="local", help="Site profile name or path (default 'local').")
@click.option("--blades", multiple=True, help="Specific blades to target for cluster execution.")
@click.option("--resume", is_flag=True, help="Resume incomplete execution.")
@click.option("--harness", "harness_path", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Explicit harness binary.")
@click.option("--chunk-size", type=int, default=None, help="Override block-cyclic chunk size.")
def grid_run(
    manifest_or_run_dir: Path,
    site: str,
    blades: tuple[str, ...],
    resume: bool,
    harness_path: Path | None,
    chunk_size: int | None,
) -> None:
    """Execute one resolved grid run across local cores or cluster blades."""
    try:
        from .execution import ExecutionBackend, load_site_profile

        manifest_file = manifest_or_run_dir
        if manifest_file.is_dir():
            manifest_file = manifest_file / "manifest.json"

        profile = load_site_profile(site)
        backend = ExecutionBackend(site_profile=profile, harness_binary=harness_path)
        summary = backend.run_grid(
            manifest_file,
            resume=resume,
            blades=list(blades) if blades else None,
            chunk_size=chunk_size,
        )
        _emit({
            "status": summary.status,
            "run_id": summary.run_id,
            "scope": summary.scope,
            "total_pools": summary.total_pools,
            "duration_seconds": summary.duration_seconds,
            "output_path": str(summary.output_path.as_posix()) if summary.output_path else None,
            "manifest_path": str(summary.manifest_path.as_posix()) if summary.manifest_path else None,
        })
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


@grid_group.command("collect")
@click.argument("manifest_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "output_file", type=click.Path(path_type=Path, dir_okay=False), default=None, help="Output JSON path.")
def grid_collect(manifest_file: Path, output_file: Path | None) -> None:
    """Strictly validate and merge shard results for an immutable grid run."""
    try:
        from .execution import collect_grid_results
        result = collect_grid_results(manifest_file, output_file=output_file)
        _emit({
            "status": "ok",
            "manifest": str(manifest_file.as_posix()),
            "output": str(result.as_posix()),
        })
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


# ---------------------------------------------------------------------------
# Optimization commands
# ---------------------------------------------------------------------------


@main.group("optimize")
def optimize_group() -> None:
    """Run and monitor adaptive pool parameter optimization searches."""


@optimize_group.command("preflight")
@click.argument("spec_path", type=click.Path(exists=True, path_type=Path))
def optimize_preflight(spec_path: Path) -> None:
    """Validate optimization configuration, lattice bounds, and initial seeds."""
    try:
        from .specs.optimization import load_optimization_spec
        from .specs.policy import load_policy_spec
        from .optimization.profiles import profile_from_policy_spec
        from .specs.common import repository_root

        root = repository_root(spec_path)
        spec = load_optimization_spec(spec_path, repository=root)
        policy = load_policy_spec(spec.policy_id, repository=root)
        profile = profile_from_policy_spec(policy, spec.parameter_space)
        _emit({
            "status": "ok",
            "optimization_id": spec.id,
            "algorithm": spec.algorithm,
            "pair": spec.pair_id,
            "policy_id": policy.id,
            "policy_source_sha256": policy.source_sha256,
            "dimensions": len(profile.bounds),
            "dense_parameter_count": profile.n_params(),
            "parameter_names": list(profile.parameter_names),
        })
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


@optimize_group.command("worker")
@click.argument("bundle_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--harness", "harness_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--root", "repository_root", type=click.Path(path_type=Path, file_okay=False), default=None)
@click.option("--out", "output_file", type=click.Path(path_type=Path, dir_okay=False), required=True)
def optimize_worker(
    bundle_path: Path,
    harness_path: Path,
    repository_root: Path | None,
    output_file: Path,
) -> None:
    """Evaluate one immutable optimization work bundle."""
    try:
        from .evaluation.client import SubprocessHarnessClient
        from .optimization.worker import OptimizationWorkBundle, evaluate_work_bundle

        bundle = OptimizationWorkBundle.from_json(bundle_path)
        client = SubprocessHarnessClient(
            harness_path,
            repository=(repository_root or bundle_path.parent).resolve(),
        )
        result = evaluate_work_bundle(bundle, client)
        result.to_json(output_file)
        client.close()
        _emit({"status": "ok", "bundle_id": result.bundle_id, "result": str(output_file)})
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


@optimize_group.command("run")
@click.argument("spec_path", type=click.Path(exists=True, path_type=Path))
@click.option("--site", default="local", help="Site profile name or path.")
@click.option("--blades", multiple=True, help="Specific blades to target for distributed execution.")
@click.option("--run-id", default=None, help="Immutable run ID.")
@click.option("--resume", is_flag=True, help="Resume incomplete optimization run.")
@click.option("--output-root", type=click.Path(path_type=Path, file_okay=False), default=Path("."))
@click.option("--harness", "harness_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="Explicit local evaluator binary.")
def optimize_run(
    spec_path: Path,
    site: str,
    blades: tuple[str, ...],
    run_id: str | None,
    resume: bool,
    output_root: Path,
    harness_path: Path | None,
) -> None:
    """Execute adaptive parameter optimization."""
    try:
        from .artifacts.store import RunStore
        from .optimization import run_optimization
        from .specs.optimization import load_optimization_spec

        store = RunStore(root=output_root)
        client = None
        if harness_path is not None:
            from .evaluation.client import SubprocessHarnessClient

            client = SubprocessHarnessClient(harness_path, repository=store.root_dir)
        spec = load_optimization_spec(spec_path)
        effective_run_id = run_id or f"opt_{spec.id}"
        result = run_optimization(
            spec,
            store=store,
            client=client,
            run_id=effective_run_id,
            resume=resume,
            site=site,
            blades=blades,
            repository=store.root_dir,
        )
        _emit({"status": "ok", "run_id": effective_run_id, "result": _jsonable(result)})
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


@optimize_group.command("status")
@click.argument("run_id_or_path")
def optimize_status(run_id_or_path: str) -> None:
    """Query current optimization progress and best candidate checkpoint."""
    try:
        from .optimization import status_optimization

        _emit(status_optimization(run_id_or_path).to_dict())
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


@optimize_group.command("collect")
@click.argument("run_id_or_path")
@click.option("--out", "output_file", type=click.Path(path_type=Path, dir_okay=False), default=None)
def optimize_collect(run_id_or_path: str, output_file: Path | None) -> None:
    """Collect and finalize optimization trajectory and candidate ranking."""
    try:
        from .optimization import collect_optimization

        result = collect_optimization(run_id_or_path)
        payload = {"status": "ok", **result.to_dict()}
        if output_file is not None:
            from .artifacts.io import atomic_write_json

            atomic_write_json(output_file, payload)
            payload["output"] = output_file.as_posix()
        _emit(payload)
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


# ---------------------------------------------------------------------------
# Replay and shiftclick commands
# ---------------------------------------------------------------------------


@main.group("replay")
def replay_group() -> None:
    """Replay candidate selections with full observation and atomic trace sidecars."""


@replay_group.command("shiftclick")
@click.argument("spec_path", type=click.Path(exists=True, path_type=Path))
@click.option("--out", "output_dir", type=click.Path(path_type=Path, file_okay=False), default=None)
@click.option("--harness", "harness_path", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--site", default="local", help="Local or SSH site profile.")
@click.option("--blades", multiple=True, help="Exactly one blade for remote replay.")
def replay_shiftclick(
    spec_path: Path,
    output_dir: Path | None,
    harness_path: Path | None,
    site: str,
    blades: tuple[str, ...],
) -> None:
    """Run one strict full-trace shiftclick replay and economic comparison."""
    try:
        from .artifacts.store import RunStore
        from .evaluation.client import SubprocessHarnessClient
        from .shiftclick import run_shiftclick
        from .specs.common import repository_root
        from .specs.shiftclick import load_shiftclick_spec

        spec = load_shiftclick_spec(spec_path)
        root = repository_root()
        store = RunStore(root)
        if site != "local" or blades:
            from .execution.site import load_site_profile
            from .shiftclick import run_remote_shiftclick

            if output_dir is not None:
                raise click.ClickException("--out is not supported for remote shiftclick")
            profile = load_site_profile(site, root=root)
            targets = list(blades) or list(profile.cluster.blades)
            if len(targets) != 1:
                raise click.ClickException("remote shiftclick requires exactly one --blades target")
            result = run_remote_shiftclick(
                spec,
                spec_path=spec_path,
                store=store,
                site=profile,
                blade=targets[0],
            )
            _emit({"status": "ok", **result.to_dict()})
            return
        if harness_path is None:
            raise click.ClickException("--harness is required for a full-trace replay")
        client = SubprocessHarnessClient(harness_path, repository=root)
        result = run_shiftclick(
            spec,
            store=store,
            client=client,
            output_dir=output_dir,
        )
        _emit({"status": "ok", **result.to_dict()})
    except Exception as exc:  # noqa: BLE001
        _fail(exc)

# ---------------------------------------------------------------------------
# Plotting commands
# ---------------------------------------------------------------------------


@main.group("plot")
def plot_group() -> None:
    """Render heatmaps, parameter response surfaces, and state trajectories."""


@plot_group.command("heatmap")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--metric", default=None, help="Metric key to plot on color axis (single view).")
@click.option(
    "--metrics",
    default=None,
    help="Comma-separated metric names for a multi-metric tiled heatmap; when omitted the single --metric view is used.",
)
@click.option("--ncol", type=int, default=2, help="Tiles per row for --metrics (default 2).")
@click.option(
    "--log-axis",
    "log_axis_values",
    multiple=True,
    help="Log-scale this grid axis's display ticks; repeatable and comma-separated. Requires --metrics.",
)
@click.option(
    "--max-pricethr",
    type=float,
    default=None,
    help="Mask cells whose max_7d_rel_price_diff exceeds this threshold (bps; the metric is a fraction, 1.0 = 100%, so bps/10000). Requires --metrics.",
)
@click.option(
    "--slipthr",
    type=float,
    default=None,
    help="Lower bound (bps) of the tw_real_slippage_1pct pass window; cells below it are masked (fraction units, bps/10000). Requires --metrics.",
)
@click.option(
    "--slipthr-max",
    type=float,
    default=None,
    help="Upper bound (bps) of the tw_real_slippage_1pct pass window; cells above it are masked (fraction units, bps/10000). Requires --metrics.",
)
@click.option("--x", "x_axis", default=None, help="Parameter name for X axis.")
@click.option("--y", "y_axis", default=None, help="Parameter name for Y axis.")
@click.option("--out", "output_file", type=click.Path(path_type=Path, dir_okay=False), default=None, help="Saved image path.")
def plot_heatmap(
    run_dir: Path,
    metric: str | None,
    metrics: str | None,
    ncol: int,
    log_axis_values: tuple[str, ...],
    max_pricethr: float | None,
    slipthr: float | None,
    slipthr_max: float | None,
    x_axis: str | None,
    y_axis: str | None,
    output_file: Path | None,
) -> None:
    """Render a parameter heatmap from an attested evaluation table.

    Always writes the PNG slice and the fxsim_heatmap_state_v1 sidecar
    (carrying the full N-D data pointers).  With --metrics renders one tile per
    metric in an --ncol grid sharing the same slice and validity mask; masked
    cells are NaN on every tile.  When a display-capable matplotlib backend is
    active the interactive slider figure is shown automatically.
    """
    try:
        from .artifacts.attestation import load_attested_evaluation_table
        from .artifacts.manifest import load_manifest
        from .plotting.heatmap import (
            HeatmapDataset,
            HeatmapState,
            MatplotlibHeatmapTilesView,
            MatplotlibHeatmapView,
            MaskSpec,
            interactive_backend_active,
        )
        from .plotting.masked_metrics import MASKED_METRIC_SOURCES

        manifest = load_manifest(run_dir / "manifest.json")
        table, table_path = load_attested_evaluation_table(
            manifest,
            run_dir=run_dir,
        )
        destination = output_file or run_dir / "heatmap.png"
        dataset = HeatmapDataset.from_table(table)
        tile_metrics = _metric_list(metrics) if metrics is not None else None
        if slipthr is not None and slipthr_max is not None and slipthr > slipthr_max:
            raise click.ClickException("--slipthr cannot exceed --slipthr-max")
        log_axes = [name for raw in log_axis_values for name in raw.split(",") if name.strip()]
        if tile_metrics is None:
            if log_axes or max_pricethr is not None or slipthr is not None or slipthr_max is not None:
                raise click.ClickException(
                    "--ncol/--log-axis/--max-pricethr/--slipthr/--slipthr-max require --metrics"
                )
            state = HeatmapState.default(
                dataset,
                metric=metric,
                x_axis=x_axis,
                y_axis=y_axis,
            )
            state.source = table_path.name
            view = MatplotlibHeatmapView(dataset, state)
            try:
                image, sidecar = view.save(destination)
            finally:
                if interactive_backend_active():
                    view.show()
                view.close()
            _emit({
                "status": "ok",
                "run_dir": run_dir.as_posix(),
                "results_source": table_path.as_posix(),
                "metric": state.metric,
                "output": image.as_posix(),
                "state": sidecar.as_posix(),
            })
            return
        available = tuple(dataset.metrics) + tuple(sorted(MASKED_METRIC_SOURCES))
        missing = [name for name in tile_metrics if name not in available]
        if missing:
            raise click.ClickException(
                f"unknown heatmap metric(s): {', '.join(missing)}; "
                f"available metrics: {', '.join(available)}"
            )
        mask = MaskSpec(
            max_price_diff_bps=max_pricethr,
            slippage_thr_bps=slipthr,
            slippage_thr_max_bps=slipthr_max,
        )
        view = MatplotlibHeatmapTilesView(
            dataset,
            tiles=tile_metrics,
            x_axis=x_axis,
            y_axis=y_axis,
            ncol=ncol,
            log_axes=log_axes,
            mask=mask,
        )
        view.state.source = table_path.name
        try:
            image, sidecar = view.save(destination)
        finally:
            if interactive_backend_active():
                view.show()
            view.close()
        _emit({
            "status": "ok",
            "run_dir": run_dir.as_posix(),
            "results_source": table_path.as_posix(),
            "metric": view.state.tiles[0],
            "metrics": list(view.state.tiles),
            "ncol": view.state.ncol,
            "log_axes": list(view.state.log_axes),
            "output": image.as_posix(),
            "state": sidecar.as_posix(),
        })
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


@plot_group.command("trajectory")
@click.argument("diagnostic_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out", "output_file", type=click.Path(path_type=Path, dir_okay=False), default=None)
def plot_trajectory(diagnostic_dir: Path, output_file: Path | None) -> None:
    """Render state trajectory from one manifest-attested trace."""
    try:
        from .artifacts.attestation import find_attested_artifact
        from .artifacts.manifest import load_manifest
        from .plotting.trajectory import render_trajectory

        manifest = load_manifest(
            diagnostic_dir / "manifest.json",
            expected_kind="shiftclick",
        )
        trace_path = find_attested_artifact(
            manifest,
            run_dir=diagnostic_dir,
            kind="trace",
        )
        destination = output_file or diagnostic_dir / "trajectory.png"
        try:
            from .plotting.shiftclick_view import render_shiftclick_figure

            actions = None
            try:
                from .artifacts.attestation import find_attested_artifact

                actions = find_attested_artifact(
                    manifest,
                    run_dir=diagnostic_dir,
                    kind="actions",
                )
            except Exception:
                actions = None
            figure = render_shiftclick_figure(
                trace_path,
                actions,
                title=diagnostic_dir.name,
                fee_source="both",
            )
            figure.savefig(destination, dpi=150)
            image, state = destination, destination.with_suffix(".state.json")
            state.write_text(
                '{"schema_version": "fxsim_shiftclick_render_v1", '
                f'"source": "{trace_path.name}"}}',
                encoding="utf-8",
            )
        except Exception:
            from .plotting.trajectory import render_trajectory

            image, state = render_trajectory(trace_path, destination)
        _emit({
            "status": "ok",
            "diagnostic_dir": diagnostic_dir.as_posix(),
            "trace": trace_path.as_posix(),
            "output": image.as_posix(),
            "state": state.as_posix(),
        })
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


# ---------------------------------------------------------------------------
# Analysis commands
# ---------------------------------------------------------------------------


@main.group("analyze")
def analyze_group() -> None:
    """Analyze evaluation tables, rank candidates, and check economic drift."""


@analyze_group.command("rank")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--metric", default="score", help="Metric key to sort by.")
@click.option("--top", type=int, default=10, help="Number of top candidates.")
@click.option("--ascending", is_flag=True, help="Sort ascending instead of descending.")
def analyze_rank(run_dir: Path, metric: str, top: int, ascending: bool) -> None:
    """Rank candidates from the common attested evaluation table."""
    try:
        from .artifacts.attestation import load_attested_evaluation_table
        from .artifacts.manifest import load_manifest
        from .grids.analysis import rank_evaluations

        if top < 0:
            raise click.ClickException("--top must be nonnegative")
        manifest = load_manifest(run_dir / "manifest.json")
        table, _ = load_attested_evaluation_table(manifest, run_dir=run_dir)
        ranked = rank_evaluations(
            table,
            ascending=(metric,) if ascending else (),
            descending=() if ascending else (metric,),
            top=top,
        )
        _emit({
            "status": "ok",
            "run_dir": run_dir.as_posix(),
            "total_rows": len(table.rows),
            "top": [item.row.to_dict() | {
                "weighted_rank": item.weighted_rank,
                "metric_ranks": dict(item.metric_ranks),
            } for item in ranked],
            "metric": metric,
        })
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


@analyze_group.command("maxima")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--metric", default=None, help="Metric key for local-maximum enumeration (default first numeric).")
@click.option("--local", is_flag=True, help="Enumerate local maxima of --metric over the grid neighborhood.")
@click.option("--connectivity", type=click.Choice(["axis", "full"]), default="full", help="Neighborhood definition for --local (default full).")
@click.option("--axis", "axis_names", multiple=True, help="Grid axis keys active in the neighborhood; repeatable (default all).")
@click.option("--desc-metrics", default="", help="Comma-separated metrics to maximize in ranked mode.")
@click.option("--asc-metrics", default="", help="Comma-separated metrics to minimize in ranked mode.")
@click.option("--weights", default="", help="Comma-separated metric=weight overrides (default 1.0).")
@click.option("--desc-thr", "desc_thresholds", default="", help="Comma-separated metric=threshold; cells meeting it rank best.")
@click.option("--asc-thr", "asc_thresholds", default="", help="Comma-separated metric=threshold; cells meeting it rank best.")
@click.option("--top", type=int, default=20, help="Number of results (<=0 for all).")
@click.option("--max-price-diff-bps", type=float, default=None, help="Mask cells whose price-diff metric exceeds this threshold (bps).")
@click.option("--max-skew-percent", type=float, default=None, help="Mask cells whose max_7d_skew exceeds this threshold (percent).")
@click.option("--max-final-price-diff-bps", type=float, default=None, help="Mask cells whose final price-diff metric exceeds this threshold (bps).")
def analyze_maxima(
    run_dir: Path,
    metric: str | None,
    local: bool,
    connectivity: str,
    axis_names: tuple[str, ...],
    desc_metrics: str,
    asc_metrics: str,
    weights: str,
    desc_thresholds: str,
    asc_thresholds: str,
    top: int,
    max_price_diff_bps: float | None,
    max_skew_percent: float | None,
    max_final_price_diff_bps: float | None,
) -> None:
    """Enumerate grid local maxima or rank cells by weighted multi-metric scores."""
    try:
        from .analysis.maxima import (
            coordinates_at,
            find_local_maxima,
            ranked_maxima,
        )
        from .artifacts.attestation import load_attested_evaluation_table
        from .artifacts.manifest import load_manifest
        from .grids.analysis import first_numeric_metric
        from .plotting.heatmap import HeatmapDataset, MaskSpec

        manifest = load_manifest(run_dir / "manifest.json")
        table, _ = load_attested_evaluation_table(manifest, run_dir=run_dir)
        dataset = HeatmapDataset.from_table(table)
        mask = MaskSpec(
            max_price_diff_bps=max_price_diff_bps,
            max_skew_percent=max_skew_percent,
            max_final_price_diff_bps=max_final_price_diff_bps,
        )
        axis_keys = dataset.axis_keys
        if axis_names:
            unknown = sorted(set(axis_names) - set(axis_keys))
            if unknown:
                raise click.ClickException(f"unknown grid axis: {', '.join(unknown)}")
            active_axes = tuple(dataset.axis_index(name) for name in axis_names)
        else:
            active_axes = None
        if local:
            resolved_metric = metric or first_numeric_metric(dataset.metrics)
            if resolved_metric not in dataset.metrics:
                raise click.ClickException(f"unknown grid metric {resolved_metric!r}")
            values = dataset.metric_array(resolved_metric, mask)
            maxima = find_local_maxima(values, axes=active_axes, connectivity=connectivity)
            points = [
                {
                    "rank": index,
                    "grid_indices": list(location),
                    "metric": float(values[location]),
                    "candidate_id": str(dataset.candidate_ids[location]),
                    "ordinal": int(dataset.ordinals[location]),
                    "coordinates": coordinates_at(dataset, location),
                }
                for index, location in enumerate(maxima, start=1)
            ]
            _emit({
                "status": "ok",
                "mode": "local",
                "run_dir": run_dir.as_posix(),
                "metric": resolved_metric,
                "connectivity": connectivity,
                "axes": list(axis_names) if axis_names else list(axis_keys),
                "local_maxima_count": len(points),
                "local_maxima": points,
            })
            return
        descending_list = _metric_list(desc_metrics)
        ascending_list = _metric_list(asc_metrics)
        if not descending_list and not ascending_list:
            # Bare `fxsim analyze maxima RUN_DIR` ranks descending by the
            # default metric, mirroring the single-metric tools.
            resolved_metric = metric or first_numeric_metric(dataset.metrics)
            if resolved_metric not in dataset.metrics:
                raise click.ClickException(f"unknown grid metric {resolved_metric!r}")
            descending_list = [resolved_metric]
        resolved_weights = _pair_map(weights)
        thresholds = {**_pair_map(desc_thresholds), **_pair_map(asc_thresholds)}
        ranked = ranked_maxima(
            dataset,
            descending=descending_list,
            ascending=ascending_list,
            weights=resolved_weights or None,
            thresholds=thresholds or None,
            top=top if top > 0 else None,
            mask=mask,
        )
        metric_names = [*descending_list, *ascending_list]
        _emit({
            "status": "ok",
            "mode": "ranked",
            "run_dir": run_dir.as_posix(),
            "descending": list(descending_list),
            "ascending": list(ascending_list),
            "weights": {name: resolved_weights.get(name, 1.0) for name in metric_names},
            "ranked_count": len(ranked),
            "ranked": [item.to_dict() for item in ranked],
        })
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


# ---------------------------------------------------------------------------
# Repository audit
# ---------------------------------------------------------------------------


@main.group("repo")
def repo_group() -> None:
    """Audit repository tracked paths against artifact exclusion rules."""


@main.command("verify")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def verify_run(run_dir: Path) -> None:
    """Re-verify every attested artifact digest of one run directory."""
    try:
        from .artifacts.attestation import (
            load_attested_evaluation_table,
            verify_manifest_artifacts,
        )
        from .artifacts.manifest import load_manifest

        manifest = load_manifest(run_dir / "manifest.json")
        artifacts = verify_manifest_artifacts(manifest, run_dir=run_dir)
        run_kind = manifest.get("run_kind")
        if run_kind in {"grid", "optimization"}:
            table, table_path = load_attested_evaluation_table(
                manifest, run_dir=run_dir, verify_digest=True
            )
            _emit(
                {
                    "status": "ok",
                    "run_id": manifest.get("run_id"),
                    "run_kind": run_kind,
                    "artifacts_verified": len(artifacts),
                    "table": table_path.name,
                    "rows": len(table),
                }
            )
        else:
            _emit(
                {
                    "status": "ok",
                    "run_id": manifest.get("run_id"),
                    "run_kind": run_kind,
                    "artifacts_verified": len(artifacts),
                }
            )
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


@repo_group.command("audit")
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None)
def repo_audit(root: Path | None) -> None:
    """Verify that no forbidden historical binaries or run outputs are tracked."""
    try:
        import subprocess

        from .specs.common import repository_root

        repo = root.resolve() if root is not None else repository_root()
        proc = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "-z"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        tracked = [path for path in proc.stdout.decode().split("\0") if path]
        forbidden_parts = {"runs", "build", "dist", ".venv", ".pytest_cache", "__pycache__"}
        forbidden = sorted(
            path
            for path in tracked
            if forbidden_parts.intersection(Path(path).parts)
            or Path(path).suffix in {".pyc", ".pyo"}
        )
        payload = {
            "status": "ok" if not forbidden else "error",
            "audit": "clean" if not forbidden else "forbidden_tracked_artifacts",
            "forbidden_artifacts_found": len(forbidden),
            "paths": forbidden,
        }
        if forbidden:
            raise click.ClickException(json.dumps(payload, sort_keys=True))
        _emit(payload)
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


if __name__ == "__main__":  # pragma: no cover
    main()
