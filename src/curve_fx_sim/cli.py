"""One coherent Click CLI for the curve-fx-sim orchestrator (`fxsim`).

Provides data verification, grid generation/execution/collection,
optimization preflight/run/status/collect, heatmap rendering, shiftclick replay,
trajectory plotting, and repository auditing.
"""

from __future__ import annotations

import json
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence

import click

from .specs.common import ProjectContext


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


def _project_context() -> ProjectContext:
    """Return the explicit root context established by the top-level command."""
    context = click.get_current_context().find_root().obj
    if not isinstance(context, ProjectContext):
        raise click.ClickException("project context is unavailable")
    return context


# ---------------------------------------------------------------------------
# Top-level main CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="curve-fx-sim")
@click.option(
    "--project-root",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("."),
    show_default=True,
    help="Project root containing configs/, data/, and policies/.",
)
@click.option(
    "--run-root",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Run output directory (default: PROJECT_ROOT/runs).",
)
@click.pass_context
def main(ctx: click.Context, project_root: Path, run_root: Path | None) -> None:
    """Reproducible Curve FX simulation, grid search, and optimization CLI."""
    ctx.obj = ProjectContext.from_root(project_root, run_root=run_root)


# ---------------------------------------------------------------------------
# Data commands
# ---------------------------------------------------------------------------


@main.group("data")
def data_group() -> None:
    """Verify checked-in market inputs and deterministic fixtures."""


@data_group.command("verify")
@click.option("--manifest", "manifest_path", type=click.Path(path_type=Path, dir_okay=False), default=None, help="Manifest path.")
def data_verify(manifest_path: Path | None) -> None:
    """Verify all datasets declared by data/manifest.toml."""
    try:
        from .data import verify_data
        verified = verify_data(
            root=_project_context().project_root,
            manifest_path=manifest_path,
        )
        _emit({
            "status": "ok",
            "verified_datasets_count": len(verified),
            "datasets": [v.to_dict() for v in verified],
        })
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Evaluator commands
# ---------------------------------------------------------------------------


@main.group("evaluator")
def evaluator_group() -> None:
    """Build and verify explicitly selected attested evaluator artifacts."""


def _context_path(value: Path, root: Path) -> Path:
    """Resolve a CLI path relative to an explicit context root."""
    return value if value.is_absolute() else root / value


def _evaluator_payload(artifact: Any, *, numeric_mode: str | None = None) -> dict[str, Any]:
    """Select compact attestation fields without exposing the full description."""
    description = artifact.description
    policy = description["policy"]
    build = description["build"]
    mode = numeric_mode
    if mode is None:
        mode = "f64" if build["numeric_mode"] == "double" else "longdouble"
    return {
        "status": "ok",
        "artifact_dir": artifact.receipt_path.parent.as_posix(),
        "receipt_path": artifact.receipt_path.as_posix(),
        "artifact_sha256": artifact.artifact_sha256,
        "binary_path": artifact.binary_path.as_posix(),
        "binary_sha256": artifact.binary_sha256,
        "build_spec_sha256": artifact.build_spec_sha256,
        "policy_id": policy["id"],
        "policy_source_sha256": policy["source_sha256"],
        "policy_parameter_count": policy["parameter_count"],
        "numeric_mode": mode,
        "real_digits": build["real_digits"],
        "parameter_schema_sha256": description["parameter_schema_sha256"],
    }


@evaluator_group.command("build")
@click.option("--pool-root", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--harness-root", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--artifact-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--policy", "policy_id", default=None)
@click.option("--numeric-mode", type=click.Choice(["f64", "longdouble"]), default="longdouble", show_default=True)
@click.option("--build-type", default="Release", show_default=True)
@click.option("--ipo", "enable_ipo", is_flag=True)
@click.option("--native-tuning", is_flag=True)
def evaluator_build(
    pool_root: Path,
    harness_root: Path,
    artifact_dir: Path,
    policy_id: str | None,
    numeric_mode: str,
    build_type: str,
    enable_ipo: bool,
    native_tuning: bool,
) -> None:
    """Build a fresh attested evaluator artifact from explicit sources."""
    try:
        from .evaluation.builder import BuildSpec, build_evaluator
        from .specs.policy import load_policy_spec

        context = _project_context()
        resolved_pool = _context_path(pool_root, context.project_root)
        resolved_harness = _context_path(harness_root, context.project_root)
        resolved_artifact = _context_path(artifact_dir, context.run_root)
        values: dict[str, Any] = {
            "pool_root": resolved_pool,
            "harness_root": resolved_harness,
            "numeric_mode": numeric_mode,
            "build_type": build_type,
            "enable_ipo": enable_ipo,
            "native_tuning": native_tuning,
        }
        if policy_id is not None:
            policy = load_policy_spec(policy_id, repository=context.project_root)
            header = Path(policy.header_file)
            values.update(
                policy_header=_context_path(header, context.project_root),
                policy_id=policy.id,
                policy_abi=policy.policy_abi,
                policy_expected_sha256=policy.source_sha256,
            )
        spec = BuildSpec(**values)
        artifact = build_evaluator(spec, resolved_artifact)
        _emit(_evaluator_payload(artifact, numeric_mode=spec.numeric_mode))
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


@evaluator_group.command("verify")
@click.argument("artifact_dir", type=click.Path(path_type=Path, file_okay=False))
def evaluator_verify(artifact_dir: Path) -> None:
    """Re-verify one explicitly selected evaluator artifact."""
    try:
        from .evaluation.builder import load_evaluator_artifact

        context = _project_context()
        resolved_artifact = _context_path(artifact_dir, context.run_root)
        artifact = load_evaluator_artifact(resolved_artifact)
        _emit(_evaluator_payload(artifact))
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
@click.option("--artifact-dir", type=click.Path(file_okay=False, path_type=Path), required=True, help="Selected evaluator artifact directory.")
@click.option("--run-id", default=None, help="Immutable run ID (generated if omitted).")
def grid_generate(
    pair_id: str,
    grid_id: str,
    scenario_id: str,
    artifact_dir: Path,
    run_id: str | None,
) -> None:
    """Generate an immutable Cartesian grid run manifest."""
    try:
        from .artifacts.store import RunStore
        from .grids.runner import compile_grid_run
        from .specs.grid import load_grid_spec
        from .specs.pair import load_pair_spec
        from .specs.scenario import load_scenario_spec
        from .artifacts.tables import MetricProjection

        context = _project_context()
        store = RunStore(context)
        pair_spec = load_pair_spec(pair_id, repository=context.project_root)
        grid_spec = load_grid_spec(grid_id, repository=context.project_root)
        scenario_spec = load_scenario_spec(scenario_id, repository=context.project_root)
        from .evaluation.selected import SelectedEvaluator
        from .evaluation.session import LocalSessionMaterialization

        selected = SelectedEvaluator.load(_context_path(artifact_dir, context.run_root))
        identity = selected.verified_evaluator
        effective_run_id = run_id or f"grid_{pair_spec.id}_{grid_spec.id}"
        with tempfile.TemporaryDirectory(prefix="fxsim-grid-session-") as temporary:
            materialization = LocalSessionMaterialization.from_scenario(
                scenario_spec, repository=context.project_root,
                manifest_root=Path(temporary), session_id=f"{effective_run_id}_baseline",
            ).validated()
            compilation = compile_grid_run(
                grid_spec,
                run_id=effective_run_id,
                pair_spec=pair_spec,
                scenario_spec=scenario_spec,
                store=store,
                metric_projection=MetricProjection.from_fields(identity.metric_fields, projection_id="grid"),
                selected_evaluator=selected,
                open_session=materialization.baseline_open_session_fields,
                scenario=materialization.closure,
            )

        _emit({
            "status": "ok",
            "run_id": compilation.manifest["run_id"],
            "manifest_path": compilation.manifest_path.as_posix(),
            "run_dir": compilation.run_dir.as_posix(),
            "pool_count": len(compilation.plan),
        })
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


@grid_group.command("run")
@click.argument("manifest_or_run_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--site", default="local", help="Site profile name or path (default 'local').")
@click.option("--blades", multiple=True, help="Specific blades to target for cluster execution.")
@click.option("--resume", is_flag=True, help="Resume incomplete execution.")
@click.option("--chunk-size", type=int, default=None, help="Override block-cyclic chunk size.")
def grid_run(
    manifest_or_run_dir: Path,
    site: str,
    blades: tuple[str, ...],
    resume: bool,
    chunk_size: int | None,
) -> None:
    """Execute one resolved grid run across local cores or cluster blades."""
    try:
        from .execution import ExecutionBackend, load_site_profile

        manifest_file = manifest_or_run_dir
        if manifest_file.is_dir():
            manifest_file = manifest_file / "manifest.json"

        context = _project_context()
        profile = load_site_profile(site, root=context.project_root)
        backend = ExecutionBackend(site_profile=profile)
        summary = backend.run_grid(
            manifest_file,
            resume=resume,
            blades=list(blades) if blades else None,
            chunk_size=chunk_size,
            repository=context.project_root,
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
@click.option("--out", "output_file", type=click.Path(path_type=Path, dir_okay=False), default=None, help="Output NPZ path.")
def grid_collect(manifest_file: Path, output_file: Path | None) -> None:
    """Strictly validate and merge shard results for an immutable grid run."""
    try:
        from .execution import collect_grid_results
        result = collect_grid_results(manifest_file, output_path=output_file)
        _emit({
            "status": "ok",
            "manifest": str(manifest_file.as_posix()),
            "output": str(result.as_posix()),
        })
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


@main.group("worker")
def worker_group() -> None:
    """Run narrow artifact-authoritative worker protocols."""


@worker_group.command("package-identity")
def worker_package_identity() -> None:
    """Print the project package identity used by execution closures."""
    from .execution.shared_nfs import package_identity_sha256
    click.echo(package_identity_sha256(_project_context().project_root))


@worker_group.command("grouped")
@click.argument("request_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "output_path", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--remote-run-root", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--blade", default=None, hidden=True)
def worker_grouped(request_path: Path, output_path: Path, remote_run_root: Path,
                   blade: str | None) -> None:
    """Execute one canonical grouped request against its run-local evaluator."""
    try:
        from .execution.grouped_remote import execute_grouped_work
        receipt = execute_grouped_work(request_path, output_path,
            remote_run_root=remote_run_root,
            repository=_project_context().project_root, blade=blade)
        _emit({"status": "ok", "request_sha256": receipt.request_sha256,
               "result": output_path.as_posix()})
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


@worker_group.command("grid")
@click.argument("request_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "output_path", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--remote-run-root", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--blade", default=None, hidden=True)
def worker_grid(request_path: Path, output_path: Path, remote_run_root: Path,
                blade: str | None) -> None:
    """Execute one compact Cartesian-plan request."""
    try:
        from .execution.grouped_remote import execute_grid_work
        receipt = execute_grid_work(
            request_path, output_path, remote_run_root=remote_run_root,
            repository=_project_context().project_root, blade=blade,
        )
        _emit({"status": "ok", "request_sha256": receipt.request_sha256,
               "result": output_path.as_posix()})
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


# ---------------------------------------------------------------------------
# Optimization commands
# ---------------------------------------------------------------------------


@main.group("optimize")
def optimize_group() -> None:
    """Run and monitor adaptive pool parameter optimization searches."""


@optimize_group.command("preflight")
@click.argument("spec_ref")
@click.option("--artifact-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
def optimize_preflight(spec_ref: str, artifact_dir: Path) -> None:
    """Validate optimization configuration, lattice bounds, and initial seeds."""
    try:
        from .specs.optimization import load_optimization_spec
        context = _project_context()
        root = context.project_root
        from .evaluation.selected import SelectedEvaluator
        from .evaluation.session import LocalSessionMaterialization
        from .optimization.search import SearchLayout
        from .specs.scenario import load_scenario_spec

        selected = SelectedEvaluator.load(_context_path(artifact_dir, context.run_root))
        spec = load_optimization_spec(spec_ref, repository=root)
        policy = selected.policy_identity
        if spec.policy_id != policy["id"]:
            raise ValueError(
                f"optimization policy_id {spec.policy_id!r} does not match "
                f"selected evaluator policy {policy['id']!r}"
            )
        scenario = load_scenario_spec(spec.scenarios[0], repository=root)
        template_json = None
        if scenario.template_path is not None:
            with (root / scenario.template_path).open("r", encoding="utf-8") as stream:
                template_json = json.load(stream)
        with tempfile.TemporaryDirectory(prefix="fxsim-opt-preflight-") as temporary:
            materialization = LocalSessionMaterialization.from_scenario(
                scenario, repository=root, manifest_root=Path(temporary),
                session_id=f"preflight_{spec.id}",
            ).validated()
            layout = SearchLayout.from_schema(
                selected.compiler.schema, spec.parameter_space, template_json,
                materialization.baseline_open_session_fields,
            )
        _emit({
            "status": "ok", "optimization_id": spec.id, "algorithm": spec.algorithm,
            "pair": spec.pair_id, "policy_id": policy["id"],
            "policy_source_sha256": policy["source_sha256"],
            "artifact_sha256": selected.artifact_sha256,
            "parameter_schema_sha256": selected.parameter_schema_sha256,
            "dimensions": len(layout.dimensions),
            "parameter_names": [item.name for item in layout.dimensions],
            "geometry_sha256": layout.sha256, "default_vector": layout.default_vector,
        })
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


@optimize_group.command("run")
@click.argument("spec_ref")
@click.option("--site", default="local", help="Site profile name or path.")
@click.option("--blades", multiple=True, help="Specific blades to target for distributed execution.")
@click.option("--run-id", default=None, help="Immutable run ID.")
@click.option("--resume", is_flag=True, help="Resume incomplete optimization run.")
@click.option("--artifact-dir", type=click.Path(file_okay=False, path_type=Path), required=True, help="Verified evaluator artifact for schema-named optimization.")
def optimize_run(
    spec_ref: str,
    site: str,
    blades: tuple[str, ...],
    run_id: str | None,
    resume: bool,
    artifact_dir: Path,
) -> None:
    """Execute adaptive parameter optimization."""
    try:
        from .artifacts.store import RunStore
        from .optimization import run_optimization
        from .specs.optimization import load_optimization_spec

        context = _project_context()
        store = RunStore(context)
        from .evaluation.selected import SelectedEvaluator

        selected = SelectedEvaluator.load(_context_path(artifact_dir, context.run_root))
        spec = load_optimization_spec(spec_ref, repository=context.project_root)
        effective_run_id = run_id or f"opt_{spec.id}"
        result = run_optimization(
            spec,
            store=store,
            run_id=effective_run_id,
            resume=resume,
            site=site,
            blades=blades,
            repository=context.project_root,
            selected_evaluator=selected,
        )
        _emit({"status": "ok", "run_id": effective_run_id, "result": _jsonable(result)})
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


@optimize_group.command("status")
@click.argument("run_id_or_path")
def optimize_status(run_id_or_path: str) -> None:
    """Query current optimization progress and best candidate checkpoint."""
    try:
        from .artifacts.store import RunStore
        from .optimization import status_optimization

        context = _project_context()
        _emit(
            status_optimization(
                run_id_or_path,
                store=RunStore(context),
                repository=context.project_root,
            ).to_dict()
        )
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


@optimize_group.command("collect")
@click.argument("run_id_or_path")
@click.option("--out", "output_file", type=click.Path(path_type=Path, dir_okay=False), default=None)
def optimize_collect(run_id_or_path: str, output_file: Path | None) -> None:
    """Collect and finalize optimization trajectory and candidate ranking."""
    try:
        from .artifacts.store import RunStore
        from .optimization import collect_optimization

        context = _project_context()
        result = collect_optimization(
            run_id_or_path,
            store=RunStore(context),
            repository=context.project_root,
        )
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
    """Replay candidate selections with full observation and attested NPZ replay traces."""


@replay_group.command("shiftclick")
@click.argument("spec_path", type=click.Path(exists=True))
@click.option("--out", "output_dir", type=click.Path(path_type=Path, file_okay=False), default=None)
@click.option("--site", default="local", help="Local or SSH site profile.")
@click.option("--blades", multiple=True, help="Exactly one blade for remote replay.")
def replay_shiftclick(
    spec_path: str,
    output_dir: Path | None,
    site: str,
    blades: tuple[str, ...],
) -> None:
    """Run one strict full-trace shiftclick replay and economic comparison."""
    try:
        from .artifacts.store import RunStore
        from .shiftclick import run_shiftclick
        from .specs.shiftclick import load_shiftclick_spec

        context = _project_context()
        root = context.project_root
        spec = load_shiftclick_spec(spec_path, repository=root)
        validated_spec_path = (root / spec.source_spec_path).resolve()
        store = RunStore(context)
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
                spec_path=validated_spec_path,
                store=store,
                site=profile,
                blade=targets[0],
            )
            _emit({"status": "ok", **result.to_dict()})
            return
        result = run_shiftclick(
            spec,
            store=store,
            output_dir=output_dir,
        )
        _emit({"status": "ok", **result.to_dict()})
    except Exception as exc:  # noqa: BLE001
        _fail(exc)

# ---------------------------------------------------------------------------
# Plotting commands
# ---------------------------------------------------------------------------


@main.command("view")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--metrics", default=None, help="Comma-separated metrics to tile (default: all table metrics).")
@click.option("--ncol", type=int, default=3, show_default=True, help="Tiles per row.")
@click.option("--log-axis", "log_axis_values", multiple=True, help="Log-scale an axis; repeat or comma-separate.")
@click.option("--max-pricethr", type=float, default=100.0, show_default=True, help="Masked max price difference threshold in bps.")
@click.option("--skewthr", type=float, default=None, help="Masked max skew threshold in percent.")
@click.option("--slipthr", type=float, default=20.0, show_default=True, help="Masked APY slippage cap in bps.")
@click.option("--slipthr-max", type=float, default=100.0, show_default=True, help="Upper range of the slippage control in bps (not a second mask bound).")
@click.option("--last-pdifthr", "last_pdiffthr", type=float, default=None, help="Masked final price difference threshold in bps.")
@click.option("--site", default="local", show_default=True, help="Local or configured SSH site for replay.")
@click.option("--blade", default=None, help="Remote blade for replay (defaults to the first configured blade).")
@click.option("--out", "output_file", type=click.Path(path_type=Path, dir_okay=False), default=None, help="PNG path; writes a state sidecar beside it.")
def view_run(
    run_dir: Path,
    metrics: str | None,
    ncol: int,
    log_axis_values: tuple[str, ...],
    max_pricethr: float,
    skewthr: float | None,
    slipthr: float,
    slipthr_max: float,
    last_pdiffthr: float | None,
    site: str,
    blade: str | None,
    output_file: Path | None,
) -> None:
    """Open the maintained three-window N-D explorer for one run directory."""
    try:
        from .artifacts.store import RunStore
        from .plotting.explorer import open_explorer
        from .plotting.heatmap import interactive_backend_active

        if slipthr > slipthr_max:
            raise click.ClickException("--slipthr cannot exceed --slipthr-max")
        selected_metrics = _metric_list(metrics) if metrics is not None else None
        log_axes = [name.strip() for raw in log_axis_values for name in raw.split(",") if name.strip()]
        explorer = open_explorer(
            run_dir,
            metrics=selected_metrics,
            ncol=ncol,
            log_axes=log_axes,
            max_pricethr=max_pricethr,
            skewthr=skewthr,
            slipthr=slipthr,
            slipthr_max=slipthr_max,
            final_pdiffthr=last_pdiffthr,
            site=site,
            blade=blade,
            store=RunStore(_project_context()),
        )
        image = sidecar = None
        try:
            if output_file is not None:
                image, sidecar = explorer.save(output_file)
            if interactive_backend_active():
                explorer.show()
        finally:
            explorer.close()
        _emit({
            "status": "ok",
            "run_dir": run_dir.as_posix(),
            "metrics": list(explorer.metrics),
            "output": image.as_posix() if image is not None else None,
            "state": sidecar.as_posix() if sidecar is not None else None,
        })
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


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
    help="Slippage cap in bps for masked APY metrics. Requires --metrics.",
)
@click.option(
    "--slipthr-max",
    type=float,
    default=None,
    help="Upper range of the slippage control in bps (not a second mask bound). Requires --metrics.",
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

    Uses the same maintained explorer as ``fxsim view`` and always writes the
    PNG slice plus its fxsim_heatmap_state_v1 sidecar.  With --metrics it
    renders one tile per metric in an --ncol grid.
    """
    try:
        from .artifacts.store import RunStore
        from .plotting.explorer import open_explorer
        from .plotting.heatmap import interactive_backend_active

        destination = output_file or run_dir / "heatmap.png"
        tile_metrics = _metric_list(metrics) if metrics is not None else ([metric] if metric else None)
        if slipthr is not None and slipthr_max is not None and slipthr > slipthr_max:
            raise click.ClickException("--slipthr cannot exceed --slipthr-max")
        log_axes = [
            name.strip()
            for raw in log_axis_values
            for name in raw.split(",")
            if name.strip()
        ]
        explorer = open_explorer(
            run_dir,
            metrics=tile_metrics,
            ncol=ncol if metrics is not None else 1,
            log_axes=log_axes,
            x_axis=x_axis,
            y_axis=y_axis,
            max_pricethr=max_pricethr,
            slipthr=slipthr,
            slipthr_max=slipthr_max,
            store=RunStore(_project_context()),
        )
        try:
            image, sidecar = explorer.save(destination)
            if interactive_backend_active():
                explorer.show()
        finally:
            explorer.close()
        _emit({
            "status": "ok",
            "run_dir": run_dir.as_posix(),
            "results_source": (run_dir / str(explorer.state.source)).as_posix(),
            "metric": explorer.state.tiles[0],
            "metrics": list(explorer.state.tiles),
            "ncol": explorer.state.ncol,
            "log_axes": list(explorer.state.log_axes),
            "output": image.as_posix(),
            "state": sidecar.as_posix(),
        })
    except Exception as exc:  # noqa: BLE001
        _fail(exc)


@plot_group.command("trajectory")
@click.argument("diagnostic_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out", "output_file", type=click.Path(path_type=Path, dir_okay=False), default=None)
def plot_trajectory(diagnostic_dir: Path, output_file: Path | None) -> None:
    """Render one scenario from a manifest-attested replay archive."""
    try:
        from .artifacts.attestation import find_attested_artifact
        from .artifacts.manifest import load_manifest
        from .plotting.shiftclick_view import render_shiftclick_figure

        manifest = load_manifest(
            diagnostic_dir / "manifest.json",
            expected_kind="shiftclick",
        )
        trace_path = find_attested_artifact(
            manifest,
            run_dir=diagnostic_dir,
            kind="replay_trace_npz",
            verify_digest=True,
        )
        companion_path = find_attested_artifact(
            manifest,
            run_dir=diagnostic_dir,
            kind="replay_trace_companion",
            verify_digest=True,
        )
        destination = output_file or diagnostic_dir / "trajectory.png"
        figure = render_shiftclick_figure(
            trace_path,
            companion_path=companion_path,
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
def repo_audit() -> None:
    """Verify that no forbidden historical binaries or run outputs are tracked."""
    try:
        import subprocess

        repo = _project_context().project_root
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
