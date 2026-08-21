from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import socket
import time
from typing import Any, Callable, Mapping

from curve_fx_harness_client.models import CandidateResult

from ..artifacts.io import atomic_write_bytes
from ..evaluation.grouping import (
    CompiledEvaluation,
    bind_local_session_group,
    decode_compiled_evaluation,
    encode_compiled_evaluation,
    group_evaluations,
)
from ..evaluation.selected import SelectedEvaluator
from ..evaluation.session import LocalSessionMaterialization
from ..specs.common import canonical_json_bytes
from ..specs.scenario import ScenarioSpec
from .collection import normalize_session_attestation
from .grouped import execute_local_groups
from .shared_nfs import package_identity_sha256
from .site import validate_remote_host
from .staging import validate_run_id

VERSION = "curve_fx_grouped_work_v1"
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class GroupedRemoteError(RuntimeError):
    """A grouped remote request or receipt is invalid."""
    pass


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise GroupedRemoteError(f"{label} must be lowercase SHA-256")
    return value


def _read_canonical(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise GroupedRemoteError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise GroupedRemoteError(f"{label} is not a canonical JSON object")
    return raw, value


def _require_fields(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    if set(value) != fields or value.get("version") != VERSION:
        raise GroupedRemoteError(f"{label} has invalid fields or version")


def _relative_scenario(spec: ScenarioSpec) -> None:
    paths = [
        spec.template_path,
        spec.source_path,
        *(item.path for item in spec.market_files),
    ]
    if any(path is not None and (path.is_absolute() or ".." in path.parts)
           for path in paths):
        raise GroupedRemoteError("grouped scenario paths must be repository-relative")


@dataclass(frozen=True, slots=True)
class GroupedWorkRequest:
    """Canonical work request for one remote grouped-evaluation job."""
    run_id: str
    request_id: str
    evaluations: tuple[CompiledEvaluation, ...]
    scenarios: tuple[ScenarioSpec, ...]
    selected_provenance_json: bytes
    chunk_size: int
    lane_count: int
    package_sha256: str

    @property
    def canonical_json(self) -> bytes:
        return canonical_json_bytes({
            "version": VERSION, "run_id": self.run_id, "request_id": self.request_id,
            "evaluations": [encode_compiled_evaluation(item) for item in self.evaluations],
            "scenarios": [item.to_dict() for item in self.scenarios],
            "selected_artifact_provenance": json.loads(self.selected_provenance_json),
            "chunk_size": self.chunk_size, "lane_count": self.lane_count,
            "package_sha256": self.package_sha256,
        })

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json).hexdigest()

    def validated(self) -> GroupedWorkRequest:
        validate_run_id(self.run_id)
        if not isinstance(self.request_id, str) or not _TOKEN.fullmatch(self.request_id):
            raise GroupedRemoteError("request_id is invalid")
        identifiers = tuple(item.evaluation_id for item in self.evaluations)
        if (not identifiers or any(not item for item in identifiers)
                or len(set(identifiers)) != len(identifiers)):
            raise GroupedRemoteError("request evaluation IDs are empty or duplicated")
        for scenario in self.scenarios:
            _relative_scenario(scenario)
        provenance = json.loads(self.selected_provenance_json)
        if (not isinstance(provenance, dict)
                or canonical_json_bytes(provenance) != self.selected_provenance_json):
            raise GroupedRemoteError("selected artifact provenance is not canonical")
        for field in ("artifact_sha256", "binary_sha256", "parameter_schema_sha256"):
            _sha256(provenance.get(field), f"selected {field}")
        if (isinstance(self.chunk_size, bool) or not isinstance(self.chunk_size, int)
                or isinstance(self.lane_count, bool)
                or not isinstance(self.lane_count, int)):
            raise GroupedRemoteError("chunk_size and lane_count must be integers")
        if self.chunk_size < 1 or self.lane_count < 1:
            raise GroupedRemoteError("chunk_size and lane_count must be positive")
        _sha256(self.package_sha256, "package_sha256")
        return self
    @classmethod
    def from_json(cls, path: Path | str) -> GroupedWorkRequest:
        """Load and validate a canonical grouped work request."""
        raw, value = _read_canonical(Path(path), "grouped request")
        _require_fields(value, {
            "version", "run_id", "request_id", "evaluations", "scenarios",
            "selected_artifact_provenance", "chunk_size", "lane_count", "package_sha256",
        }, "grouped request")
        try:
            request = cls(
                value["run_id"], value["request_id"],
                tuple(decode_compiled_evaluation(item) for item in value["evaluations"]),
                tuple(ScenarioSpec.from_dict(item) for item in value["scenarios"]),
                canonical_json_bytes(value["selected_artifact_provenance"]),
                value["chunk_size"], value["lane_count"], value["package_sha256"],
            ).validated()
        except (KeyError, TypeError, ValueError) as exc:
            raise GroupedRemoteError("grouped request contents are invalid") from exc
        if request.canonical_json != raw:
            raise GroupedRemoteError("grouped request is noncanonical")
        return request


@dataclass(frozen=True, slots=True)
class GroupedWorkReceipt:
    """Canonical result receipt for one remote grouped-evaluation job."""
    request_sha256: str
    blade: str
    artifact_sha256: str
    group_session_attestations: Mapping[str, Mapping[str, str]]
    results: tuple[CandidateResult, ...]
    elapsed_seconds: float

    @property
    def canonical_json(self) -> bytes:
        return canonical_json_bytes({
            "version": VERSION, "request_sha256": self.request_sha256,
            "blade": self.blade, "artifact_sha256": self.artifact_sha256,
            "group_session_attestations": self.group_session_attestations,
            "results": [item.model_dump() for item in self.results],
            "elapsed_seconds": self.elapsed_seconds,
        })

    @classmethod
    def from_json(cls, path: Path | str) -> GroupedWorkReceipt:
        """Load and validate a canonical grouped work receipt."""
        raw, value = _read_canonical(Path(path), "grouped receipt")
        _require_fields(value, {
            "version", "request_sha256", "blade", "artifact_sha256",
            "group_session_attestations", "results", "elapsed_seconds",
        }, "grouped receipt")
        if isinstance(value["elapsed_seconds"], bool):
            raise GroupedRemoteError("grouped receipt elapsed_seconds is invalid")
        try:
            receipt = cls(
                _sha256(value["request_sha256"], "request_sha256"),
                validate_remote_host(value["blade"], "receipt blade"),
                _sha256(value["artifact_sha256"], "artifact_sha256"),
                {_sha256(key, "session group id"): normalize_session_attestation(item)
                 for key, item in value["group_session_attestations"].items()},
                tuple(CandidateResult.model_validate(item) for item in value["results"]),
                float(value["elapsed_seconds"]),
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise GroupedRemoteError("grouped receipt contents are invalid") from exc
        if not math.isfinite(receipt.elapsed_seconds) or receipt.elapsed_seconds < 0:
            raise GroupedRemoteError("grouped receipt elapsed_seconds is invalid")
        if receipt.canonical_json != raw:
            raise GroupedRemoteError("grouped receipt is noncanonical")
        return receipt


def _exact_worker_paths(
    remote_run_root: Path | str, request_path: Path | str,
    output_path: Path | str,
) -> tuple[Path, Path, str, str]:
    root = Path(remote_run_root).resolve()
    request = Path(request_path).resolve()
    output = Path(output_path).resolve()
    try:
        parts = request.relative_to(root).parts
    except ValueError as exc:
        raise GroupedRemoteError("grouped request escapes remote_run_root") from exc
    if (
        len(parts) != 3
        or parts[1] != "grouped_requests"
        or not parts[2].endswith(".json")
    ):
        raise GroupedRemoteError("grouped request path is not canonical")
    run_id, request_id = parts[0], parts[2][:-5]
    validate_run_id(run_id)
    if not _TOKEN.fullmatch(request_id):
        raise GroupedRemoteError("request path has an invalid request_id")
    expected_output = root / run_id / "grouped_receipts" / f"{request_id}.json"
    if (
        request != root / run_id / "grouped_requests" / f"{request_id}.json"
        or output != expected_output
    ):
        raise GroupedRemoteError("grouped request or receipt path is not exact")
    return request, output, run_id, request_id


def execute_grouped_work(
    request_path: Path | str, output_path: Path | str, *,
    remote_run_root: Path | str, repository: Path,
    client_factory: Callable[[SelectedEvaluator, Path], Any] | None = None,
    blade: str | None = None,
) -> GroupedWorkReceipt:
    """Execute one exact-path grouped request and persist its receipt."""
    request_file, output, run_id, request_id = _exact_worker_paths(
        remote_run_root, request_path, output_path)
    request = GroupedWorkRequest.from_json(request_file)
    if request.run_id != run_id or request.request_id != request_id:
        raise GroupedRemoteError("request identity differs from its exact path")
    run_root = Path(remote_run_root).resolve() / run_id
    repository = Path(repository).resolve()
    selected = SelectedEvaluator.load(run_root / "evaluator_artifact")
    if selected.provenance != json.loads(request.selected_provenance_json):
        raise GroupedRemoteError("selected evaluator differs from request provenance")
    if package_identity_sha256(repository) != request.package_sha256:
        raise GroupedRemoteError("worker package closure differs from request")
    groups = group_evaluations(
        request.evaluations,
        artifact_sha256=selected.artifact_sha256,
        parameter_schema=selected.compiler.schema,
    )
    blade_name = validate_remote_host(
        blade or socket.gethostname().split(".", 1)[0],
        "worker blade",
    )
    work_root = run_root / "remote_groups" / request_id / blade_name
    scenario_by_key = {}
    for index, scenario in enumerate(request.scenarios):
        materialized = LocalSessionMaterialization.from_scenario(
            scenario, repository=repository, manifest_root=work_root,
            session_id=f"scenario_{index}").validated()
        if materialized.scenario_key in scenario_by_key:
            raise GroupedRemoteError("request scenarios derive duplicate ScenarioKeys")
        scenario_by_key[materialized.scenario_key] = scenario
    if {group.scenario_key for group in groups} != set(scenario_by_key):
        raise GroupedRemoteError("scenario records do not exactly cover evaluations")
    bindings = {}
    for group in groups:
        materialized = LocalSessionMaterialization.from_scenario(
            scenario_by_key[group.scenario_key], repository=repository,
            manifest_root=work_root,
            session_id=f"sess_{run_id}_{group.key.sha256[:12]}").validated()
        bindings[group.key.sha256] = bind_local_session_group(group, materialized)
    started = time.monotonic()
    optional = {} if client_factory is None else {"client_factory": client_factory}
    execution = execute_local_groups(
        selected, groups,
        lambda group: bindings[group.key.sha256],
        tuple(str(item.evaluation_id) for item in request.evaluations),
        work_dir=work_root,
        chunk_size=request.chunk_size, max_workers=request.lane_count, **optional)
    receipt = GroupedWorkReceipt(
        request.sha256,
        blade_name,
        selected.artifact_sha256,
        {
            key: value.session_attestation
            for key, value in execution.receipts_by_session_group_id.items()
        },
        tuple(
            execution.results_by_evaluation_id[str(item.evaluation_id)]
            for item in request.evaluations
        ),
        time.monotonic() - started,
    )
    atomic_write_bytes(output, receipt.canonical_json)
    return receipt
