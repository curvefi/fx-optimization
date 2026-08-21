"""Build and verify one evaluator from explicit pool, harness, and policy sources."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping

from ..artifacts.io import atomic_write_json, sha256_path
from ..specs.common import canonical_json_bytes

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.+-]+$")
_DESCRIPTION_SCHEMA = "curve_fx_evaluator_description_v1"
_ARTIFACT_SCHEMA = "curve_fx_evaluator_artifact_v1"
_PASSTHROUGH = Path("include/pools/twocrypto_fx/policies/compiled_passthrough.hpp")

class EvaluatorBuildError(RuntimeError):
    """Raised when sources, compilation, or evaluator attestation disagree."""


@dataclass(frozen=True)
class BuildSpec:
    """All source and CMake inputs that select one evaluator binary."""

    pool_root: Path
    harness_root: Path
    policy_header: Path | None = None
    policy_id: str | None = None
    policy_abi: str = "twocrypto_policy_v1"
    policy_expected_sha256: str | None = None
    numeric_mode: Literal["f64", "longdouble"] = "longdouble"
    build_type: str = "Release"
    enable_ipo: bool = False
    native_tuning: bool = False

    def __post_init__(self) -> None:
        pool = _source_root(self.pool_root, "pool_root")
        harness = _source_root(self.harness_root, "harness_root")
        object.__setattr__(self, "pool_root", pool)
        object.__setattr__(self, "harness_root", harness)
        if self.numeric_mode not in ("f64", "longdouble"):
            raise ValueError("numeric_mode must be 'f64' or 'longdouble'")
        if not _TOKEN_RE.fullmatch(self.build_type):
            raise ValueError("build_type contains unsupported characters")
        if self.policy_abi != "twocrypto_policy_v1":
            raise ValueError("policy_abi must be exactly 'twocrypto_policy_v1'")

        if self.policy_header is None:
            if self.policy_id is not None or self.policy_expected_sha256 is not None:
                raise ValueError("policy_id/digest require policy_header")
            return
        raw_header = Path(self.policy_header)
        if raw_header.is_symlink():
            raise ValueError(f"policy_header must not be a symlink: {raw_header}")
        header = raw_header.resolve(strict=True)
        if not header.is_file():
            raise ValueError(f"policy_header is not a regular file: {header}")
        if not self.policy_id or not _TOKEN_RE.fullmatch(self.policy_id):
            raise ValueError("policy_id is required and must be a safe token")
        digest = self.policy_expected_sha256
        if digest is None or not _SHA256_RE.fullmatch(digest):
            raise ValueError("policy_expected_sha256 must be a lowercase SHA-256 digest")
        actual = sha256_path(header)
        if actual != digest:
            raise ValueError(f"policy digest mismatch: expected {digest}, calculated {actual}")
        object.__setattr__(self, "policy_header", header)


@dataclass(frozen=True)
class FileReceipt:
    path: str
    sha256: str
    size: int

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}


@dataclass(frozen=True)
class SourceClosureReceipt:
    sha256: str
    files: tuple[FileReceipt, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "files": [item.to_dict() for item in self.files]}


@dataclass(frozen=True)
class EvaluatorArtifact:
    binary_path: Path
    binary_sha256: str
    description: Mapping[str, Any]
    pool_closure: SourceClosureReceipt
    harness_closure: SourceClosureReceipt
    policy_closure: SourceClosureReceipt
    build_spec_sha256: str
    artifact_sha256: str
    receipt_path: Path

def _source_root(value: Path, label: str) -> Path:
    raw = Path(value)
    if raw.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {raw}")
    root = raw.resolve(strict=True)
    if not root.is_dir() or not (root / "CMakeLists.txt").is_file():
        raise ValueError(f"{label} must contain CMakeLists.txt: {root}")
    return root


def _closure(root: Path, subtree: str) -> SourceClosureReceipt:
    selected = [root / "CMakeLists.txt"]
    tree = root / subtree
    if not tree.is_dir() or tree.is_symlink():
        raise EvaluatorBuildError(f"required source directory is missing or unsafe: {tree}")
    for directory, dirnames, filenames in os.walk(tree, followlinks=False):
        directory_path = Path(directory)
        for name in tuple(dirnames):
            candidate = directory_path / name
            if candidate.is_symlink():
                raise EvaluatorBuildError(f"source closure contains symlink: {candidate}")
        for name in filenames:
            candidate = directory_path / name
            if candidate.is_symlink() or not candidate.is_file():
                raise EvaluatorBuildError(f"source closure contains unsafe file: {candidate}")
            selected.append(candidate)
    files = tuple(
        FileReceipt(
            path=path.relative_to(root).as_posix(),
            sha256=sha256_path(path),
            size=path.stat().st_size,
        )
        for path in sorted(selected, key=lambda item: item.relative_to(root).as_posix())
    )
    return SourceClosureReceipt(_receipt_digest(files), files)

def _policy_closure(spec: BuildSpec) -> SourceClosureReceipt:
    header = spec.policy_header or (spec.pool_root / _PASSTHROUGH)
    if header.is_symlink() or not header.is_file():
        raise EvaluatorBuildError(f"compiled policy header is missing or unsafe: {header}")
    item = FileReceipt("policy_header", sha256_path(header), header.stat().st_size)
    if spec.policy_expected_sha256 and item.sha256 != spec.policy_expected_sha256:
        raise EvaluatorBuildError("compiled policy bytes no longer match BuildSpec")
    return SourceClosureReceipt(_receipt_digest((item,)), (item,))

def _receipt_digest(files: tuple[FileReceipt, ...]) -> str:
    return hashlib.sha256(
        canonical_json_bytes([item.to_dict() for item in files])
    ).hexdigest()

def build_receipt(spec: BuildSpec) -> dict[str, Any]:
    """Return the path-independent receipt for the exact current source closure."""
    policy = _policy_closure(spec)
    expected_id = spec.policy_id or "native_passthrough"
    receipt: dict[str, Any] = {
        "numeric_mode": spec.numeric_mode,
        "build_type": spec.build_type,
        "enable_ipo": spec.enable_ipo,
        "native_tuning": spec.native_tuning,
        "policy": {
            "id": expected_id,
            "abi": spec.policy_abi,
            "source_sha256": policy.files[0].sha256,
        },
        "source_closures": {
            "pool": _closure(spec.pool_root, "include").to_dict(),
            "harness": _closure(spec.harness_root, "cpp").to_dict(),
            "policy": policy.to_dict(),
        },
    }
    receipt["build_spec_sha256"] = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    return receipt

def _run(argv: list[str], *, timeout: int = 1800) -> str:
    try:
        completed = subprocess.run(
            argv,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        raise EvaluatorBuildError(
            f"command failed: {argv!r}" + (f"\n{stderr.strip()}" if stderr else "")
        ) from exc
    return completed.stdout

def _binary_in(build_dir: Path, target: str, build_type: str) -> Path:
    candidates = (build_dir / target, build_dir / build_type / target)
    found = [path.resolve() for path in candidates if path.is_file() and not path.is_symlink()]
    if len(found) != 1:
        raise EvaluatorBuildError(f"expected exactly one built {target!r}, found {found}")
    return found[0]

def _describe(binary: Path) -> dict[str, Any]:
    output = _run([str(binary), "--describe-json"], timeout=30)
    try:
        value = json.loads(output)
    except json.JSONDecodeError as exc:
        raise EvaluatorBuildError("evaluator emitted malformed --describe-json output") from exc
    if not isinstance(value, dict):
        raise EvaluatorBuildError("evaluator description must be a JSON object")
    return value


def _validate_description(
    description: Mapping[str, Any], binary: Path, receipt: Mapping[str, Any]
) -> None:
    if description.get("schema_version") != _DESCRIPTION_SCHEMA:
        raise EvaluatorBuildError("unsupported evaluator description schema")
    actual_binary_sha = sha256_path(binary)
    if description.get("binary_sha256") != actual_binary_sha:
        raise EvaluatorBuildError("description binary_sha256 does not match evaluator bytes")
    expected_policy = receipt["policy"]
    policy = description.get("policy")
    if not isinstance(policy, Mapping):
        raise EvaluatorBuildError("description omitted policy identity")
    for key in ("id", "abi", "source_sha256"):
        if policy.get(key) != expected_policy[key]:
            raise EvaluatorBuildError(f"description policy {key} does not match build receipt")
    count = policy.get("parameter_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise EvaluatorBuildError("description policy parameter_count is invalid")
    if policy.get("descriptor_abi_version") != 1:
        raise EvaluatorBuildError("unsupported policy descriptor ABI")

    build = description.get("build")
    if not isinstance(build, Mapping):
        raise EvaluatorBuildError("description omitted build identity")
    target = "arb_evaluator_f64" if receipt["numeric_mode"] == "f64" else "arb_evaluator_ld"
    expected_build = {
        "type": receipt["build_type"],
        "target": target,
        "numeric_mode": "double" if receipt["numeric_mode"] == "f64" else "longdouble",
        "ipo_enabled": receipt["enable_ipo"],
        "native_tuning": receipt["native_tuning"],
        "wire_real_type": "IEEE-754 binary64",
        "wire_real_digits": 53,
    }
    for key, expected in expected_build.items():
        if build.get(key) != expected:
            raise EvaluatorBuildError(f"description build {key} does not match build receipt")

    schema_digest = description.get("parameter_schema_sha256")
    if not isinstance(schema_digest, str) or not _SHA256_RE.fullmatch(schema_digest):
        raise EvaluatorBuildError("description parameter_schema_sha256 is invalid")
    schema = description.get("parameter_schema")
    if not isinstance(schema, Mapping) or schema.get("schema_version") != "curve_fx_parameter_schema_v1":
        raise EvaluatorBuildError("description parameter schema is invalid")
    canonical_schema = description.get("parameter_schema_canonical_json")
    if not isinstance(canonical_schema, str):
        raise EvaluatorBuildError("description omitted canonical parameter schema")
    if hashlib.sha256(canonical_schema.encode()).hexdigest() != schema_digest:
        raise EvaluatorBuildError("description parameter_schema_sha256 does not match canonical schema")
    try:
        parsed_canonical_schema = json.loads(canonical_schema)
    except json.JSONDecodeError as exc:
        raise EvaluatorBuildError("description canonical parameter schema is malformed") from exc
    if parsed_canonical_schema != schema:
        raise EvaluatorBuildError("description canonical parameter schema does not match schema")
    parameters = schema.get("parameters")
    if not isinstance(parameters, list):
        raise EvaluatorBuildError("description parameter schema omitted parameters")
    policy_parameters = [item for item in parameters if isinstance(item, Mapping) and str(item.get("name", "")).startswith("policy.")]
    if len(policy_parameters) != count:
        raise EvaluatorBuildError("policy descriptor count does not match policy identity")
    for order, parameter in enumerate(policy_parameters):
        required = ("name", "type", "unit", "default", "minimum", "maximum", "quantum")
        if parameter.get("order") != order or any(key not in parameter for key in required):
            raise EvaluatorBuildError("policy parameter descriptor is incomplete or out of order")
        if parameter.get("lowering_path") != f"evaluate_batch.candidates[].policy_params[{order}]":
            raise EvaluatorBuildError("policy parameter descriptor has the wrong lowering path")
        if parameter.get("classification") != "candidate" or parameter.get("wire_representation") != "finite_binary64":
            raise EvaluatorBuildError("policy parameter descriptor has the wrong wire contract")
        bounds = (parameter["default"], parameter["minimum"], parameter["maximum"], parameter["quantum"])
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in bounds):
            raise EvaluatorBuildError("policy parameter descriptor has invalid numeric bounds")
        if parameter["minimum"] > parameter["default"] or parameter["default"] > parameter["maximum"] or parameter["quantum"] < 0:
            raise EvaluatorBuildError("policy parameter descriptor bounds are inconsistent")


def build_evaluator(spec: BuildSpec, artifact_dir: Path) -> EvaluatorArtifact:
    """Compile and atomically publish a new artifact; the destination must be fresh."""
    destination = Path(artifact_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"artifact directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = build_receipt(spec)
    target = "arb_evaluator_f64" if spec.numeric_mode == "f64" else "arb_evaluator_ld"

    with tempfile.TemporaryDirectory(prefix="curve-fx-build-") as temporary:
        temp = Path(temporary)
        pool_build, pool_install, harness_build = temp / "pool-build", temp / "pool-install", temp / "harness-build"
        _run([
            "cmake", "-S", str(spec.pool_root), "-B", str(pool_build),
            f"-DCMAKE_BUILD_TYPE={spec.build_type}",
            f"-DCMAKE_INSTALL_PREFIX={pool_install}",
            "-DTWOCRYPTO_POOL_BUILD_TESTS=OFF",
            "-DTWOCRYPTO_POOL_BUILD_BENCHMARKS=OFF",
        ])
        _run(["cmake", "--build", str(pool_build), "--target", "install", "--config", spec.build_type, "--parallel"])
        configure = [
            "cmake", "-S", str(spec.harness_root), "-B", str(harness_build),
            f"-DCMAKE_BUILD_TYPE={spec.build_type}",
            f"-DCMAKE_PREFIX_PATH={pool_install}",
            "-DBUILD_TESTING=OFF",
            f"-DCURVE_FX_ENABLE_IPO={'ON' if spec.enable_ipo else 'OFF'}",
            f"-DCURVE_FX_NATIVE_TUNING={'ON' if spec.native_tuning else 'OFF'}",
        ]
        if spec.policy_header is not None:
            configure.extend([
                f"-DPOLICY_HEADER_PATH={spec.policy_header}",
                f"-DPOLICY_ID={spec.policy_id}",
                f"-DPOLICY_ABI={spec.policy_abi}",
                f"-DPOLICY_EXPECTED_SHA256={spec.policy_expected_sha256}",
            ])
        _run(configure)
        _run(["cmake", "--build", str(harness_build), "--target", target, "--config", spec.build_type, "--parallel"])
        binary = _binary_in(harness_build, target, spec.build_type)
        description = _describe(binary)
        after = build_receipt(spec)
        if after != before:
            raise EvaluatorBuildError("source closure changed while evaluator was being built")
        _validate_description(description, binary, before)
        _publish(destination, binary, description, before)
    return load_evaluator_artifact(destination)


def _publish(destination: Path, binary: Path, description: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        published_binary = staging / "evaluator"
        shutil.copy2(binary, published_binary)
        published_binary.chmod(published_binary.stat().st_mode | stat.S_IXUSR)
        payload = {
            "schema_version": _ARTIFACT_SCHEMA,
            "binary": {"path": "evaluator", "sha256": sha256_path(published_binary)},
            "description": dict(description),
            "build_receipt": dict(receipt),
        }
        payload["artifact_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        atomic_write_json(staging / "artifact.json", payload)
        if destination.exists():
            raise FileExistsError(f"artifact directory appeared during build: {destination}")
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _closure_from_dict(value: Any, label: str) -> SourceClosureReceipt:
    if not isinstance(value, Mapping) or not isinstance(value.get("files"), list):
        raise EvaluatorBuildError(f"artifact {label} closure is malformed")
    files: list[FileReceipt] = []
    for raw in value["files"]:
        if not isinstance(raw, Mapping):
            raise EvaluatorBuildError(f"artifact {label} closure entry is malformed")
        path, digest, size = raw.get("path"), raw.get("sha256"), raw.get("size")
        if not isinstance(path, str) or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
            raise EvaluatorBuildError(f"artifact {label} closure path is unsafe")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest) or not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise EvaluatorBuildError(f"artifact {label} closure entry is invalid")
        files.append(FileReceipt(path, digest, size))
    receipt = SourceClosureReceipt(_receipt_digest(tuple(files)), tuple(files))
    if value.get("sha256") != receipt.sha256:
        raise EvaluatorBuildError(f"artifact {label} closure digest is invalid")
    return receipt


def load_evaluator_artifact(artifact_dir: Path) -> EvaluatorArtifact:
    """Load an explicitly selected artifact and re-verify its receipt and executable."""
    requested_root = Path(artifact_dir)
    if requested_root.is_symlink():
        raise EvaluatorBuildError("artifact directory must not be a symlink")
    root = requested_root.resolve(strict=True)
    receipt_path = root / "artifact.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise EvaluatorBuildError(f"artifact receipt is missing or unsafe: {receipt_path}")
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluatorBuildError("artifact receipt is unreadable") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != _ARTIFACT_SCHEMA:
        raise EvaluatorBuildError("unsupported evaluator artifact schema")
    declared_artifact_sha = payload.pop("artifact_sha256", None)
    calculated_artifact_sha = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if declared_artifact_sha != calculated_artifact_sha:
        raise EvaluatorBuildError("artifact receipt SHA-256 is invalid")
    binary_record = payload.get("binary")
    if not isinstance(binary_record, Mapping) or binary_record.get("path") != "evaluator":
        raise EvaluatorBuildError("artifact binary path is invalid")
    binary_candidate = root / "evaluator"
    if binary_candidate.is_symlink():
        raise EvaluatorBuildError("artifact binary must not be a symlink")
    binary = binary_candidate.resolve(strict=True)
    if not binary.is_relative_to(root) or not binary.is_file():
        raise EvaluatorBuildError("artifact binary escapes its directory or is unsafe")
    binary_sha = sha256_path(binary)
    if binary_record.get("sha256") != binary_sha:
        raise EvaluatorBuildError("artifact binary SHA-256 is invalid")

    receipt = payload.get("build_receipt")
    if not isinstance(receipt, dict):
        raise EvaluatorBuildError("artifact build receipt is missing")
    closures = receipt.get("source_closures")
    if not isinstance(closures, Mapping):
        raise EvaluatorBuildError("artifact source closures are missing")
    pool = _closure_from_dict(closures.get("pool"), "pool")
    harness = _closure_from_dict(closures.get("harness"), "harness")
    policy = _closure_from_dict(closures.get("policy"), "policy")
    declared_build_sha = receipt.pop("build_spec_sha256", None)
    calculated_build_sha = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    receipt["build_spec_sha256"] = declared_build_sha
    if declared_build_sha != calculated_build_sha:
        raise EvaluatorBuildError("artifact build receipt SHA-256 is invalid")
    description = payload.get("description")
    if not isinstance(description, dict):
        raise EvaluatorBuildError("artifact evaluator description is missing")
    _validate_description(description, binary, receipt)
    if _describe(binary) != description:
        raise EvaluatorBuildError("artifact description differs from executable description")
    return EvaluatorArtifact(
        binary_path=binary,
        binary_sha256=binary_sha,
        description=description,
        pool_closure=pool,
        harness_closure=harness,
        policy_closure=policy,
        build_spec_sha256=declared_build_sha,
        artifact_sha256=declared_artifact_sha,
        receipt_path=receipt_path,
    )

__all__ = [
    "BuildSpec", "EvaluatorArtifact", "EvaluatorBuildError",
    "build_evaluator", "build_receipt", "load_evaluator_artifact",
]
