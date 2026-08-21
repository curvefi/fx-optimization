"""Orchestrator adapter for the canonical curve_fx_harness_client transport."""

from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Sequence

from curve_fx_harness_client import EvaluatorClient
from curve_fx_harness_client.exceptions import ProtocolViolationError as ProtocolError
from curve_fx_harness_client.models import (
    BatchResultFrame,
    CandidateSpec,
    ObservationSpec,
    SessionReadyFrame,
)

from ..specs.common import assert_contained_path
from ..specs.scenario import ScenarioSpec
from .identity import VerifiedEvaluator, inspect_binary_identity
from .plans import CandidatePlan
from .session import LocalSessionMaterialization, _verified_plan_request


class HarnessClient(ABC):
    """Domain adapter around the canonical curve_fx_eval_v1 client."""

    @abstractmethod
    def prepare(self) -> VerifiedEvaluator:
        """Start the evaluator and attest its immutable binary identity."""

    @abstractmethod
    def open_session(
        self,
        scenario_spec: ScenarioSpec,
        session_id: str | None = None,
    ) -> SessionReadyFrame:
        """Open an immutable, hash-attested simulation session."""

    def open_compiled_session(
        self,
        plan: CandidatePlan,
        materialization: LocalSessionMaterialization,
    ) -> SessionReadyFrame:
        """Open a typed compiled session when the concrete client supports it."""
        raise NotImplementedError("this harness client does not support compiled sessions")

    @abstractmethod
    def evaluate_batch(
        self,
        candidates: Sequence[CandidateSpec],
        observation: ObservationSpec | None = None,
    ) -> BatchResultFrame:
        """Evaluate canonical candidate requests over the active session."""

    @abstractmethod
    def close(self) -> None:
        """Close the session and evaluator process."""

    @property
    def artifact_root(self) -> Path:
        """Directory against which evaluator artifact paths are resolved."""
        raise NotImplementedError("this harness client does not publish artifacts")

    def artifact_directory(self, path: Path) -> str:
        """Convert a confined artifact directory to the evaluator's relative form."""
        resolved = assert_contained_path(path, self.artifact_root, allow_symlinks=False)
        relative = resolved.relative_to(self.artifact_root)
        if relative == Path("."):
            raise ValueError("artifact directory must be below the evaluator work directory")
        return relative.as_posix()


class SubprocessHarnessClient(HarnessClient):
    """Resolve inputs from the project while confining evaluator output to work_dir."""

    def __init__(
        self,
        binary_path: str | os.PathLike[str],
        *,
        repository: Path,
        work_dir: Path,
        expected_policy_id: str | None = None,
        expected_policy_source_sha256: str | None = None,
        expected_policy_abi: str | None = None,
        expected_policy_parameter_count: int | None = None,
        expected_numeric_mode: str | None = None,
    ) -> None:
        self.root = repository.resolve()
        self.work_dir = work_dir.resolve()
        self.binary_path = Path(binary_path).resolve()
        if not self.binary_path.is_file():
            raise FileNotFoundError(f"evaluator binary not found: {self.binary_path}")
        self.expected_policy_id = expected_policy_id
        self.expected_policy_source_sha256 = expected_policy_source_sha256
        self.expected_policy_abi = expected_policy_abi
        self.expected_policy_parameter_count = expected_policy_parameter_count
        self.expected_numeric_mode = expected_numeric_mode
        self._identity: VerifiedEvaluator | None = None
        self._hello = None
        self._session_id: str | None = None
        self._session_fingerprint: str | None = None
        self._legacy_scenario_fingerprint: str | None = None
        self._session_ready: SessionReadyFrame | None = None
        self._session_manifest_dir = tempfile.TemporaryDirectory(prefix="fxsim-session-")
        self._client = self._make_protocol_client()

    def _make_protocol_client(self) -> EvaluatorClient:
        return EvaluatorClient(
            executable_path=self.binary_path,
            work_dir=self.work_dir,
            expected_policy_id=self.expected_policy_id,
            expected_policy_source_sha256=self.expected_policy_source_sha256,
            expected_policy_abi=self.expected_policy_abi,
            expected_policy_parameter_count=self.expected_policy_parameter_count,
        )
    def clone(self) -> SubprocessHarnessClient:
        """Create a fresh protocol client with identical immutable expectations."""
        return type(self)(
            self.binary_path,
            repository=self.root,
            work_dir=self.work_dir,
            expected_policy_id=self.expected_policy_id,
            expected_policy_source_sha256=self.expected_policy_source_sha256,
            expected_policy_abi=self.expected_policy_abi,
            expected_policy_parameter_count=self.expected_policy_parameter_count,
            expected_numeric_mode=self.expected_numeric_mode,
        )

    @property
    def artifact_root(self) -> Path:
        return self.work_dir


    def prepare(self) -> VerifiedEvaluator:
        if self._hello is None:
            self._hello = self._client.start()
        if self._identity is None:
            self._identity = inspect_binary_identity(self.binary_path)
        remote = self._hello.evaluator_identity
        if remote.binary_sha256.lower() != self._identity.sha256.lower():
            raise ProtocolError(
                f"evaluator hello reported binary_sha256 {remote.binary_sha256!r} "
                f"!= computed file digest {self._identity.sha256!r}"
            )
        if remote.numeric_mode != self._identity.numeric_mode:
            raise ProtocolError(
                f"evaluator hello numeric_mode {remote.numeric_mode!r} "
                f"!= inspected {self._identity.numeric_mode!r}"
            )
        if self.expected_numeric_mode and remote.numeric_mode != self.expected_numeric_mode:
            raise ProtocolError(
                f"evaluator numeric_mode {remote.numeric_mode!r} != expected {self.expected_numeric_mode!r}"
            )
        if self.expected_policy_id and remote.policy_id != self.expected_policy_id:
            raise ProtocolError(
                f"evaluator policy {remote.policy_id!r} != expected {self.expected_policy_id!r}"
            )
        if self.expected_policy_source_sha256 and (
            not remote.policy_source_sha256
            or remote.policy_source_sha256.lower() != self.expected_policy_source_sha256.lower()
        ):
            raise ProtocolError(
                "evaluator policy source SHA-256 "
                f"{remote.policy_source_sha256!r} != expected {self.expected_policy_source_sha256!r}"
            )
        if self.expected_policy_abi and remote.policy_abi != self.expected_policy_abi:
            raise ProtocolError(
                f"evaluator policy ABI {remote.policy_abi!r} != expected {self.expected_policy_abi!r}"
            )
        if (
            self.expected_policy_parameter_count is not None
            and remote.policy_parameter_count != self.expected_policy_parameter_count
        ):
            raise ProtocolError(
                "evaluator policy parameter count "
                f"{remote.policy_parameter_count!r} != expected {self.expected_policy_parameter_count}"
            )
        return self._identity

    def materialize_session(
        self,
        scenario_spec: ScenarioSpec,
        session_id: str | None = None,
    ) -> LocalSessionMaterialization:
        """Verify local scenario bytes and publish the narrow v1 session manifest."""
        return LocalSessionMaterialization.from_scenario(
            scenario_spec,
            repository=self.root,
            manifest_root=Path(self._session_manifest_dir.name),
            session_id=f"sess_{scenario_spec.id}" if session_id is None else session_id,
        )

    def _open_verified_request(
        self,
        request: dict[str, object],
        request_sha256: str,
        *,
        legacy_scenario_fingerprint: str | None = None,
    ) -> SessionReadyFrame:
        session_id = request["session_id"]
        assert isinstance(session_id, str)
        if self._session_id is not None:
            if request_sha256 == self._session_fingerprint:
                assert self._session_ready is not None
                return self._session_ready
            self._client.shutdown()
            self._client = self._make_protocol_client()
            self._hello = None
            self._session_id = None
            self._session_fingerprint = None
            self._legacy_scenario_fingerprint = None
            self._session_ready = None
        self.prepare()
        ready = self._client.open_session(**request)
        self._session_id = session_id
        self._session_fingerprint = request_sha256
        self._legacy_scenario_fingerprint = legacy_scenario_fingerprint
        self._session_ready = ready
        return ready

    def open_session(
        self,
        scenario_spec: ScenarioSpec,
        session_id: str | None = None,
    ) -> SessionReadyFrame:
        fingerprint = scenario_spec.scenario_fingerprint()
        sess_id = session_id or f"sess_{scenario_spec.id}"
        if self._session_id is not None and fingerprint == self._legacy_scenario_fingerprint:
            if sess_id != self._session_id:
                raise ProtocolError(
                    "an evaluator process admits one immutable session; "
                    f"cannot reopen {self._session_id!r} as {sess_id!r}"
                )
            assert self._session_ready is not None
            return self._session_ready
        materialization = self.materialize_session(scenario_spec, session_id=sess_id)
        return self._open_verified_request(
            materialization.baseline_open_session_fields,
            materialization.baseline_request_sha256,
            legacy_scenario_fingerprint=fingerprint,
        )

    def open_compiled_session(
        self,
        plan: CandidatePlan,
        materialization: LocalSessionMaterialization,
    ) -> SessionReadyFrame:
        """Verify and open the compiler's exact canonical protocol-v1 request."""
        request, request_sha256 = _verified_plan_request(plan, materialization)
        return self._open_verified_request(request, request_sha256)

    def evaluate_batch(
        self,
        candidates: Sequence[CandidateSpec],
        observation: ObservationSpec | None = None,
    ) -> BatchResultFrame:
        if self._session_id is None:
            raise ProtocolError("no active simulation session; call open_session() first")
        return self._client.evaluate_batch(list(candidates), observation=observation)

    def close(self) -> None:
        try:
            if self._session_id is not None:
                self._client.close_session(self._session_id)
        finally:
            self._session_id = None
            self._session_fingerprint = None
            self._legacy_scenario_fingerprint = None
            self._session_ready = None
            self._client.shutdown()
            self._session_manifest_dir.cleanup()

class ScenarioHarnessClient(HarnessClient):
    """Keep one immutable subprocess/session per scenario for a run."""

    def __init__(
        self,
        scenarios: Sequence[ScenarioSpec],
        client_factory: Callable[[ScenarioSpec], HarnessClient],
    ) -> None:
        self._scenarios = tuple(scenarios)
        self._scenario_ids = frozenset(scenario.id for scenario in self._scenarios)
        self._clients: dict[str, HarnessClient] = {}
        self._identities: dict[str, VerifiedEvaluator] = {}
        self._factory = client_factory
        self._active_scenario: str | None = None

    def _client_for(self, scenario: ScenarioSpec) -> HarnessClient:
        existing = self._clients.get(scenario.id)
        if existing is None:
            existing = self._factory(scenario)
            self._clients[scenario.id] = existing
        return existing

    def prepare(self) -> VerifiedEvaluator:
        if not self._scenarios:
            raise ValueError("at least one scenario is required")
        first = self._client_for(self._scenarios[0])
        identity = first.prepare()
        self._identities[self._scenarios[0].id] = identity
        return identity

    def open_session(
        self,
        scenario_spec: ScenarioSpec,
        session_id: str | None = None,
    ) -> SessionReadyFrame:
        if scenario_spec.id not in self._scenario_ids:
            raise ValueError(f"scenario {scenario_spec.id!r} is not part of this client pool")
        client = self._client_for(scenario_spec)
        identity = client.prepare()
        first_identity = next(iter(self._identities.values()), identity)
        if identity.sha256.lower() != first_identity.sha256.lower():
            raise ProtocolError("all scenario evaluator processes must have the same binary SHA-256")
        self._identities[scenario_spec.id] = identity
        self._active_scenario = scenario_spec.id
        return client.open_session(scenario_spec, session_id=session_id)

    def evaluate_batch(
        self,
        candidates: Sequence[CandidateSpec],
        observation: ObservationSpec | None = None,
    ) -> BatchResultFrame:
        if self._active_scenario is None:
            raise ProtocolError("no active simulation scenario; call open_session() first")
        return self._clients[self._active_scenario].evaluate_batch(candidates, observation=observation)

    @property
    def artifact_root(self) -> Path:
        if self._active_scenario is None:
            raise ProtocolError("no active simulation scenario; call open_session() first")
        return self._clients[self._active_scenario].artifact_root

    def close(self) -> None:
        clients = tuple(self._clients.values())
        self._active_scenario = None
        self._clients.clear()
        self._identities.clear()
        for client in clients:
            client.close()




__all__ = ["HarnessClient", "SubprocessHarnessClient", "ScenarioHarnessClient"]
