"""Persistent canonical evaluator clients keyed by blade, policy, session, and worker slot."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from curve_fx_harness_client import EvaluatorClient

from .site import SSHConfig

logger = logging.getLogger("curve_fx_sim.execution.evaluator_pool")

ClientFactory = Callable[[str, str], EvaluatorClient]


def _ssh_launch_argv(
    blade: str,
    executable_path: str,
    ssh_config: SSHConfig,
) -> list[str]:
    argv = ["ssh", *ssh_config.options]
    if ssh_config.key:
        argv.extend(["-i", str(ssh_config.key)])
    if ssh_config.port != 22:
        argv.extend(["-p", str(ssh_config.port)])
    user_host = f"{ssh_config.user}@{blade}" if ssh_config.user else blade
    argv.extend([user_host, executable_path, "serve"])
    return argv


def default_client_factory(
    binary_path: Path | str = "arb_evaluator_ld",
    *,
    ssh_config: SSHConfig | None = None,
    timeout: float = 300.0,
) -> ClientFactory:
    """Create the same typed protocol client over local or SSH process launch."""
    executable = str(binary_path)

    def factory(blade: str, policy_identity: str) -> EvaluatorClient:
        if blade == "local" or not blade:
            return EvaluatorClient(
                executable_path=executable,
                expected_policy_id=policy_identity,
                timeout=timeout,
            )

        config = ssh_config or SSHConfig()
        return EvaluatorClient(
            executable_path=executable,
            expected_policy_id=policy_identity,
            launch_argv=_ssh_launch_argv(blade, executable, config),
            verify_local_inputs=False,
            timeout=timeout,
        )

    return factory


class EvaluatorRegistry:
    """Maintain one persistent client for each transport/policy/session/slot tuple.

    The evaluator protocol intentionally permits only one opened session per
    process. Session identity therefore belongs in the process-cache key, while
    worker slots provide independent processes for concurrent local shards.
    """

    def __init__(
        self,
        client_factory: ClientFactory | None = None,
        *,
        binary_path: Path | str = "arb_evaluator_ld",
        ssh_config: SSHConfig | None = None,
        default_timeout: float = 300.0,
    ) -> None:
        self._factory = client_factory or default_client_factory(
            binary_path=binary_path,
            ssh_config=ssh_config,
            timeout=default_timeout,
        )
        self._clients: dict[tuple[str, str, str, int], EvaluatorClient] = {}

    def get_or_create(
        self,
        blade: str,
        policy_identity: str,
        session_identity: str,
        *,
        worker_slot: int = 0,
    ) -> EvaluatorClient:
        if not session_identity:
            raise ValueError("session_identity must be non-empty")
        if worker_slot < 0:
            raise ValueError("worker_slot must be >= 0")
        key = (blade, policy_identity, session_identity, worker_slot)
        if key not in self._clients:
            client = self._factory(blade, policy_identity)
            client.start()
            self._clients[key] = client
        return self._clients[key]

    def has_client(
        self,
        blade: str,
        policy_identity: str,
        session_identity: str,
        *,
        worker_slot: int = 0,
    ) -> bool:
        return (blade, policy_identity, session_identity, worker_slot) in self._clients

    def close_all(self) -> None:
        for (blade, policy_id, session_id, worker_slot), client in list(self._clients.items()):
            try:
                client.shutdown()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Error shutting down evaluator client (%s, %s, %s, slot %s): %s",
                    blade,
                    policy_id,
                    session_id,
                    worker_slot,
                    exc,
                )
        self._clients.clear()

    def __enter__(self) -> "EvaluatorRegistry":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close_all()


__all__ = ["ClientFactory", "EvaluatorRegistry", "default_client_factory"]
