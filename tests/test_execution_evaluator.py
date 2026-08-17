"""Tests for evaluator processes keyed by blade, policy, and run session."""

from curve_fx_harness_client import EvaluatorClient

from curve_fx_sim.execution.evaluator_pool import EvaluatorRegistry, default_client_factory
from curve_fx_sim.execution.site import SSHConfig


class MockEvaluatorClient:
    def __init__(self, blade: str, policy_identity: str) -> None:
        self.blade = blade
        self.policy_identity = policy_identity
        self.started = False
        self.shutdown_called = False

    def start(self) -> None:
        self.started = True

    def shutdown(self) -> None:
        self.shutdown_called = True


def test_evaluator_registry_one_per_blade_policy_session() -> None:
    created_clients: list[MockEvaluatorClient] = []

    def mock_factory(blade: str, policy_id: str) -> MockEvaluatorClient:
        client = MockEvaluatorClient(blade, policy_id)
        created_clients.append(client)
        return client

    registry = EvaluatorRegistry(client_factory=mock_factory)

    # First lookup for (blade-1, policy-A)
    client1 = registry.get_or_create("blade-1", "policy-A", "run-A")
    assert client1.started
    assert len(created_clients) == 1

    # Second lookup for same key should return the existing instance
    client2 = registry.get_or_create("blade-1", "policy-A", "run-A")
    assert client2 is client1
    assert len(created_clients) == 1

    # Lookup for distinct key (blade-2, policy-A) should create a second instance
    client3 = registry.get_or_create("blade-2", "policy-A", "run-A")
    assert client3 is not client1
    assert len(created_clients) == 2

    # Lookup for distinct policy on same blade (blade-1, policy-B)
    client4 = registry.get_or_create("blade-1", "policy-B", "run-A")
    assert client4 is not client1
    assert len(created_clients) == 3

    # The protocol is one-session-per-process: a new run on the same blade and
    # policy must receive a fresh process.
    client5 = registry.get_or_create("blade-1", "policy-A", "run-B")
    assert client5 is not client1
    assert len(created_clients) == 4

    # Close all
    registry.close_all()
    for c in created_clients:
        assert c.shutdown_called


def test_remote_factory_uses_canonical_protocol_client() -> None:
    factory = default_client_factory(
        "/shared/arb_evaluator_ld",
        ssh_config=SSHConfig(user="worker", key=None, options=("-o", "BatchMode=yes")),
        timeout=17,
    )
    client = factory("blade-a1", "compiled-policy")

    assert isinstance(client, EvaluatorClient)
    assert client.verify_local_inputs is False
    assert client.expected_policy_id == "compiled-policy"
    assert client.launch_argv == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "worker@blade-a1",
        "/shared/arb_evaluator_ld",
        "serve",
    ]
