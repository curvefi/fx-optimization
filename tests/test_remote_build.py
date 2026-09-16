import shlex
import subprocess

from fxopt.placement import REMOTE_BASE, rebuild_shared_evaluator


def test_remote_build_provisions_depth_input_dependencies(monkeypatch):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    rebuild_shared_evaluator(
        "blade-a5",
        str(REMOTE_BASE / "curve-fx-arb-harness/build/test/arb_evaluator_f64"),
    )
    environment = shlex.split(commands[-1][-1])
    packages = set(environment[environment.index("-p") + 1:environment.index("--run")])
    assert {"minizip", "zlib"} <= packages
