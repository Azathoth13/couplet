"""Published model profiles must produce portable, ciphertext-linked plans."""
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from run_benchmark import ROOT, build_plan


def _args(model, world, phase="linked", scheme="odin", replay=False):
    return Namespace(model=model, world_size=world, phase=phase, scheme=scheme,
                     data_root=Path("/tmp/odin-release-data"),
                     output=Path("/tmp/odin-release-output"), dry_run=True,
                     decode_tokens=3 if replay else 1, hot_replay=replay,
                     devices=None, cuda_arch=None)


@pytest.mark.parametrize("model", ["1b", "3b", "8b"])
@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_odin_plan_uses_matching_profile_and_real_kv(model, world):
    command, env = build_plan(_args(model, world))
    assert command[command.index("--world-size") + 1] == str(world)
    assert command[command.index("--decode-tokens") + 1] == "1"
    assert "--skip-prefill" not in command
    assert "--no-emit-kv-cache" not in command
    assert (Path(env["COUPLET_KERNEL_ROOT"]) / "keyswitch.py").is_file()
    profile = json.loads((ROOT / "configs/models" / f"{model}.json").read_text())
    for relative in profile["polynomials"].values():
        assert (ROOT / relative).is_file()
    assert "{repo}" not in str(env)
    assert "/workspace/private-source" not in str(env)


def test_prefill_keeps_kv_emit_and_replay_is_explicit():
    command, _ = build_plan(_args("1b", 8, phase="prefill"))
    assert command[command.index("--decode-tokens") + 1] == "0"
    assert "--no-emit-kv-cache" not in command
    replay, _ = build_plan(_args("1b", 8, replay=True))
    assert "--replay-decode-same-token" in replay


@pytest.mark.parametrize("model", ["1b", "3b", "8b"])
def test_sc_plan_uses_process_isolated_provider(model):
    command, env = build_plan(_args(model, 4, scheme="sc"))
    assert command[1] == str(ROOT / "baselines/sc/run.py")
    assert command[command.index("--model") + 1] == model
    assert command[-1] == "--dry-run"
    assert env == {}


def test_couplet_alias_dry_run_does_not_execute_the_backend(tmp_path):
    result = subprocess.run([
        sys.executable, str(ROOT / "run_benchmark.py"), "--model", "3b",
        "--scheme", "couplet", "--world-size", "8", "--data-root", str(tmp_path / "data"),
        "--output", str(tmp_path / "output"), "--dry-run",
    ], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["scheme"] == "couplet"
    assert plan["implementation"] == "odin"
    assert "--world-size" in plan["command"]
    assert not (tmp_path / "output").exists()
