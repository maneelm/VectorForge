"""Execute the real proxy startup policy, without a Docker daemon."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "frontend/docker-entrypoint.d/05-access-policy.envsh"
SHELL = shutil.which("sh") or "C:/Program Files/Git/usr/bin/sh.exe"


@pytest.mark.parametrize(
    "mode,binding,port,expected",
    [
        ("local", "127.0.0.1", "8000", True),
        ("local", "0.0.0.0", "8000", False),
        ("local", "192.168.1.5", "8000", False),
        ("unknown", "127.0.0.1", "8000", False),
        ("local", "127.0.0.1", "8000;echo unsafe", False),
        ("local", "127.0.0.1", "0", False),
        ("local", "127.0.0.1", "65536", False),
    ],
)
def test_startup_guard_fails_closed(mode, binding, port, expected):
    if not Path(SHELL).is_file():
        pytest.skip("POSIX shell not available for proxy policy test")
    environment = {
        **os.environ,
        "VECTORFORGE_ACCESS_MODE": mode,
        "VECTORFORGE_BIND_ADDRESS": binding,
        "VECTORFORGE_API_PORT": port,
    }
    process = subprocess.run(
        [SHELL, "-c", '. "$1"; printf "%s" "$VECTORFORGE_AUTH_REALM"', "policy", str(SCRIPT)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (process.returncode == 0) == expected
    if expected:
        assert process.stdout == "off"


def test_authenticated_policy_requires_secret():
    if not Path(SHELL).is_file():
        pytest.skip("POSIX shell not available for proxy policy test")
    if Path("/run/secrets/vectorforge_htpasswd").exists():
        pytest.skip("Host has an existing authentication secret")
    environment = {
        **os.environ,
        "VECTORFORGE_ACCESS_MODE": "authenticated",
        "VECTORFORGE_API_PORT": "8000",
    }
    process = subprocess.run(
        [SHELL, str(SCRIPT)], env=environment, capture_output=True, text=True, check=False
    )
    assert process.returncode != 0 and "htpasswd" in process.stderr
