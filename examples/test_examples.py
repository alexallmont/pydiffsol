import json5
import pathlib
import pytest
import re
import subprocess
import sys


BASE_DIR = pathlib.Path(__file__).parent.parent


def all_example_paths():
    """
    Return all <n>_<n>_<examplename>.py filenames in examples dir
    """
    file_regex = re.compile(r"^\d+_\d+_.*\.py$")
    files = [
        p for p in (BASE_DIR / "examples").iterdir()
        if file_regex.match(p.name)
    ]
    return sorted(files)


@pytest.fixture(params=all_example_paths(), ids=lambda p: p.name)
def example_path(request):
    """
    Parameterised fixture per example
    """
    return request.param


@pytest.fixture(scope="session")
def launch_json_options():
    """
    Get options json block for `exampleFile` in launch.json, used
    to select which example to run in debug.
    """
    with (BASE_DIR / ".vscode" / "launch.json").open() as f:
        data = json5.load(f) # json5 allows for looser vscode format
        for input in data["inputs"]:
            if input["id"] == "exampleFile":
                return input["options"]
        raise RuntimeError("no exampleFile found in launch.json")


def test_example_smoke_test(example_path):
    """
    Smoke test per example - check each runs with no exceptions
    """
    proc = subprocess.run(
        [sys.executable, str(example_path)],
        capture_output=True,
        text=True
    )

    if proc.returncode != 0:
        pytest.fail(
            f"Example {example_path.name} exited with code {proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )


def test_example_in_launch_json(example_path, launch_json_options):
    """
    Check each example is launchable in vscode debug
    """
    assert f"examples/{example_path.name}" in launch_json_options
