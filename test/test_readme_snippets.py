import os
import pytest
import re
import subprocess
import sys
import textwrap

from pathlib import Path


FENCE_REGEX = re.compile(r"^```(:?py|python)\s*\n(.*?)\n```", re.S | re.M)


def extract_python_blocks(text: str):
    return [textwrap.dedent(m.group(2)) for m in FENCE_REGEX.finditer(text)]


@pytest.mark.parametrize(
    "readme_path",
    [Path("README.md"), Path("README_PyPI.md")],
    ids=lambda p: p.name
)
def test_markdown_python_snippets_run(readme_path, tmp_path):
    # Get all py blocks in from markdown
    content = readme_path.read_text(encoding="utf8")
    blocks = extract_python_blocks(content)
    assert blocks, f"No python code fences found in {readme_path}"

    # Write each block to a tmp file for execution
    for idx, code in enumerate(blocks):
        script_path = tmp_path / f"{readme_path.name.replace('.', '_')}_block{idx}.py"
        script_path.write_text(code, encoding="utf8")

        # Ensure project root is importable by snippets
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path.cwd())
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            timeout=10,
        )

        out = proc.stdout.decode(errors="replace")
        if proc.returncode != 0:
            pytest.fail(
                f"Snippet #{idx} in {readme_path} exited {proc.returncode}.\n--- code ---\n{code}\n--- output ---\n{out}"
            )
