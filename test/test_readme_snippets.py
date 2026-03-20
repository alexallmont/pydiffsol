import os
import pytest
import re
import subprocess
import sys
import textwrap

from docutils import nodes
from docutils.core import publish_doctree
from pathlib import Path


FENCE_REGEX = re.compile(r"^```(:?py|python)\s*\n(.*?)\n```", re.S | re.M)


def extract_md_python_blocks(text: str):
    return [textwrap.dedent(m.group(2)) for m in FENCE_REGEX.finditer(text)]


def extract_rst_python_blocks(text: str):
    doctree = publish_doctree(text)
    blocks = []

    for node in doctree.findall(nodes.literal_block):
        classes = set(node.get("classes", []))
        language = node.get("language", "")

        if language in {"python", "py"} or {"code", "python"} <= classes or {"code", "py"} <= classes:
            blocks.append(node.astext())

    return blocks


def run_python_block(code: str, script_path: Path):
    script_path.write_text(code, encoding="utf8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    return subprocess.run(
        [sys.executable, str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        timeout=10,
    )


@pytest.mark.parametrize(
    "md_path",
    sorted(Path(".").rglob("*.md")),
    ids=lambda p: p.name
)
def test_md_python_snippets_run(md_path, tmp_path):
    content = md_path.read_text(encoding="utf8")
    blocks = extract_md_python_blocks(content)
    if not blocks:
        pytest.skip(f"No md python code blocks found in {md_path}")

    for idx, code in enumerate(blocks):
        script_path = tmp_path / f"{md_path.name.replace('.', '_')}_block{idx}.py"
        proc = run_python_block(code, script_path)

        out = proc.stdout.decode(errors="replace")
        if proc.returncode != 0:
            pytest.fail(
                f"Snippet #{idx} in {md_path} exited {proc.returncode}.\n--- code ---\n{code}\n--- output ---\n{out}"
            )


@pytest.mark.parametrize(
    "rst_path",
    sorted(Path("docs").rglob("*.rst")),
    ids=lambda p: str(p),
)
def test_rst_python_snippets_run(rst_path, tmp_path):
    content = rst_path.read_text(encoding="utf8")
    blocks = extract_rst_python_blocks(content)
    if not blocks:
        pytest.skip(f"No rst python code blocks found in {rst_path}")

    for idx, code in enumerate(blocks):
        script_path = tmp_path / f"{rst_path.as_posix().replace('/', '_').replace('.', '_')}_block{idx}.py"
        proc = run_python_block(code, script_path)

        out = proc.stdout.decode(errors="replace")
        if proc.returncode != 0:
            pytest.fail(
                f"Snippet #{idx} in {rst_path} exited {proc.returncode}.\n--- code ---\n{code}\n--- output ---\n{out}"
            )
