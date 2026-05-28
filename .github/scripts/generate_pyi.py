# Regenerate .pyi autocomplete stub for wheel
#
# The pyo3_stub_gen docs recommend creating a standalone src/bin/stub_gen.rs to
# create the pyi file. This does not work for pydiffsol because we regenerate pyi
# on CI to accommodate differences between platforms, and building the stub_gen
# binary in cibuildwheel is tricky; we would ideally do it in the BEFORE or
# REPAIR CIBW_ hooks but those do not run in same environment as build wheel
# itself, causing linker discrepancies and errors.
#
# Instead we provide a private _generate_pyi method in pydiffsol which is
# equivalent src/bin/stub_gen.rs approach. This can be launched from Python to
# generate pyi from the same binary and copy in as a repair step.
#
# This script also appends common pydiffsol enums like `ds.bdf` and `ds.f64` as
# pyo3_stub_gen does not support adding module attrs like this natively.
#
# This script expects to be run in the project root as it sets CARGO_MANIFEST_DIR
# from the current working dir. This was found to be the best approach to support
# all platforms; Windows builds had trouble picking this up externally.
#
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pydiffsol as ds


def generate_pydiffsol_pyi():
    """
    Regen base .pyi file and append pydiffsol enums
    """
    # Generate bulk of pydiffsol.pyi using library method
    print(f"Setting CARGO_MANIFEST_DIR to {os.getcwd()}")
    os.environ["CARGO_MANIFEST_DIR"]=os.getcwd()
    ds._generate_pyi()

    # Append common enums to end of pyi file
    print("Amending enums to .pyi")
    with open("pydiffsol.pyi", "a") as pyi_file:
        enums = [ds.JitBackendType, ds.LinearSolverType, ds.MatrixType, ds.OdeSolverType, ds.ScalarType]
        for enum in enums:
            for member in enum.all():
                print(f"{member} = {member.__class__.__name__}.{member}", file=pyi_file)


def repackage_with_pyi(wheel: Path, dest_dir: Path):
    """
    Unpack the wheel, build pyi through introspection and repackage wheel
    """
    generate_pydiffsol_pyi()

    # Ensure temp work dir and dest dir exist
    work_dir = Path(tempfile.mkdtemp(prefix="wheel_add_pyi_"))
    if not work_dir.is_dir():
        raise RuntimeError(f"pyi repair temp dir not created: {work_dir}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Unpack, add pyi file, then repackage
    print(f"Unpacking wheel {wheel} to {work_dir}")
    subprocess.run(["wheel", "unpack", str(wheel), "-d", str(work_dir)], check=True)

    pkg = next(work_dir.iterdir())
    print(f"Copying pydiffsol.pyi to {pkg}/pydiffsol")
    shutil.copy(
        "pydiffsol.pyi",
        pkg / "pydiffsol" / "pydiffsol.pyi",
    )

    print(f"Packing wheel {pkg} to {dest_dir}")
    subprocess.run(["wheel", "pack", str(pkg), "-d", str(dest_dir)], check=True)

    print("pydiffsol pyi repair complete!")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Regenerate .pyi autocomplete stubs for pydiffsol.")
        print("Run from project root for setting CARGO_MANIFEST_DIR.")
        print("Usage:")
        print("  python .github/scripts/generate_pyi.py <wheel> <dest_dir>")
        exit()

    wheel = Path(sys.argv[1])
    dest_dir = Path(sys.argv[2])

    repackage_with_pyi(wheel, dest_dir)
