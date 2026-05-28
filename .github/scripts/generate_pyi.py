# Regenerate .pyi autocomplete stub for wheel
#
# The pyo3_stub_gen docs recommend creating a standalone src/bin/stub_gen.rs to
# create the pyi file. This does not work for pydiffsol because we regenerate pyi
# on CI to accommodate differences between platforms, and building the stub_gen
# binary in cibuildwheel is tricky; we would ideally do it in the BEFORE or
# REPAIR CIBW_ hooks but those are not running in same environment as build wheel
# itself which causes linker issues.
#
# Instead we provide a private _generate_pyi method in pydiffsol which is
# equivalent src/bin/stub_gen.rs approach. This can be launched from Python to
# generate pyi from the same binary and copy in as a repair step.
#
# This script also appends common pydiffsol enums like `ds.bdf` and `ds.f64` as
# pyo3_stub_gen does not support adding module attrs like this natively.
#
# Run with the CARGO_MANIFEST_DIR variable set to the repo root for introspection,
# see the usage example below.
#
from pathlib import Path
import pydiffsol as ds
import shutil
import subprocess
import sys
import tempfile


def generate_pydiffsol_pyi(project_dir: Path):
    # Generate bulk of pydiffsol.pyi using library method
    print("FIXME_DEBUG_3")
    print("FIXME_DEBUG_3_FORCE_MANIFEST_DIR")
    os.environ["CARGO_MANIFEST_DIR"]=os.getcwd()
    print(f"FIXME_DEBUG_3_CARGO_MANIFEST_DIR={os.environ['CARGO_MANIFEST_DIR']}")
    print(f"FIXME_DEBUG_3_CARGO_MANIFEST_DIR_LS={os.listdir(os.environ['CARGO_MANIFEST_DIR'])}")

    ds._generate_pyi()

    print("FIXME_DEBUG_4")

    # Append common enums to end of pyi file
    print("Amending enums to .pyi")
    with (project_dir / "pydiffsol.pyi").open("a") as pyi_file:
        enums = [ds.JitBackendType, ds.LinearSolverType, ds.MatrixType, ds.OdeSolverType, ds.ScalarType]
        for enum in enums:
            for member in enum.all():
                print(f"{member} = {member.__class__.__name__}.{member}", file=pyi_file)


def repackage_with_pyi(project_dir: Path, wheel: Path, dest_dir: Path):
    generate_pydiffsol_pyi(project_dir)

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
    print("FIXME_DEBUG_1")
    if len(sys.argv) < 4:
        print("Usage:")
        print("  CARGO_MANIFEST_DIR=. python .github/scripts/generate_pyi.py <project_dir> <wheel> <dest_dir>")
        exit()

    project_dir = Path(sys.argv[1])
    wheel = Path(sys.argv[2])
    dest_dir = Path(sys.argv[3])

    print("FIXME_DEBUG_2")
    print(f"FIXME_DEBUG_2_PROJECT_DIR={str(project_dir)}")
    print(f"FIXME_DEBUG_2_WHEEL={str(wheel)}")
    print(f"FIXME_DEBUG_2_DEST_DIR={str(dest_dir)}")
    import os
    print(f"FIXME_DEBUG_2_CWD={os.getcwd()}")
    print(f"FIXME_DEBUG_2_CWD_LS={os.listdir()}")

    if "CARGO_MANIFEST_DIR" in os.environ:
        print(f"FIXME_DEBUG_2_CARGO_MANIFEST_DIR={os.environ['CARGO_MANIFEST_DIR']}")
        print(f"FIXME_DEBUG_2_CARGO_MANIFEST_DIR_LS={os.listdir(os.environ['CARGO_MANIFEST_DIR'])}")
    else:
        print(f"FIXME_DEBUG_2_CARGO_MANIFEST_DIR NOT SET")

    repackage_with_pyi(project_dir, wheel, dest_dir)
