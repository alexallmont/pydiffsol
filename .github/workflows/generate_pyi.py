# Regenerate wheel with pyi autocomplete file created via pydiffsol
#
# This is a non-standard approach to pyo3_stub_gen, building a standalone app with a
# separate src/bin/stub_gen.rs binary. We build pyi on CI to accommodate differences
# between platforms, and this does not work with the stub_gen.rs approach because on
# some platforms it can't be built in the BEFORE or REPAIR CIBW_ hooks because those
# are not running in same environment as build wheel itself, causing linker issues.
#
# Instead we provide a private _generate_pyi method in pydiffsol which is equivalent
# to the src/bin/stub_gen.rs approach. This requires the CARGO_MANIFEST_DIR variable
# to be set for introspection, see the usage example below.
#
# This also does some post processing on the pyi file to append pydiffsol enums like
# `ds.bdf` and `ds.f64` because pyo3_stub_gen does not support this natively.
#
from pathlib import Path
import pydiffsol as ds
import shutil
import subprocess
import sys


def generate_pydiffsol_pyi():
    # Generate bulk of pydiffsol.pyi using library method
    ds._generate_pyi()

    # Append common enums to end of pyi file
    print("Amending enums to .pyi")
    with open("pydiffsol.pyi", "a") as pyi_file:
        enums = [ds.JitBackendType, ds.LinearSolverType, ds.MatrixType, ds.OdeSolverType, ds.ScalarType]
        for enum in enums:
            for member in enum.all():
                print(f"{member} = {member.__class__.__name__}.{member}", file=pyi_file)


def repackage_with_pyi(wheel, dest_dir):
    generate_pydiffsol_pyi()

    # Rebuild in a temp work dir
    work_dir = Path("/tmp/wheel_add_pyi")
    work_dir.mkdir(parents=True, exist_ok=True)

    # Unpack, add pyi file, then repackage
    subprocess.run(["wheel", "unpack", str(wheel), "-d", str(work_dir)], check=True)
    pkg = next(work_dir.iterdir())
    shutil.copy(
        "pydiffsol.pyi",
        pkg / "pydiffsol" / "pydiffsol.pyi",
    )
    subprocess.run(["wheel", "pack", str(pkg), "-d", str(dest_dir)], check=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  CARGO_MANIFEST_DIR=. python .github/workflows/generate_pyi.py <wheel> <dest_dir>")
        exit()

    wheel = Path(sys.argv[1])
    dest_dir = Path(sys.argv[2])
    repackage_with_pyi(wheel, dest_dir)
