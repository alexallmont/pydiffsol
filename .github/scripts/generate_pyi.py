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

    # Append common enums to end of pyi file whilst collating for __all__ later
    print("Collating pydiffsol enums...")
    all_enum_members = []
    all_enum_defs = []
    enums = [ds.JitBackendType, ds.LinearSolverType, ds.MatrixType, ds.OdeSolverType, ds.ScalarType]
    for enum in enums:
        for member in enum.all():
            all_enum_members.append(str(member))
            all_enum_defs.append(f"{member} = {member.__class__.__name__}.{member}")


    # Find __all__ block and replace with new enums
    print("Replacing __all__ and enums in .pyi")
    pyi_contents = open("pydiffsol.pyi").read()
    with open("pydiffsol.pyi", "w") as pyi_file:
        # Replace from leading __all__ block for guaranteed grep
        all_sentinel = "__all__ = [\n"
        replace_str = all_sentinel + "\n".join(f'    "{name}",' for name in all_enum_members) + "\n"
        new_pyi_contents = pyi_contents.replace(all_sentinel, replace_str)

        # Append the enum definition strings to the end of the file and write
        new_pyi_contents += "\n".join(all_enum_defs) + "\n"
        pyi_file.write(new_pyi_contents)


def repackage_with_pyi(wheel: Path):
    """
    Unpack the wheel, build pyi through introspection and repackage wheel
    """
    generate_pydiffsol_pyi()

    # Ensure temp work dir and dest dir exist
    work_dir = Path(tempfile.mkdtemp(prefix="wheel_add_pyi_"))
    if not work_dir.is_dir():
        raise RuntimeError(f"pyi repair temp dir not created: {work_dir}")

    # Unpack, add pyi file, then repackage
    print(f"Unpacking wheel {wheel} to {work_dir}")
    subprocess.run(["wheel", "unpack", str(wheel), "-d", str(work_dir)], check=True)

    pkg = next(work_dir.iterdir())
    print(f"Copying pydiffsol.pyi to {pkg}/pydiffsol")
    shutil.copy(
        "pydiffsol.pyi",
        pkg / "pydiffsol" / "pydiffsol.pyi",
    )

    with tempfile.TemporaryDirectory() as td:
        print(f"Packing wheel {pkg} back into to {wheel}")
        subprocess.run(["wheel", "pack", str(pkg), "-d", td], check=True)
        new_wheel = next(Path(td).glob("*.whl"))
        shutil.move(new_wheel, wheel)

    print("pydiffsol pyi repair complete!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Regenerate .pyi autocomplete stubs and write back to wheel.")
        print("Run from project root for setting CARGO_MANIFEST_DIR.")
        print("Usage:")
        print("  python .github/scripts/generate_pyi.py <wheel>")
        exit(1)

    wheel = Path(sys.argv[1])

    repackage_with_pyi(wheel)
