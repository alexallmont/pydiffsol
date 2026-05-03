from pathlib import Path
import pydiffsol as ds
import tomli


ROOT = Path(__file__).resolve().parent.parent


def package_version_from_cargo_toml() -> str:
    cargo_toml = ROOT / "Cargo.toml"
    with cargo_toml.open("rb") as f:
        data = tomli.load(f)
    return data["package"]["version"]


def package_version_from_cargo_lock(name: str) -> str:
    cargo_lock = ROOT / "Cargo.lock"
    with cargo_lock.open("rb") as f:
        data = tomli.load(f)

    for package in data["package"]:
        if package["name"] == name:
            return package["version"]

    raise AssertionError(f'Could not find package "{name}" in Cargo.lock')


def test_pydiffsol_version_matches_cargo_toml():
    assert ds.version() == package_version_from_cargo_toml()


def test_dependency_versions_match_cargo_lock():
    assert ds.diffsol_version() == package_version_from_cargo_lock("diffsol")
    assert ds.diffsl_version() == package_version_from_cargo_lock("diffsl")
