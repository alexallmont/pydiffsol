// Cargo build script embeds the resolved diffsol/diffsl versions from Cargo.lock into compile-time
// environment variables for the Python module.

use cargo_lock::Lockfile;
use std::env;
use std::path::Path;

fn package_version(lockfile: &Lockfile, name: &str) -> Option<String> {
    lockfile
        .packages
        .iter()
        .find(|package| package.name.as_str() == name)
        .map(|package| package.version.to_string())
}

fn emit_rerun_rules(lockfile_path: &Path) {
    println!("cargo:rerun-if-changed={}", lockfile_path.display());
}

fn emit_diffsol_versions(lockfile_path: &Path) {
    // Save versions to rustc-env values to be baked into Python API at compile time
    let (diffsol_version, diffsl_version) = match Lockfile::load(&lockfile_path) {
        Ok(lockfile) => (
            package_version(&lockfile, "diffsol"),
            package_version(&lockfile, "diffsl"),
        ),
        Err(_) => (None, None),
    };

    println!(
        "cargo:rustc-env=PYDIFFSOL_DIFFSOL_VERSION={}",
        diffsol_version.as_deref().unwrap_or("unknown")
    );
    println!(
        "cargo:rustc-env=PYDIFFSOL_DIFFSL_VERSION={}",
        diffsl_version.as_deref().unwrap_or("unknown")
    );
}

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is not set");
    let lockfile_path = Path::new(&manifest_dir).join("Cargo.lock");

    emit_rerun_rules(&lockfile_path);
    emit_diffsol_versions(&lockfile_path);
}
