"""
This import is an aid for debugging when running the example code via LLDB, for
example when debugging Rust internals via `launch.json` in VSCode. In macos, the
matplotlib import fails with an error about setting view as first responder for
window. The workaround below forces a headless backend.
"""
import sys

if sys.platform == "darwin":
    # Force headless on macos
    import matplotlib
    matplotlib.use("Agg")
