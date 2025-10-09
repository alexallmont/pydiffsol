import pydiffsol as ds
import sys


def test_is_klu_available():
    # KLU currently supported on linux
    if sys.platform.startswith("linux"):
        assert ds.is_klu_available()
    elif sys.platform == "darwin":
        assert not ds.is_klu_available()
    elif sys.platform == "win32":
        assert not ds.is_klu_available()
    else:
        raise RuntimeError(f"Unsupported platform {sys.platform}")
