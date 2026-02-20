from .pydiffsol import *
from .multi_step_ode import MultiStepOde, MultiStepOdeConfig

__doc__ = pydiffsol.__doc__

if hasattr(pydiffsol, "__all__"):
	__all__ = list(pydiffsol.__all__) + ["MultiStepOde", "MultiStepOdeConfig"]
else:
	__all__ = [name for name in globals() if not name.startswith("_")]
