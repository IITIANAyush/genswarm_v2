"""
core/loader.py
──────────────
Loads the generated controller.py and returns controller_step.

main.py does:
    from core.loader import load_controller
    controller_fn = load_controller()

This module finds the generated file automatically so main.py needs
zero knowledge of where the file lives.
"""
import importlib.util
import os
import sys


def load_controller(path: str = None) -> callable:
    """
    Load workspace/generated/controller.py and return its controller_step.

    Parameters
    ----------
    path : str, optional
        Explicit path to controller.py.  If omitted, searches the standard
        output location: <project_root>/workspace/generated/controller.py

    Returns
    -------
    callable — the controller_step function
    """
    if path is None:
        # Walk up from this file's directory to find workspace/generated/
        here        = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(here)          # core/ → project root
        path        = os.path.join(project_root, "workspace", "generated", "controller.py")

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Controller not found at: {path}\n"
            f"Run without --skip-gen first to generate a controller."
        )

    # Use importlib so the module gets its own fully-populated globals
    # (numpy, assign_goals, _hungarian_assign, etc.).
    # This avoids the 'np not defined' error that bare exec() causes.
    spec   = importlib.util.spec_from_file_location("genswarm_controller", path)
    module = importlib.util.module_from_spec(spec)

    # Register in sys.modules so relative references inside controller.py work
    sys.modules["genswarm_controller"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "controller_step"):
        raise AttributeError(
            f"controller.py at {path} has no controller_step() function.\n"
            f"Re-run generation to rebuild the controller."
        )

    return module.controller_step
