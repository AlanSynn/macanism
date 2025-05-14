"""Macanism Library"""

__version__ = "0.1.0"

from .macanism_engine import MacanismEngine # This seems to be a different engine
from .common_elements import Point # Vector removed from here for now
# Import from mechanism.py (the one used by fourbarlinkage.py example)
from .mechanism import macanism, Joint, get_joints
# Import the Vector class used by the examples and core mechanism logic
from .vectors import Vector # This is the Kinematic Vector (Position/Velocity/Acceleration wrapper)
# Import from linkages.py, avoiding Joint name collision if LinkageSystem is still desired
from .linkages import Link, LinkageSystem # Removed Joint from here
from .gears import SpurGear, SpurGearParameters

# Import the new stagger module and PathGenerator
from . import stagger
from .path_generator import PathGenerator

__all__ = [
    "__version__",
    "MacanismEngine", # Keep if it's a separate public API component
    "Point",          # From common_elements
    "Vector",         # Now from .vectors
    "macanism",       # The main class from mechanism.py
    "Joint",          # The Joint class from mechanism.py
    "get_joints",     # The helper function from mechanism.py
    "Link",           # From linkages.py
    "LinkageSystem",  # From linkages.py
    "SpurGear",
    "SpurGearParameters",
    "stagger",        # Expose the stagger submodule
    "PathGenerator",  # Expose the PathGenerator class
]
