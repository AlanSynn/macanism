import abc
import math
import numpy as np
import logging
from typing import List, Tuple, Any

# Assuming Anchor and Bar are refactored (e.g., using dataclasses)
from .anchor import Anchor
from .bar import Bar

logger = logging.getLogger(__name__)


class MotionStudy(abc.ABC):
    """Abstract base class for motion studies of linkage systems."""

    def __init__(self, drive1: Anchor, drive2: Anchor, bar1: Bar, bar2: Bar):
        """
        Initializes the MotionStudy.

        Args:
            drive1: The first driving anchor.
            drive2: The second driving anchor.
            bar1: The first bar connected to drive1.
            bar2: The second bar connected to drive2 and bar1.
        """
        # Store components
        self.drive1 = drive1
        self.drive2 = drive2
        self.bar1 = bar1
        self.bar2 = bar2

        # Parameters determined by drive speeds
        self.resolution: int = 1080 # Increased from 360
        self.total_frames: int = self.resolution
        self.step_size: float = 1.0

        self._calculate_speed_params()

    @abc.abstractmethod
    def get_members(self) -> List[str]:
        """Returns a list of component names managed by this study."""
        pass

    @abc.abstractmethod
    def set_value(self, member: str, parameter: str, value: Any) -> None:
        """Sets a parameter value for a specific component member."""
        pass

    def _calculate_speed_params(self) -> None:
        """Calculates total frames and step size based on drive speeds."""
        speed1 = abs(self.drive1.speed) # Use absolute speed for cycle calculation
        speed2 = abs(self.drive2.speed)

        if speed1 == 0 or speed2 == 0:
            logger.warning("One or both drive speeds are zero. Using default resolution.")
            self.total_frames = self.resolution
            self.step_size = 1.0 if self.total_frames > 0 else 0.0 # Avoid division by zero
            return

        if speed1 == speed2:
            # Equal speeds: one cycle completes in base resolution
            self.total_frames = self.resolution
        else:
            # Use LCM concept based on GCD for differing speeds
            # Convert speeds to integers if they represent gear ratios or similar discrete steps
            # If speeds are continuous rates, this calculation might need adjustment.
            # For now, assuming they represent relative whole number steps per cycle.
            try:
                # Attempt conversion to int; handle potential non-integer speeds
                int_speed1 = int(speed1)
                int_speed2 = int(speed2)
                if int_speed1 != speed1 or int_speed2 != speed2:
                    logger.warning("Non-integer speeds detected. Calculation based on integer part.")

                common_divisor = math.gcd(int_speed1, int_speed2)
                # Least Common Multiple (LCM) * resolution / GCD gives total frames for cycle repeat
                # LCM(a,b) = (|a*b|) / GCD(a,b)
                self.total_frames = abs(int_speed1 * int_speed2) // common_divisor * self.resolution
            except TypeError:
                 logger.error("Could not determine GCD for speeds. Using default resolution.")
                 self.total_frames = self.resolution

        # Calculate step size: total angle range (total_frames) / number of steps (resolution)
        self.step_size = float(self.total_frames) / self.resolution if self.resolution > 0 else 0.0
        logger.debug(f"Calculated speed params: total_frames={self.total_frames}, step_size={self.step_size}")

    @staticmethod
    def sides_to_angle(a: float, b: float, c: float) -> float:
        """Calculates angle C of a triangle using the Law of Cosines.

        Args:
            a: Length of side opposite angle A.
            b: Length of side opposite angle B.
            c: Length of side opposite angle C (the side between A and B).

        Returns:
            The angle C in radians.

        Raises:
            ValueError: If the sides cannot form a valid triangle (triangle inequality).
        """
        # Check triangle inequality (ensure floating point comparisons are safe)
        eps = 1e-9 # Small tolerance for floating point checks
        if a + b <= c + eps or a + c <= b + eps or b + c <= a + eps:
            raise ValueError(f"Sides ({a}, {b}, {c}) violate triangle inequality.")

        # Law of Cosines: c^2 = a^2 + b^2 - 2ab * cos(C)
        # => cos(C) = (a^2 + b^2 - c^2) / (2ab)
        # Clamp the argument to arccos to [-1, 1] to avoid domain errors due to float precision
        cos_C_arg = (a**2 + b**2 - c**2) / (2.0 * a * b)
        cos_C_clamped = np.clip(cos_C_arg, -1.0, 1.0)

        angle_C_rad = np.arccos(cos_C_clamped)

        return angle_C_rad

    @staticmethod
    def line_end(start_x: float, start_y: float, length: float, angle_rad: float) -> Tuple[float, float]:
        """Calculates the end point of a line segment.

        Args:
            start_x: x-coordinate of the starting point.
            start_y: y-coordinate of the starting point.
            length: The length of the line segment.
            angle_rad: The angle of the line segment in radians (relative to the positive x-axis).

        Returns:
            A tuple (end_x, end_y) representing the coordinates of the end point.
        """
        end_x = start_x + np.cos(angle_rad) * length
        end_y = start_y + np.sin(angle_rad) * length
        return end_x, end_y