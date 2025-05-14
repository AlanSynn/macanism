import numpy as np
from typing import Tuple, Literal
from dataclasses import dataclass

@dataclass
class Anchor:
    """Represents a fixed anchor point with a rotating drive.

    Attributes:
        x: The x-coordinate of the anchor center.
        y: The y-coordinate of the anchor center.
        r: The radius of the drive crank.
        speed: The rotational speed (multiplier for input angle). Defaults to 1.
        initial: The initial angle offset in degrees. Defaults to 0.
    """
    x: float
    y: float
    r: float
    speed: float = 1.0
    initial: float = 0.0

    @property
    def xy(self) -> Tuple[float, float]:
        """Returns the (x, y) coordinates of the anchor center."""
        return (self.x, self.y)

    def base_point(self, angle: float) -> Tuple[float, float]:
        """Calculates the attachment point coordinates for a given angle.

        The base point is where the linkage attaches to the rotating drive crank.

        Args:
            angle: The input angle in degrees (relative to the start).

        Returns:
            A tuple (x, y) representing the coordinates of the attachment point.
        """
        # Adjust angle based on speed direction and initial offset
        effective_angle = self.initial + angle * self.speed

        # Calculate coordinates using trigonometry
        point_x = (self.r * self._deg_to_x(effective_angle)) + self.x
        point_y = (self.r * self._deg_to_y(effective_angle)) + self.y

        return point_x, point_y

    def set_value(self, parameter: Literal['x', 'y', 'r', 'speed', 'initial'], value: float) -> None:
        """Sets the value of a specified parameter.

        Args:
            parameter: The name of the parameter to set.
            value: The new value for the parameter.

        Raises:
            ValueError: If the parameter name is invalid.
            TypeError: If the value type is incompatible (implicitly handled by type hints).
        """
        if hasattr(self, parameter):
            # Basic type check could be added here if needed, beyond dataclass typing
            setattr(self, parameter, value)
        else:
            raise ValueError(f"Parameter '{parameter}' does not exist for Anchor!")

    def distance_angle_from(self, target_x: float, target_y: float) -> Tuple[float, float]:
        """Calculates the distance and angle from this anchor to a target point.

        Args:
            target_x: The x-coordinate of the target point.
            target_y: The y-coordinate of the target point.

        Returns:
            A tuple containing (distance, angle in radians).
        """
        delta_x = self.x - target_x
        delta_y = self.y - target_y
        angle_rad = self._xy_to_angle(delta_x, delta_y)
        distance = self._xy_to_hyp(delta_x, delta_y)

        return distance, angle_rad

    def base_point_distance(self, angle1: float, angle2: float, end_anchor: 'Anchor') -> Tuple[float, float]:
        """Calculates the distance and angle between the base points of two anchors.

        Args:
            angle1: The angle for this anchor's base point (degrees).
            angle2: The angle for the end_anchor's base point (degrees).
            end_anchor: The other Anchor object.

        Returns:
            A tuple containing (distance, angle in radians) between the base points.
        """
        start_point = self.base_point(angle1)
        end_point = end_anchor.base_point(angle2)

        delta_x = start_point[0] - end_point[0]
        delta_y = start_point[1] - end_point[1]

        angle_rad = self._xy_to_angle(delta_x, delta_y)
        distance = self._xy_to_hyp(delta_x, delta_y)

        return distance, angle_rad

    # Private static methods for trigonometric calculations
    @staticmethod
    def _deg_to_x(angle: float) -> float:
        """Converts degrees to the x-component of a unit vector."""
        return np.cos(np.deg2rad(angle))

    @staticmethod
    def _deg_to_y(angle: float) -> float:
        """Converts degrees to the y-component of a unit vector."""
        return np.sin(np.deg2rad(angle))

    @staticmethod
    def _xy_to_angle(x: float, y: float) -> float:
        """Calculates the angle (in radians) of a vector defined by x, y."""
        # Use atan2 for quadrant correctness and handling x=0
        return np.arctan2(y, x)

    @staticmethod
    def _xy_to_hyp(x: float, y: float) -> float:
        """Calculates the hypotenuse (magnitude) of a vector defined by x, y."""
        # np.hypot is numerically stable
        return np.hypot(x, y)