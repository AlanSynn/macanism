import numpy as np
import logging
from typing import Tuple, List, Any, Dict

from .motionstudy import MotionStudy
from .bar import Bar
from .anchor import Anchor

logger = logging.getLogger(__name__)

class TwoBar(MotionStudy):
    """Represents a two-bar linkage system driven by two anchors.

    Inherits from MotionStudy to provide basic calculation methods.
    """
    def __init__(self, drive1: Anchor, drive2: Anchor, bar1: Bar, bar2: Bar):
        """
        Initializes the TwoBar system.

        Args:
            drive1: The first driving anchor.
            drive2: The second driving anchor.
            bar1: The bar connected to drive1.
            bar2: The bar connected to drive2 and bar1 (at bar1.joint).

        Raises:
            ValueError: If the initial configuration is physically impossible.
        """
        # Validate first before calling super().__init__ which might use properties
        self.validate_physics(drive1, drive2, bar1, bar2)
        super().__init__(drive1, drive2, bar1, bar2)
        logger.debug(f"Initialized TwoBar system: {drive1}, {drive2}, {bar1}, {bar2}")

    def get_members(self) -> List[str]:
        """Returns the list of component names in this system."""
        return ['drive1', 'drive2', 'bar1', 'bar2']

    def set_value(self, member: str, parameter: str, value: Any) -> None:
        """Sets a parameter value for a specific component member.

        Args:
            member: The name of the component ('drive1', 'drive2', 'bar1', 'bar2').
            parameter: The name of the parameter within the component.
            value: The new value for the parameter.

        Raises:
            ValueError: If the member name is invalid or the component doesn't have the parameter.
        """
        try:
            component = getattr(self, member)
            component.set_value(parameter, value)
            # Recalculate speed dependent parameters if speed changes
            if member in ['drive1', 'drive2'] and parameter == 'speed':
                self._calculate_speed_params()
            # Optionally re-validate physics if geometry changes
            if member in ['drive1', 'drive2', 'bar1', 'bar2']:
                self.validate_physics(self.drive1, self.drive2, self.bar1, self.bar2)
        except AttributeError:
            raise ValueError(f"Member '{member}' does not exist!") from None
        # Let component's set_value raise ValueError for invalid parameters

    @staticmethod
    def validate_physics(drive1: Anchor, drive2: Anchor, bar1: Bar, bar2: Bar) -> None:
        """Checks if the given linkage configuration is physically possible.

        Args:
            drive1: The first driving anchor.
            drive2: The second driving anchor.
            bar1: The bar connected to drive1.
            bar2: The bar connected to drive2 and bar1.

        Raises:
             ValueError: If the linkage cannot connect or reach.
        """
        dist_anchors, _ = drive1.distance_angle_from(drive2.x, drive2.y)

        # Maximum possible reach from drive1 anchor center
        max_reach = drive1.r + bar1.joint + bar2.length
        # Minimum possible distance to drive2 anchor center needed for connection
        min_dist_needed = dist_anchors - drive2.r

        if max_reach < min_dist_needed - 1e-9: # Add tolerance
            raise ValueError(
                f"Linkage cannot reach: Max reach ({max_reach:.3f}) is less than "
                f"min distance needed ({min_dist_needed:.3f}) between anchor centers."
            )

        # Minimum possible reach from drive1 anchor center (when bars fold back)
        min_reach = abs(drive1.r + bar1.joint - bar2.length)
        # Maximum possible distance to drive2 anchor center allowing connection
        max_dist_allowed = dist_anchors + drive2.r

        if min_reach > max_dist_allowed + 1e-9: # Add tolerance
            raise ValueError(
                f"Linkage cannot connect (too long/folded): Min reach ({min_reach:.3f}) is greater than "
                f"max distance allowed ({max_dist_allowed:.3f}) between anchor centers."
            )
        logger.debug("Physics validation passed.")

    def end_path(self, input_angle: float) -> Tuple[float, float]:
        """Calculates the end effector point coordinates for a given input angle.

        Args:
            input_angle: The input angle (e.g., in degrees) driving the system.
                         Typically corresponds to the rotation of drive1 if speeds are relative.

        Returns:
            A tuple (x, y) representing the coordinates of the end effector.
            The end effector is located at `bar1.joint` distance along `bar1`.

        Raises:
            ValueError: If the geometry becomes impossible at this angle (e.g., cannot form triangle).
        """
        angle1 = input_angle
        drive1_speed = self.drive1.speed if self.drive1.speed != 0 else 1.0
        # Calculate effective angle for drive2 based on relative speeds
        angle2 = input_angle * (self.drive2.speed / drive1_speed)

        # Position of the point where bar1 connects to drive1's anchor arm
        base1_x, base1_y = self.drive1.base_point(angle1)

        # Distance and angle of the line connecting the two driven points (bar1_base and bar2_base)
        dist_base_points, angle_base_points_rad = self.drive1.base_point_distance(angle1, angle2, self.drive2)

        try:
            # Calculate angle at base1 within the triangle formed by (base1, base2, bar1-bar2 joint)
            # Here, 'a' is the length of bar1 (self.bar1.length) in the triangle
            angle_at_base1_rad = self.sides_to_angle(
                a=self.bar1.length,
                b=dist_base_points,
                c=self.bar2.length
            )
        except ValueError as e:
            error_msg = (
                f"Cannot form linkage triangle at input angle {input_angle:.2f} (deg): {e}\n"
                f"Dist base points: {dist_base_points:.2f}, Bar1.length: {self.bar1.length:.2f}, Bar2.length: {self.bar2.length:.2f}"
            )
            # logger.error(error_msg) # Using logger from the top of the file
            # For consistency with get_linkage_positions, return NaN on calculation failure
            # raise ValueError(error_msg) from e
            return np.nan, np.nan

        # Absolute angle of bar1 in the coordinate system
        # This is the orientation of the segment of bar1 that has length self.bar1.length
        absolute_angle_of_bar1_rad = angle_base_points_rad + angle_at_base1_rad # Assuming elbow up configuration

        # The end effector is located at distance `self.bar1.joint` along bar1
        end_effector_x, end_effector_y = self.line_end(
            start_x=base1_x,
            start_y=base1_y,
            length=self.bar1.joint, # Use bar1.joint for the end effector position
            angle_rad=absolute_angle_of_bar1_rad
        )

        # Optional debug logging for clarity during development
        # if abs(input_angle) < 10 or abs(input_angle - 180) < 10 or abs(input_angle - 350) < 10 : # Log specific angles
        #     logger.debug(
        #         f"end_path({input_angle=:.1f}) -> "
        #         f"b1=({base1_x:.1f},{base1_y:.1f}), "
        #         f"dist_bp={dist_base_points:.1f}, ang_bp={np.rad2deg(angle_base_points_rad):.1f}, "
        #         f"ang_at_b1={np.rad2deg(angle_at_base1_rad):.1f}, "
        #         f"abs_ang_bar1={np.rad2deg(absolute_angle_of_bar1_rad):.1f}, "
        #         f"EE=({end_effector_x:.1f}, {end_effector_y:.1f}) using bar1.joint={self.bar1.joint}"
        #     )

        return end_effector_x, end_effector_y

    def get_linkage_positions(self, input_angle: float) -> Dict[str, Tuple[float, float]]:
        """
        Calculates the positions of all key points in the linkage for a given input angle.

        Args:
            input_angle: The input angle in degrees.

        Returns:
            A dictionary containing the (x, y) coordinates for:
            'anchor1': Position of drive1 center.
            'anchor2': Position of drive2 center.
            'driven1': Position of the point on drive1 connected to bar1.
            'driven2': Position of the point on drive2 connected to bar2.
            'joint': Position of the joint connecting bar1 and bar2.
            'end_effector': Position of the end effector.
            Returns NaNs for coordinates if the configuration is impossible.
        """
        angle_rad = np.radians(input_angle)

        # Anchor positions (fixed)
        anchor1_pos = (self.drive1.x, self.drive1.y)
        anchor2_pos = (self.drive2.x, self.drive2.y)

        # Calculate current positions of driven points on anchors
        x1_driven = self.drive1.x + self.drive1.r * np.cos(np.radians(self.drive1.initial) + self.drive1.speed * angle_rad)
        y1_driven = self.drive1.y + self.drive1.r * np.sin(np.radians(self.drive1.initial) + self.drive1.speed * angle_rad)
        driven1_pos = (x1_driven, y1_driven)

        x2_driven = self.drive2.x + self.drive2.r * np.cos(np.radians(self.drive2.initial) + self.drive2.speed * angle_rad)
        y2_driven = self.drive2.y + self.drive2.r * np.sin(np.radians(self.drive2.initial) + self.drive2.speed * angle_rad)
        driven2_pos = (x2_driven, y2_driven)

        # --- Joint Calculation (same logic as end_path) ---
        dist_driven = np.hypot(x1_driven - x2_driven, y1_driven - y2_driven)

        nan_pos = (np.nan, np.nan)
        if dist_driven > (self.bar1.length + self.bar2.length) or \
           dist_driven < abs(self.bar1.length - self.bar2.length) or \
           dist_driven == 0:
            joint_pos = nan_pos
            end_effector_pos = nan_pos
        else:
            try:
                angle_at_driven1 = np.arccos(
                    np.clip( # Use np.clip to avoid domain errors due to floating point inaccuracies
                        (self.bar1.length**2 + dist_driven**2 - self.bar2.length**2) /
                        (2 * self.bar1.length * dist_driven),
                        -1.0, 1.0
                    )
                )
                angle_driven_points = np.arctan2(y2_driven - y1_driven, x2_driven - x1_driven)

                # Using '+' elbow configuration
                joint_angle = angle_driven_points + angle_at_driven1
                x_joint = x1_driven + self.bar1.length * np.cos(joint_angle)
                y_joint = y1_driven + self.bar1.length * np.sin(joint_angle)
                joint_pos = (x_joint, y_joint)

                # --- End Effector Calculation (same logic as end_path) ---
                vec_driven1_joint_x = x_joint - x1_driven
                vec_driven1_joint_y = y_joint - y1_driven
                norm_driven1_joint = np.hypot(vec_driven1_joint_x, vec_driven1_joint_y)

                if norm_driven1_joint == 0:
                    end_effector_pos = nan_pos
                else:
                    unit_vec_x = vec_driven1_joint_x / norm_driven1_joint
                    unit_vec_y = vec_driven1_joint_y / norm_driven1_joint
                    x_end = x1_driven + self.bar1.joint * unit_vec_x
                    y_end = y1_driven + self.bar1.joint * unit_vec_y
                    end_effector_pos = (x_end, y_end)

            except (ValueError, ZeroDivisionError) as e:
                 logging.warning(f"Calculation error at angle {input_angle:.2f}: {e}")
                 joint_pos = nan_pos
                 end_effector_pos = nan_pos


        return {
            'anchor1': anchor1_pos,
            'anchor2': anchor2_pos,
            'driven1': driven1_pos,
            'driven2': driven2_pos,
            'joint': joint_pos,
            'end_effector': end_effector_pos
        }

    @property
    def parameters(self) -> Tuple[Any, ...]:
        """Returns a tuple of all defining parameters for this system."""
        return (
            self.bar1.length, self.bar1.joint,
            self.bar2.length,
            self.drive1.x, self.drive1.y, self.drive1.r, self.drive1.speed, self.drive1.initial,
            self.drive2.x, self.drive2.y, self.drive2.r, self.drive2.speed, self.drive2.initial
        )