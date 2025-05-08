"""
Module for defining and generating gear geometries.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import math
import logging

from .common_elements import Point # Gears will be defined by points on their profile

logger = logging.getLogger(__name__)

@dataclass
class SpurGearParameters:
    """Parameters defining a spur gear."""
    module: float | None = None
    diametral_pitch: float | None = None
    num_teeth: int = 20
    pressure_angle_deg: float = 20.0
    # Convenience properties will convert between module and diametral_pitch

    def __post_init__(self):
        if self.module is None and self.diametral_pitch is None:
            raise ValueError("Either module or diametral_pitch must be specified.")
        if self.module is not None and self.diametral_pitch is not None:
            # Check for consistency if both are provided, or prioritize one
            # For now, let's assume if module is given, it's the primary
            self.diametral_pitch = 25.4 / self.module
        elif self.module is None and self.diametral_pitch is not None:
            self.module = 25.4 / self.diametral_pitch
        elif self.diametral_pitch is None and self.module is not None:
            self.diametral_pitch = 25.4 / self.module

    @property
    def pitch_diameter(self) -> float:
        """Calculates the pitch diameter."""
        if self.module is not None:
            return self.module * self.num_teeth
        # self.diametral_pitch should be non-None here due to __post_init__
        return self.num_teeth / self.diametral_pitch

    @property
    def base_diameter(self) -> float:
        """Calculates the base circle diameter."""
        return self.pitch_diameter * math.cos(math.radians(self.pressure_angle_deg))

class SpurGear:
    """
    Represents a spur gear and provides methods for generating its tooth profile.
    """
    def __init__(self, params: SpurGearParameters, name: str = "SpurGear"):
        self.name = name
        self.params = params
        self.tooth_profile_points: List[Point] = [] # List of points defining one tooth flank
        logger.info(f"SpurGear '{self.name}' created with {self.params.num_teeth} teeth.")

    def generate_tooth_profile(self) -> None:
        """
        Generates the involute tooth profile points.
        (This is a placeholder - actual implementation is complex).
        """
        logger.info(f"Generating tooth profile for {self.name}...")
        # Placeholder for involute curve generation algorithm
        # This would involve calculating points on the involute curve from the base circle
        # up to the addendum circle, considering tooth thickness.

        # Example: add a few dummy points for one flank
        if not self.tooth_profile_points: # Avoid re-generating if already done
            base_radius = self.params.base_diameter / 2.0
            pitch_radius = self.params.pitch_diameter / 2.0
            # For simplicity, let's add points relative to the gear center
            # and for a single tooth centered on the x-axis for now.
            # Actual involute generation is much more involved.
            self.tooth_profile_points.append(Point(base_radius, 0)) # Start of involute
            self.tooth_profile_points.append(Point(pitch_radius * 1.01, 0.05 * pitch_radius)) # Mid point
            self.tooth_profile_points.append(Point(pitch_radius * 1.02, 0.1 * pitch_radius)) # Outer point

        logger.debug(f"Tooth profile for {self.name} (simplified): {self.tooth_profile_points}")

    def get_full_gear_points(self, num_points_per_tooth_flank=10) -> List[List[Point]]:
        """
        Generates points for all teeth of the gear.
        Returns a list of lists, where each inner list contains points for one tooth.
        (This is a placeholder)
        """
        if not self.tooth_profile_points:
            self.generate_tooth_profile()

        all_teeth_points: List[List[Point]] = []
        angle_increment = 2 * math.pi / self.params.num_teeth # Angle between teeth

        for i in range(self.params.num_teeth):
            tooth_angle_offset = i * angle_increment
            current_tooth_flank: List[Point] = []
            # Apply rotation to the reference tooth profile points
            for p_ref in self.tooth_profile_points:
                # Rotate p_ref by tooth_angle_offset around origin (0,0)
                x_rot = p_ref.x * math.cos(tooth_angle_offset) - p_ref.y * math.sin(tooth_angle_offset)
                y_rot = p_ref.x * math.sin(tooth_angle_offset) + p_ref.y * math.cos(tooth_angle_offset)
                current_tooth_flank.append(Point(x_rot, y_rot))
            # Here we would also mirror this flank to get the full tooth, add root fillet, top land etc.
            all_teeth_points.append(current_tooth_flank)
        logger.info(f"Points for {self.params.num_teeth} teeth generated (simplified).")
        return all_teeth_points

    def get_properties(self) -> Dict[str, Any]:
        """Returns key properties of the gear."""
        return {
            "name": self.name,
            "module": self.params.module,
            "diametral_pitch": self.params.diametral_pitch,
            "num_teeth": self.params.num_teeth,
            "pressure_angle_deg": self.params.pressure_angle_deg,
            "pitch_diameter": self.params.pitch_diameter,
            "base_diameter": self.params.base_diameter,
        }

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    try:
        params = SpurGearParameters(module=2, num_teeth=30, pressure_angle_deg=20)
        gear1 = SpurGear(params, name="ModuleGear")
        print(f"\nGear 1 ({gear1.name}) Properties:")
        for key, val in gear1.get_properties().items():
            print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
        gear1.generate_tooth_profile()
        all_points = gear1.get_full_gear_points()
        print(f"Generated {len(all_points)} (simplified) teeth point sets for {gear1.name}")
        if all_points:
             print(f"  Points for first tooth flank (simplified): {all_points[0]}")

        params_dp = SpurGearParameters(diametral_pitch=10, num_teeth=50)
        gear2 = SpurGear(params_dp, name="DPGear")
        print(f"\nGear 2 ({gear2.name}) Properties:")
        for key, val in gear2.get_properties().items():
            print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")

    except ValueError as e:
        logger.error(f"Error creating gear: {e}")
