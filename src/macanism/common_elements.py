"""
Module for common geometric elements used in macanism simulations.
"""
from dataclasses import dataclass
import math

@dataclass
class Point:
    """Represents a point in 2D space."""
    x: float = 0.0
    y: float = 0.0

    def distance_to(self, other: 'Point') -> float:
        """Calculates the distance to another point."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

@dataclass
class Vector:
    """Represents a 2D vector."""
    x: float = 0.0
    y: float = 0.0

    @property
    def magnitude(self) -> float:
        """Calculates the magnitude (length) of the vector."""
        return math.sqrt(self.x**2 + self.y**2)

    @property
    def angle(self) -> float:
        """Calculates the angle (in radians) of the vector from the positive x-axis."""
        return math.atan2(self.y, self.x)

    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Vector': # Scalar multiplication
        return Vector(self.x * scalar, self.y * scalar)

    @classmethod
    def from_polar(cls, magnitude: float, angle: float) -> 'Vector':
        """Creates a Vector from polar coordinates (magnitude and angle in radians)."""
        return cls(magnitude * math.cos(angle), magnitude * math.sin(angle))

if __name__ == '__main__':
    p1 = Point(1, 2)
    p2 = Point(4, 6)
    print(f"Point 1: {p1}")
    print(f"Point 2: {p2}")
    print(f"Distance between p1 and p2: {p1.distance_to(p2):.2f}")

    v1 = Vector(3, 4)
    print(f"Vector 1: {v1}")
    print(f"Vector 1 magnitude: {v1.magnitude:.2f}")
    print(f"Vector 1 angle: {math.degrees(v1.angle):.2f} degrees")

    v_polar = Vector.from_polar(5, math.radians(30))
    print(f"Vector from polar (5, 30deg): {v_polar}")
    print(f"Magnitude: {v_polar.magnitude:.2f}, Angle: {math.degrees(v_polar.angle):.2f} deg")

    v2 = Vector(1, 1)
    v_sum = v1 + v2
    print(f"v1 + v2 = {v_sum}")
    v_scaled = v1 * 2
    print(f"v1 * 2 = {v_scaled}")