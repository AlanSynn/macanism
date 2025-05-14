from dataclasses import dataclass, field
from typing import Literal

@dataclass
class Bar:
    """Represents a bar (link) in a linkage system.

    Attributes:
        length: The length of the bar.
        joint: The position along the length where another bar connects, often used
               in multi-bar systems like TwoBar where one bar drives another.
               Defaults to 0, implying connection at the start.
    """
    length: float
    joint: float = 0.0

    def set_value(self, parameter: Literal['length', 'joint'], value: float) -> None:
        """Sets the value of a specified parameter.

        Args:
            parameter: The name of the parameter to set ('length' or 'joint').
            value: The new value for the parameter.

        Raises:
            ValueError: If the parameter name is invalid.
        """
        if parameter == 'length':
            self.length = value
        elif parameter == 'joint':
            self.joint = value
        else:
            # Use f-string for formatted error message
            raise ValueError(f"Parameter '{parameter}' does not exist for Bar!")