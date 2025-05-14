import logging
from typing import List, Tuple, Any, Iterator as TypingIterator
from dataclasses import dataclass, field

# Assuming MotionStudy or a similar base class with get_members and set_value
# Needs actual import based on refactored structure
from .motionstudy import MotionStudy # Or appropriate base class

logger = logging.getLogger(__name__)

@dataclass
class IterableParameter:
    """Stores definition for one parameter to iterate over."""
    component_name: str
    parameter_name: str
    min_value: float
    max_value: float
    step: float
    num_steps: int = field(init=False)

    def __post_init__(self):
        if self.step == 0:
            # Avoid division by zero, treat as single step if min==max
            self.num_steps = 1 if self.min_value == self.max_value else 0
        else:
            # Calculate steps, handle float precision carefully
            # Add small epsilon to max_value to include endpoint if step divides evenly
            self.num_steps = int(((self.max_value - self.min_value) / self.step) + 1e-9) + 1
        if self.num_steps <= 0:
             logger.warning(f"IterableParameter {self.parameter_name} has non-positive steps ({self.num_steps}). Check min/max/step.")
             self.num_steps = 0 # Ensure it's non-negative

    def get_value(self, step_index: int) -> float:
        """Calculates the parameter value for a given step index."""
        if step_index < 0 or step_index >= self.num_steps:
             raise IndexError("Step index out of range")
        return self.min_value + step_index * self.step

class SystemIterator:
    """Iterates through parameter variations of a given motion system."""

    def __init__(self, system: MotionStudy):
        """Initializes the iterator with a motion system instance.

        Args:
            system: The motion system instance to iterate parameters on.
        """
        self.system = system
        self.iterables: List[IterableParameter] = []
        self._max_indices: List[int] = []
        self._current_indices: List[int] = []
        self._baked: bool = False

    def add_iterator(
        self,
        component_name: str,
        parameter_name: str,
        min_value: float,
        max_value: float,
        step: float
    ) -> None:
        """Adds a parameter to iterate over.

        Args:
            component_name: Name of the system component (e.g., 'drive1').
            parameter_name: Name of the parameter within the component (e.g., 'x').
            min_value: The starting value for the iteration.
            max_value: The ending value for the iteration.
            step: The increment value for each step.

        Raises:
            ValueError: If the component or parameter does not exist in the system.
            RuntimeError: If called after `bake()` has been called.
        """
        if self._baked:
             raise RuntimeError("Cannot add iterators after baking.")

        # Check if component exists
        if component_name not in self.system.get_members():
            raise ValueError(f"Component '{component_name}' does not exist in system!")

        # Basic check if parameter might exist (set_value will do final check)
        try:
            # Attempt to get the component to see if parameter might be settable
            # This isn't foolproof but catches basic typos
            component = getattr(self.system, component_name)
            # We don't have the value yet, so can't fully test set_value here
            # rely on MotionStudy's set_value for the final check during iteration.
        except AttributeError:
             # Should not happen if get_members is correct, but as a safeguard.
             raise ValueError(f"Component '{component_name}' not found via getattr, despite being in get_members().")

        iterable = IterableParameter(component_name, parameter_name, min_value, max_value, step)
        if iterable.num_steps > 0:
             self.iterables.append(iterable)
             logger.debug(f"Added iterator: {iterable}")
        else:
             logger.warning(f"Iterator for {parameter_name} added but has zero steps. Will be ignored.")

    def bake(self) -> None:
        """Prepares the iterator for execution. Must be called after adding iterators.

        Raises:
            RuntimeError: If no iterables were added.
        """
        if not self.iterables:
            # Allow baking with no iterables - __next__ will just raise StopIteration immediately
            logger.warning("Baking iterator with no iterable parameters defined.")
            self._baked = True
            return

        self._max_indices = [it.num_steps - 1 for it in self.iterables]
        # Start indices at the maximum to allow first __next__ call to decrement correctly
        self._current_indices = list(self._max_indices)
        self._baked = True
        # Add a special state to handle the very first call to next()
        self._first_next_call = True
        logger.info(f"Iterator baked. Max indices: {self._max_indices}")

    def __iter__(self) -> TypingIterator['SystemIterator']:
        """Returns the iterator object itself."""
        if not self._baked:
            logger.warning("Iterator accessed before baking. Automatically baking now.")
            self.bake()
        # Reset indices to start iteration from the beginning
        self._current_indices = list(self._max_indices)
        self._first_next_call = True # Reset for new iteration loop
        return self

    def __next__(self) -> 'SystemIterator':
        """Advances to the next parameter combination and updates the system.

        Returns:
            The iterator instance itself, allowing access to the updated system.

        Raises:
            StopIteration: When all parameter combinations have been exhausted.
            RuntimeError: If `bake()` was not called before iteration.
        """
        if not self._baked:
            raise RuntimeError("Iterator must be baked before iterating. Call bake().")
        if not self.iterables:
             raise StopIteration # No parameters to iterate

        # Special handling for the very first call after __iter__ or bake
        if self._first_next_call:
            self._first_next_call = False
            # Don't decrement yet, just apply the initial state (all max indices)
            # Or should we start from index 0? Let's assume start from 0,0,...0
            self._current_indices = [0] * len(self.iterables)
            self._apply_current_parameters()
            return self
        else:
             # Increment the pointer/indices (like an odometer)
            if not self._increment_indices():
                raise StopIteration

        self._apply_current_parameters()
        return self

    def _increment_indices(self) -> bool:
         """Increments the internal indices to the next state. Returns False if iteration is finished."""
         num_iterables = len(self.iterables)
         for i in range(num_iterables - 1, -1, -1):
             if self._current_indices[i] < self._max_indices[i]:
                 self._current_indices[i] += 1
                 # Reset subsequent indices to 0
                 for j in range(i + 1, num_iterables):
                     self._current_indices[j] = 0
                 return True  # Indices incremented successfully
         return False  # Reached the end of iteration


    def _apply_current_parameters(self) -> None:
        """Applies the parameter values corresponding to the current indices to the system."""
        logger.debug(f"Applying parameters for indices: {self._current_indices}")
        for i, iterable in enumerate(self.iterables):
            current_value = iterable.get_value(self._current_indices[i])
            try:
                self.system.set_value(iterable.component_name, iterable.parameter_name, current_value)
                logger.debug(f"  Set {iterable.component_name}.{iterable.parameter_name} = {current_value}")
            except Exception as e:
                # Catch potential errors from set_value (e.g., ValueError)
                logger.error(
                    f"Error setting {iterable.component_name}.{iterable.parameter_name} to {current_value}: {e}",
                    exc_info=True
                )
                # Optionally re-raise or handle differently
                raise

    def print_iterables(self) -> None:
        """Prints the configured iterables for debugging."""
        print("Configured Iterables:")
        if not self.iterables:
            print("  None")
            return
        for i, iterable in enumerate(self.iterables):
            print(f"  {i}: {iterable}")
        if self._baked:
            print(f"Max Indices: {self._max_indices}")
        else:
            print("(Iterator not baked yet)")

# Example Usage (requires a MotionStudy implementation)
# if __name__ == '__main__':
#     # Assuming a MockMotionStudy or a real one like TwoBar exists
#     class MockMotionStudy(MotionStudy):
#         def __init__(self):
#             self.drive1 = type('obj', (object,), {'x': 0, 'y': 0})()
#             self.drive2 = type('obj', (object,), {'r': 5})()
#         def get_members(self): return ['drive1', 'drive2']
#         def set_value(self, member, param, value):
#              setattr(getattr(self, member), param, value)
#              print(f"Set {member}.{param} = {value}")
#
#     mock_system = MockMotionStudy()
#     iterator = SystemIterator(mock_system)
#
#     iterator.add_iterator('drive1', 'x', -10, 10, 5) # 5 steps: -10, -5, 0, 5, 10
#     iterator.add_iterator('drive2', 'r', 1, 3, 1)   # 3 steps: 1, 2, 3
#
#     iterator.print_iterables()
#     iterator.bake()
#     iterator.print_iterables()
#
#     print("\nStarting iteration...")
#     count = 0
#     try:
#         for current_state in iterator:
#              count += 1
#              print(f"Iteration {count}: drive1.x={current_state.system.drive1.x}, drive2.r={current_state.system.drive2.r}")
#     except Exception as e:
#          print(f"Error during iteration: {e}")
#
#     print(f"\nTotal iterations: {count}") # Should be 5 * 3 = 15