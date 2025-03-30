"""
Module for the Macanism Engine.
"""
import logging
from typing import Any

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MacanismEngine:
    """
    A class representing the core Macanism Engine.

    This engine is responsible for demonstrating macanism principles.
    """

    def __init__(self, name: str = "Macanism Engine") -> None:
        """
        Initializes the MacanismEngine.

        Args:
            name (str): The name of this engine instance.
        """
        self._name: str = name
        self._status: str = "initialized"
        logging.info(f"MacanismEngine '{self._name}' created.")

    @property
    def name(self) -> str:
        """str: Gets the name of the engine."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Sets the name of the engine."""
        if not isinstance(value, str) or not value:
            raise ValueError("Name must be a non-empty string.")
        logging.info(f"MacanismEngine name changed from '{self._name}' to '{value}'.")
        self._name = value

    @property
    def status(self) -> str:
        """str: Gets the current status of the engine."""
        return self._status

    def start(self) -> None:
        """Starts the engine's main process."""
        if self._status == "running":
            logging.warning(f"Engine '{self.name}' is already running.")
            return
        self._status = "running"
        logging.info(f"MacanismEngine '{self.name}' started.")
        # Placeholder for actual start logic

    def stop(self) -> None:
        """Stops the engine's main process."""
        if self._status != "running":
            # It could be 'initialized' or 'stopped' already
            if self._status == "initialized":
                logging.info(f"Engine '{self.name}' was initialized but not started. No action taken to stop.")
            elif self._status == "stopped":
                logging.warning(f"Engine '{self.name}' is already stopped.")
            else:
                 logging.warning(f"Engine '{self.name}' is not in a running state (current status: {self.status}).")
            return
        self._status = "stopped"
        logging.info(f"MacanismEngine '{self.name}' stopped.")
        # Placeholder for actual stop logic

    def get_info(self) -> dict[str, Any]:
        """
        Retrieves information about the engine.

        Returns:
            dict[str, Any]: A dictionary containing engine information.
        """
        return {
            "name": self.name,
            "status": self.status,
            "version": "0.1.0" # Example version, could come from __init__
        }

    def __str__(self) -> str:
        """
        Returns a string representation of the MacanismEngine.

        Returns:
            str: String representation of the engine.
        """
        return f"<MacanismEngine name='{self.name}' status='{self.status}'>"

if __name__ == '__main__':
    # Example Usage
    engine = MacanismEngine(name="MyMacanismInstance")
    print(engine)
    print(f"Engine Name: {engine.name}")
    print(f"Engine Status: {engine.status}")

    engine.start()
    print(f"Engine Status after start: {engine.status}")
    info = engine.get_info()
    print(f"Engine Info: {info}")

    engine.stop()
    print(f"Engine Status after stop: {engine.status}")

    # Test stopping an already stopped engine
    engine.stop()

    # Test starting an already started engine
    engine.start() # Should warn it's already running
    engine.start() # Call start again

    # Test stopping an engine that was initialized but not started
    engine2 = MacanismEngine(name="UnstartedEngine")
    engine2.stop()

    try:
        engine.name = ""
    except ValueError as e:
        logging.error(f"Error setting name: {e}")