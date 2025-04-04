"""
Unit tests for the MacanismEngine class.
"""
import unittest
import logging
from macanism.macanism_engine import MacanismEngine # Updated import

# Suppress logging during most tests unless specifically needed for a test
logging.disable(logging.CRITICAL)

class TestMacanismEngine(unittest.TestCase):
    """
    Test suite for the MacanismEngine.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Disable logging for the entire test class once."""
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls) -> None:
        """Restore logging configuration after all tests in the class have run."""
        logging.disable(logging.NOTSET)

    def setUp(self) -> None:
        """Set up for each test method."""
        self.engine_name = "TestEngine"
        self.engine = MacanismEngine(name=self.engine_name)

    def test_creation(self) -> None:
        """Test basic engine creation and initial properties."""
        self.assertIsNotNone(self.engine)
        self.assertEqual(self.engine.name, self.engine_name)
        self.assertEqual(self.engine.status, "initialized")
        self.assertIn(self.engine_name, str(self.engine))
        self.assertIn("initialized", str(self.engine))

    def test_name_property_valid(self) -> None:
        """Test the name property getter and setter with valid input."""
        new_name = "UpdatedTestEngine"
        self.engine.name = new_name
        self.assertEqual(self.engine.name, new_name)

    def test_name_property_invalid_empty(self) -> None:
        """Test setting an empty name, expecting ValueError."""
        with self.assertRaises(ValueError):
            self.engine.name = ""

    def test_name_property_invalid_type(self) -> None:
        """Test setting a name with an invalid type, expecting ValueError or TypeError."""
        with self.assertRaises(ValueError): # Current implementation raises ValueError due to `not isinstance(value, str)`
            self.engine.name = 123 # type: ignore

    def test_start_engine_from_initialized(self) -> None:
        """Test the start method when engine is initialized."""
        self.engine.start()
        self.assertEqual(self.engine.status, "running")

    def test_start_engine_when_already_running(self) -> None:
        """Test starting an engine that is already running (should remain running and log warning)."""
        self.engine.start()
        self.assertEqual(self.engine.status, "running")
        # Capture warnings or check log if testing log output is critical
        self.engine.start() # Call again
        self.assertEqual(self.engine.status, "running")

    def test_stop_engine_from_running(self) -> None:
        """Test the stop method when engine is running."""
        self.engine.start() # Must be running to stop
        self.engine.stop()
        self.assertEqual(self.engine.status, "stopped")

    def test_stop_engine_when_already_stopped(self) -> None:
        """Test stopping an engine that is already stopped (should remain stopped and log warning)."""
        self.engine.start()
        self.engine.stop()
        self.assertEqual(self.engine.status, "stopped")
        self.engine.stop() # Call again
        self.assertEqual(self.engine.status, "stopped")

    def test_stop_engine_when_initialized_not_started(self) -> None:
        """Test stopping an engine that was initialized but never started."""
        self.assertEqual(self.engine.status, "initialized")
        self.engine.stop() # Should log info/warning, status remains "initialized"
        self.assertEqual(self.engine.status, "initialized")

    def test_get_info(self) -> None:
        """Test the get_info method."""
        info = self.engine.get_info()
        self.assertIsInstance(info, dict)
        self.assertEqual(info["name"], self.engine_name)
        self.assertEqual(info["status"], "initialized")
        self.assertIn("version", info)
        self.assertEqual(info["version"], "0.1.0")

    def test_string_representation(self) -> None:
        """Test the __str__ method for correct format."""
        expected_str = f"<MacanismEngine name='{self.engine_name}' status='initialized'>"
        self.assertEqual(str(self.engine), expected_str)
        self.engine.start()
        expected_str_running = f"<MacanismEngine name='{self.engine_name}' status='running'>"
        self.assertEqual(str(self.engine), expected_str_running)

if __name__ == '__main__':
    unittest.main()