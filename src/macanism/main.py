"""
Main executable for the Macanism project.
"""
import logging
from macanism.macanism_engine import MacanismEngine # Updated import

# Configure basic logging for the main application
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_simulation() -> None:
    """
    Runs a simple simulation using the MacanismEngine.
    """
    logger.info("Initializing Macanism simulation...")
    try:
        engine = MacanismEngine(name="PrimaryMacanism")
        logger.debug(f"Engine created: {engine}")

        engine.start()
        logger.info(f"Engine '{engine.name}' status: {engine.status}")

        engine_info = engine.get_info()
        logger.info(f"Engine info: {engine_info}")

        # Simulate some work
        logger.info("Simulating work with the engine...")
        # In a real scenario, you would call engine methods that do work.

        engine.stop()
        logger.info(f"Engine '{engine.name}' status after stopping: {engine.status}")
        logger.info("Macanism simulation finished successfully.")

    except Exception as e:
        logger.error(f"An error occurred during the simulation: {e}", exc_info=True)

if __name__ == "__main__":
    logger.info("Starting Macanism application.")
    run_simulation()
    logger.info("Macanism application finished.")