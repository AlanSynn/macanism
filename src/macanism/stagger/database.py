import json
import logging
import os
import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# Assuming TwoBar provides a .parameters tuple
from .twobar import TwoBar

logger = logging.getLogger(__name__)

class Database:
    """Handles database operations using a JSON file as a simple key-value store."""

    _DEFAULT_STRUCTURE = {
        "studies": [],
        "parameters_2bar": [],
        "points": [],
        "_counters": {"study_id": 0, "param_id": 0} # Internal counters for IDs
    }

    def __init__(self, filename: str):
        """Initializes the database by loading from the JSON file.

        Args:
            filename: The path to the JSON database file.
        """
        self.db_path = Path(filename)
        self.data: Dict[str, Any] = {}
        self._load_db_from_json()

    def _get_next_id(self, counter_key: str) -> int:
        """Gets the next available ID for a given type and increments the counter."""
        if "_counters" not in self.data:
            self.data["_counters"] = self._DEFAULT_STRUCTURE["_counters"].copy()

        current_id = self.data["_counters"].get(counter_key, 0)
        next_id = current_id + 1
        self.data["_counters"][counter_key] = next_id
        return next_id

    def _load_db_from_json(self) -> None:
        """Loads the database state from the JSON file."""
        try:
            if self.db_path.exists():
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                # Ensure essential keys exist after loading
                for key, default_value in self._DEFAULT_STRUCTURE.items():
                     if key not in self.data:
                         self.data[key] = default_value
                logger.info(f"Loaded database state from: {self.db_path}")
            else:
                logger.info(f"Database file not found: {self.db_path}. Initializing new state.")
                self.data = self._DEFAULT_STRUCTURE.copy() # Use a copy
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading database file {self.db_path}. Initializing new state. Error: {e}", exc_info=True)
            self.data = self._DEFAULT_STRUCTURE.copy() # Use a copy on error

    def save(self) -> None:
        """Saves the current database state to the JSON file."""
        try:
            # Ensure parent directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=4)
            logger.info(f"Saved database state to: {self.db_path}")
        except IOError as e:
            logger.error(f"Error saving database file {self.db_path}: {e}", exc_info=True)

    def create_default_tables(self) -> None:
        """Ensures the basic structure exists in the loaded data. (No-op for file creation)."""
        for key, default_value in self._DEFAULT_STRUCTURE.items():
            if key not in self.data:
                self.data[key] = default_value
        logger.debug("Default data structure verified/initialized in memory.")

    def insert_study(self, name: str) -> Optional[int]:
        """Inserts a new study record.

        Args:
            name: The name of the study.

        Returns:
            The ID of the newly inserted study, or None if insertion failed.
        """
        if 'studies' not in self.data:
             self.data['studies'] = []

        new_id = self._get_next_id("study_id")
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        new_study = {"id": new_id, "name": name, "created_at": timestamp}
        self.data['studies'].append(new_study)
        logger.info(f"Inserted study '{name}' with ID {new_id}")
        return new_id

    def insert_parameters(self, study_id: int, name: str, study_params: TwoBar) -> Optional[int]:
        """Inserts a set of parameters for a given study.

        Args:
            study_id: The ID of the parent study.
            name: A descriptive name for this parameter set.
            study_params: The TwoBar system object containing the parameters.

        Returns:
            The ID of the newly inserted parameter set, or None if insertion failed.
        """
        if 'parameters_2bar' not in self.data:
             self.data['parameters_2bar'] = []

        new_id = self._get_next_id("param_id")
        params_tuple = study_params.parameters
        params_dict = {
            "id": new_id,
            "study_id": study_id,
            "name": name,
            "bar1_length": params_tuple[0],
            "bar1_joint": params_tuple[1],
            "bar2_length": params_tuple[2],
            "anchor1_x": params_tuple[3],
            "anchor1_y": params_tuple[4],
            "anchor1_r": params_tuple[5],
            "anchor1_speed": params_tuple[6],
            "anchor1_initial": params_tuple[7],
            "anchor2_x": params_tuple[8],
            "anchor2_y": params_tuple[9],
            "anchor2_r": params_tuple[10],
            "anchor2_speed": params_tuple[11],
            "anchor2_initial": params_tuple[12],
        }
        self.data['parameters_2bar'].append(params_dict)
        logger.info(f"Inserted parameter set '{name}' for study ID {study_id} with ID {new_id}")
        return new_id

    def insert_endpoints(
        self,
        parameter_set_id: int,
        path_data: List[Tuple[float, float]],
        input_angles: Optional[List[float]] = None
    ) -> None:
        """Inserts a list of endpoint coordinates for a given parameter set.

        Args:
            parameter_set_id: The ID of the parameter set these points belong to.
            path_data: A list of (x, y) tuples representing the path points.
            input_angles: Optional list of input angles corresponding to each point.
        """
        if 'points' not in self.data:
             self.data['points'] = []

        points_to_insert = []
        for i, point in enumerate(path_data):
            angle = input_angles[i] if input_angles and i < len(input_angles) else None
            points_to_insert.append({
                "parameter_set_id": parameter_set_id,
                "x": point[0],
                "y": point[1],
                "input_angle": angle
            })

        self.data['points'].extend(points_to_insert)
        logger.info(f"Inserted {len(points_to_insert)} endpoints for parameter set ID {parameter_set_id}.")