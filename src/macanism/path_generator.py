#!filepath src/macanism/path_generator.py
import logging
import os
from pathlib import Path
from typing import List, Tuple, Any, Optional
import numpy as np

from PIL import Image, ImageDraw, ImageOps
# Import spline functions from scipy
from scipy import interpolate

# Import matplotlib for animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter # PillowWriter for GIF

# Assuming stagger is now part of the macanism package
from .stagger import Anchor, Bar, Database, SystemIterator, TwoBar

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PathGenerator:
    """
    Generates motion paths for a two-bar linkage system, saves them to a
    database (JSON file), and exports them as PNG images.
    """

    def __init__(self, output_dir: str = "outputs"):
        """
        Initializes the PathGenerator.

        Args:
            output_dir: The directory to save database (JSON) and image files.
        """
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.output_path / "motion_studies.json"

        # Instantiate Database directly
        self.db: Optional[Database] = None
        try:
            self.db = Database(str(self.db_path)) # Loads existing or initializes new
            self.db.create_default_tables() # Ensures structure
        except Exception as e:
            logging.error(f"Failed to initialize database {self.db_path}: {e}", exc_info=True)
            self.db = None # Proceed without DB if init fails

        self.motion_system: TwoBar | None = None
        self.iterable_system: SystemIterator | None = None

        self._create_system()
        if self.motion_system:
            self._run_iterations()
        else:
            logging.error("System creation failed. Cannot run iterations.")

        # Explicitly save the database at the end of operations if it exists
        if self.db:
            try:
                self.db.save()
            except Exception as e:
                logging.error(f"Failed to save database {self.db_path}: {e}", exc_info=True)

    def _create_system(self) -> None:
        """Creates the default two-bar linkage system."""
        logging.info("Creating two-bar linkage system.")
        try:
            bar1 = Bar(length=35, joint=30) # Assuming joint relates to connection point on bar1
            bar2 = Bar(length=40)

            # Original parameters: Anchor(-20, -20, 10, 6, 0), Anchor(15, -22, 6, 3, 180)
            drive1 = Anchor(x=-20, y=-20, r=10, speed=6.0, initial=0.0)
            drive2 = Anchor(x=15, y=-22, r=6, speed=3.0, initial=180.0)

            self.motion_system = TwoBar(drive1, drive2, bar1, bar2)
            logging.info("System created successfully.")

            # Setup iteration parameters
            self.iterable_system = SystemIterator(self.motion_system)
            # Example iteration: Vary drive1's x and y coordinates
            # self.iterable_system.add_iterator(
            #     component_name='drive1',
            #     parameter_name='x',
            #     min_value=-25.0,
            #     max_value=15.0,
            #     step=2.0
            # )
            # self.iterable_system.add_iterator(
            #     component_name='drive1',
            #     parameter_name='y',
            #     min_value=-25.0,
            #     max_value=15.0,
            #     step=2.0
            # )
            # self.iterable_system.bake() # Prepares the iterator
            # self.iterable_system.print_iterables() # Logs the iteration setup

        except ValueError as e:
            logging.error(f"Failed to create system due to invalid parameters: {e}", exc_info=True)
            self.motion_system = None # Ensure system is None if creation fails
        except Exception as e:
            logging.error(f"An unexpected error occurred during system creation: {e}", exc_info=True)
            self.motion_system = None

    def _run_iterations(self) -> None:
        """Runs the path generation for each parameter set defined in the iterator."""
        assert self.motion_system is not None
        assert self.iterable_system is not None

        try:
             self.iterable_system.bake()
        except RuntimeError as e:
             logging.error(f"Failed to bake iterator: {e}", exc_info=True)
             return

        if not self.iterable_system.iterables:
             logging.info("No iterations defined. Generating path for the base system.")
             input_angles = [step * self.motion_system.step_size for step in range(int(self.motion_system.resolution))]
             self._generate_and_save_path(self.motion_system, "base_system", input_angles)
             return

        logging.info("Starting path generation iterations.")
        i = 0
        try:
            for iteration_state in self.iterable_system:
                 i += 1
                 system_label = f"iteration_{i}"
                 logging.info(f"Running {system_label}")
                 current_system = iteration_state.system
                 input_angles = [step * current_system.stepSize for step in range(int(current_system.resolution))]
                 self._generate_and_save_path(current_system, system_label, input_angles)
            logging.info(f"Completed {i} iterations.")
        except Exception as e:
             logging.error(f"Error during iteration run: {e}", exc_info=True)


    def _generate_and_save_path(self, system: TwoBar, label: str, input_angles: List[float]) -> None:
         """Generates, saves (DB and PNG) the path for a given system configuration."""
         assert system is not None

         logging.debug(f"Generating path for {label} with {len(input_angles)} steps.")
         try:
             # Generate the full path data using all input angles for PNG/DB
             raw_path_data = [system.end_path(angle) for angle in input_angles]
             path_data = [(x,y) for x,y in raw_path_data if not (np.isnan(x) or np.isnan(y))]

             if not path_data:
                 logging.warning(f"No valid path data generated for {label} after filtering NaNs. Skipping save.")
                 return

             # Save to database if DB is available (using full path data)
             if self.db:
                  self._save_to_database(system, path_data, label, input_angles)

             # Save as PNG (using full path data, smoothed)
             png_filename = self.output_path / f"{label}.png"
             self._save_path_as_png(str(png_filename), path_data)

             # --- Animation Generation (Limited Duration) ---
             # Only animate specific labels (e.g., base system)
             if label == "base_system" or label == "iteration_1":
                 gif_filename = self.output_path / f"{label}_animation.gif"

                 # Determine angles for one 360-degree cycle for animation
                 step_size = system.step_size
                 # Ensure step_size is positive to avoid infinite loop or zero division
                 if step_size <= 0:
                      logging.warning(f"System step_size ({step_size}) is not positive. Cannot determine animation cycle. Skipping animation.")
                 else:
                      num_animation_frames = int(round(360.0 / step_size))
                      if num_animation_frames <= 0:
                          logging.warning(f"Calculated zero or negative animation frames ({num_animation_frames}) based on step_size {step_size}. Skipping animation.")
                      else:
                          # Use only the angles corresponding to the first 360-degree cycle
                          animation_input_angles = [i * step_size for i in range(num_animation_frames)]
                          # We pass the full path_data for potential background plotting later,
                          # but the animation itself runs using animation_input_angles.
                          self._save_path_as_animation(system, animation_input_angles, str(gif_filename), path_data)
                 # --- End Animation Generation ---

         except (ValueError, ZeroDivisionError) as e:
             logging.error(f"Could not calculate or save path for {label}: {e}", exc_info=True)
         except Exception as e:
             logging.error(f"An unexpected error occurred for {label}: {e}", exc_info=True)


    def _save_to_database(
        self,
        system: TwoBar,
        path_data: List[Tuple[float, float]],
        label: str,
        input_angles: List[float]
    ) -> None:
        """Saves the system parameters and generated path points to the database."""
        if not self.db:
            logging.warning("Database not available. Skipping database save.")
            return

        try:
            # Assuming study_name should be descriptive
            study_name = f"Motion Study - {label}"
            motion_study_id = self.db.insert_study(study_name)

            if motion_study_id:
                # Assuming parameter_set_name can be generic or derived
                parameter_set_name = f"Parameters - {label}"
                parameter_set_id = self.db.insert_parameters(motion_study_id, parameter_set_name, system)

                if parameter_set_id:
                    self.db.insert_endpoints(parameter_set_id, path_data, input_angles)
                    logging.info(f"Saved system parameters (ID: {parameter_set_id}) and path for '{label}' to database (Study ID: {motion_study_id}).")
                else:
                    logging.error(f"Failed to insert parameter set '{parameter_set_name}' into database.")
            else:
                 logging.error(f"Failed to insert study '{study_name}' into database.")

        except Exception as e:
            logging.error(f"Failed to save data to database for {label}: {e}", exc_info=True)


    def _save_path_as_png(
        self,
        filename: str,
        path_data: List[Tuple[float, float]],
        scaling: float = 10.0,
        bg_color: int = 255,
        line_color: int = 0,
        line_width: int = 2,
        smooth_factor: int = 5 # Factor to increase points for smoothing
    ) -> None:
        """
        Saves the generated path as a PNG image, optionally smoothing with splines.

        Args:
            filename: The path to save the PNG file.
            path_data: A list of (x, y) tuples representing the calculated path points.
            scaling: Factor to scale the path coordinates for the image.
            bg_color: Background color (0-255).
            line_color: Line color (0-255).
            line_width: Width of the path line in pixels.
            smooth_factor: Multiplier for the number of points used for spline smoothing.
                           Set to 1 or less to disable smoothing.
        """
        if not path_data or len(path_data) < 2:
            logging.warning(f"Not enough path data ({len(path_data)} points) provided for {filename}. Skipping PNG save.")
            return

        logging.info(f"Saving path to PNG: {filename}")

        points_to_draw = path_data
        num_original_points = len(path_data)

        # --- Spline Smoothing Logic ---
        if smooth_factor > 1 and num_original_points >= 4: # Need enough points for splprep
            logging.debug(f"Applying spline smoothing with factor {smooth_factor}")
            try:
                # Ensure path_data is closed loop for splprep if needed (connect first and last)
                # If the path naturally closes, this might not be necessary or could be handled
                # by checking if path_data[0] is close to path_data[-1]
                closed_path = path_data + [path_data[0]] # Assume we want a closed loop
                x = [p[0] for p in closed_path]
                y = [p[1] for p in closed_path]

                # Find the B-spline representation, k=3 for cubic spline
                # s=0 forces interpolation through all points (can be adjusted for smoothing)
                tck, u = interpolate.splprep([x, y], s=0, per=True)

                # Evaluate the spline at a denser set of points
                num_smooth_points = num_original_points * smooth_factor
                u_smooth = np.linspace(u.min(), u.max(), num_smooth_points)
                x_smooth, y_smooth = interpolate.splev(u_smooth, tck)

                points_to_draw = list(zip(x_smooth, y_smooth))
                logging.debug(f"Smoothed path from {num_original_points} to {len(points_to_draw)} points.")

            except Exception as e:
                logging.warning(f"Spline smoothing failed for {filename}: {e}. Drawing original points.", exc_info=True)
                points_to_draw = path_data # Fallback to original points
        else:
            if smooth_factor > 1:
                 logging.warning(f"Not enough points ({num_original_points}) for spline smoothing. Drawing original points.")
        # --- End Spline Smoothing Logic ---

        try:
            scaled_data, bounding_box = self._reposition_and_scale(points_to_draw, scaling)

            if not bounding_box or bounding_box[0] <= 0 or bounding_box[1] <= 0:
                 logging.warning(f"Invalid bounding box {bounding_box} after scaling for {filename}. Skipping PNG.")
                 return

            img_width = max(1, int(bounding_box[0]))
            img_height = max(1, int(bounding_box[1]))

            im = Image.new('L', (img_width, img_height), bg_color)
            draw = ImageDraw.Draw(im)

            if len(scaled_data) > 1:
                 int_scaled_data = [(int(p[0]), int(p[1])) for p in scaled_data]
                 # Use the possibly smoothed points for drawing
                 draw.line(int_scaled_data, fill=line_color, width=line_width, joint="curve")

            del draw
            im = ImageOps.flip(im)
            im.save(filename, "PNG")
            logging.info(f"Successfully saved PNG: {filename}")

        except Exception as e:
            logging.error(f"Failed to save PNG {filename}: {e}", exc_info=True)

    def _reposition_and_scale(
        self,
        data: List[Tuple[float, float]],
        scaling: float = 1.0
    ) -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
        """
        Repositions path data so the minimum x, y are at (0, 0) and scales it.

        Args:
            data: List of (x, y) path points.
            scaling: Scaling factor.

        Returns:
            A tuple containing:
            - The repositioned and scaled path data.
            - The dimensions (width, height) of the scaled bounding box.
        """
        if not data:
            return [], (0.0, 0.0)

        # Find min/max coordinates
        x_coords, y_coords = zip(*data)
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Calculate dimensions
        width = x_max - x_min
        height = y_max - y_min

        # Reposition and scale
        repo_data = [
            ((x - x_min) * scaling, (y - y_min) * scaling)
            for x, y in data
        ]

        scaled_width = width * scaling
        scaled_height = height * scaling

        return repo_data, (scaled_width, scaled_height)

    def _save_path_as_animation(
        self,
        system: TwoBar,
        input_angles: List[float],
        filename: str,
        path_data: List[Tuple[float, float]], # The end-effector path
        fps: int = 30
    ) -> None:
        """
        Saves an animation of the linkage movement as a GIF.

        Args:
            system: The TwoBar system instance.
            input_angles: List of input angles for each frame.
            filename: The path to save the GIF file.
            path_data: Pre-calculated end-effector path points.
            fps: Frames per second for the animation.
        """
        logging.info(f"Generating animation: {filename}")

        if not path_data:
            logging.warning(f"No path data for animation {filename}. Skipping.")
            return

        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8) # Adjust figure size as needed

        # Determine plot limits based on the full linkage movement and path
        all_x = []
        all_y = []
        valid_input_angles = []

        for angle in input_angles:
            positions = system.get_linkage_positions(angle)
            if not any(np.isnan(coord) for pos in positions.values() for coord in pos):
                valid_input_angles.append(angle)
                for point_name, (px, py) in positions.items():
                    all_x.append(px)
                    all_y.append(py)

        if not all_x or not all_y:
            logging.warning(f"No valid linkage positions to determine animation bounds for {filename}. Skipping.")
            plt.close(fig)
            return

        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)

        # Add some padding to the limits
        padding = max((x_max - x_min) * 0.1, (y_max - y_min) * 0.1, 5) # Min padding of 5 units
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"Linkage Animation: {Path(filename).stem}")
        ax.grid(True)

        # Artists for the linkage components
        anchor1_dot, = ax.plot([], [], 'ko', ms=5, label='Anchor 1') # Drive 1 center
        anchor2_dot, = ax.plot([], [], 'ko', ms=5, label='Anchor 2') # Drive 2 center

        driven1_arm_line, = ax.plot([], [], 'r-', lw=2, label='Drive Arm 1') # From anchor1 to driven1
        driven2_arm_line, = ax.plot([], [], 'b-', lw=2, label='Drive Arm 2') # From anchor2 to driven2

        bar1_line, = ax.plot([], [], 'g-', lw=3, label='Bar 1')      # From driven1 to joint
        bar2_line, = ax.plot([], [], 'm-', lw=3, label='Bar 2')      # From driven2 to joint

        # End effector related lines
        # Line from driven1 to joint (along which end effector might be)
        bar1_extension_line, = ax.plot([], [], 'g--', lw=1, alpha=0.7)
        end_effector_dot, = ax.plot([], [], 'yo', ms=6, label='End Effector')

        # Path trace
        path_trace_line, = ax.plot([], [], 'k--', lw=1, alpha=0.5, label='Path Trace')

        # Static anchor positions (drawn once)
        anchor1_dot.set_data([system.drive1.x], [system.drive1.y])
        anchor2_dot.set_data([system.drive2.x], [system.drive2.y])

        # Store path points for tracing
        traced_x, traced_y = [], []

        # Filter input_angles to only those that produce valid linkage positions
        # to match the number of frames.
        num_frames = len(valid_input_angles)
        if num_frames == 0:
            logging.warning(f"No valid frames for animation {filename}. Skipping.")
            plt.close(fig)
            return

        def init():
            driven1_arm_line.set_data([], [])
            driven2_arm_line.set_data([], [])
            bar1_line.set_data([], [])
            bar2_line.set_data([], [])
            bar1_extension_line.set_data([],[])
            end_effector_dot.set_data([], [])
            path_trace_line.set_data([], [])
            # Keep anchors static
            return (driven1_arm_line, driven2_arm_line, bar1_line, bar2_line,
                    bar1_extension_line, end_effector_dot, path_trace_line, anchor1_dot, anchor2_dot)

        def update(frame_idx):
            angle = valid_input_angles[frame_idx]
            positions = system.get_linkage_positions(angle)

            a1_pos = positions['anchor1'] # Should be static, but for consistency
            a2_pos = positions['anchor2']
            d1_pos = positions['driven1']
            d2_pos = positions['driven2']
            j_pos = positions['joint']
            ee_pos = positions['end_effector']

            # Update drive arms if radius > 0
            if system.drive1.r > 0:
                driven1_arm_line.set_data([a1_pos[0], d1_pos[0]], [a1_pos[1], d1_pos[1]])
            else:
                driven1_arm_line.set_data([], []) # No arm if radius is zero

            if system.drive2.r > 0:
                driven2_arm_line.set_data([a2_pos[0], d2_pos[0]], [a2_pos[1], d2_pos[1]])
            else:
                driven2_arm_line.set_data([], [])

            # Update bars
            bar1_line.set_data([d1_pos[0], j_pos[0]], [d1_pos[1], j_pos[1]])
            bar2_line.set_data([d2_pos[0], j_pos[0]], [d2_pos[1], j_pos[1]])

            # Update end effector and its guiding line (bar1 extension)
            # The end effector is on the line from d1_pos through j_pos, scaled by bar1.joint / bar1.length
            bar1_extension_line.set_data([d1_pos[0], j_pos[0]], [d1_pos[1], j_pos[1]]) # Show the full bar1
            end_effector_dot.set_data([ee_pos[0]], [ee_pos[1]])

            # Update path trace
            if not (np.isnan(ee_pos[0]) or np.isnan(ee_pos[1])):
                traced_x.append(ee_pos[0])
                traced_y.append(ee_pos[1])
                path_trace_line.set_data(traced_x, traced_y)

            ax.set_title(f"Linkage Animation (Frame {frame_idx+1}/{num_frames}) Angle: {angle:.1f}°")

            return (driven1_arm_line, driven2_arm_line, bar1_line, bar2_line,
                    bar1_extension_line, end_effector_dot, path_trace_line, anchor1_dot, anchor2_dot)

        ani = FuncAnimation(fig, update, frames=num_frames,
                            init_func=init, blit=True, interval=1000/fps)

        ax.legend(loc='upper right', fontsize='small')

        try:
            writer = PillowWriter(fps=fps)
            ani.save(filename, writer=writer)
            logging.info(f"Successfully saved animation: {filename}")
        except Exception as e:
            logging.error(f"Failed to save animation {filename}: {e}", exc_info=True)
        finally:
            plt.close(fig) # Close the figure to free memory


if __name__ == '__main__':
    logging.info("Starting PathGenerator example.")
    # Example: Create a generator that saves to 'generated_paths' directory
    generator = PathGenerator(output_dir="generated_paths")
    # The generator automatically creates the system and runs iterations (if any)
    # upon initialization.
    logging.info("PathGenerator example finished.")

    # Optional: Example of creating a specific system and generating its path manually
    # logging.info("\nRunning manual example:")
    # try:
    #     manual_bar1 = Bar(length=50, joint=25)
    #     manual_bar2 = Bar(length=60)
    #     manual_drive1 = Anchor(x=0, y=0, r=15, speed=2.0)
    #     manual_drive2 = Anchor(x=40, y=5, r=10, speed=-1.0)
    #     manual_system = TwoBar(manual_drive1, manual_drive2, manual_bar1, manual_bar2)
    #
    #     # Create a separate generator instance for the manual run if desired
    #     manual_generator = PathGenerator(output_dir="manual_path_output")
    #     # Manually set the system for this instance
    #     manual_generator.motion_system = manual_system
    #     manual_generator.iterable_system = SystemIterator(manual_system) # Create iterator for it
    #     # Generate path without iterations for this specific system
    #     manual_input_angles = [step * manual_system.stepSize for step in range(int(manual_system.resolution))]
    #     manual_generator._generate_and_save_path(manual_system, "manual_example", manual_input_angles)
    #     logging.info("Manual example finished.")
    #
    # except ValueError as e:
    #     logging.error(f"Manual example failed: {e}")
    # except Exception as e:
    #      logging.error(f"Manual example failed with unexpected error: {e}", exc_info=True)