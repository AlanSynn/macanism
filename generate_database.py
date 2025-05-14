import sys
import os
import json
import numpy as np
import itertools
import hashlib

# Ensure src directory is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from macanism.mechanism_templates import create_four_bar_linkage_template, create_crank_slider_template
from macanism import macanism # Imports the class from mechanism.py via __init__.py

def ensure_dir(directory):
    """Ensures that a directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_four_bar_variations(base_output_dir="output_db"):
    """
    Generates path data for variations of a four-bar linkage.
    """
    template_name = "FourBarLinkage_TypeA"
    output_dir = os.path.join(base_output_dir, template_name)
    ensure_dir(output_dir)

    # Define parameter ranges
    # For testing, keep ranges small and steps few
    param_ranges = {
        "r_O2A": np.linspace(0.8, 1.2, 2),  # Crank length
        "r_AB": np.linspace(2.0, 3.0, 2),   # Coupler length
        "r_O4B": np.linspace(1.5, 2.5, 2),  # Rocker length
        "O4_x": np.linspace(2.0, 3.0, 2),    # O4 pivot x-coordinate
        "O4_y": np.linspace(-0.5, 0.5, 2), # O4 pivot y-coordinate
        # "output_joint_name": ["B"], # For now, always B
        # "input_revolutions": [1],
        # "num_steps": [180] # Fixed for now
    }

    # Create all combinations of parameters
    keys, values = zip(*param_ranges.items())
    parameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Starting generation for {template_name}. Total variations: {len(parameter_combinations)}")
    generated_count = 0

    for i, p_combo in enumerate(parameter_combinations):
        # Add fixed parameters
        current_params = p_combo.copy()
        current_params["output_joint_name"] = "B"
        current_params["input_revolutions"] = 1
        current_params["num_steps"] = 180 # Standard 2-degree steps for one revolution

        print(f"\nProcessing variation {i+1}/{len(parameter_combinations)}: {current_params}")

        template_result = create_four_bar_linkage_template(current_params)
        if not template_result:
            print(f"Skipping invalid parameters: {current_params}")
            continue

        vectors, origin, loops_func, guess, motion, output_joint_name_str = template_result

        # Create a unique ID for this variation (hash of sorted param items)
        param_str = json.dumps(current_params, sort_keys=True)
        variation_id = hashlib.md5(param_str.encode('utf-8')).hexdigest()[:10]

        try:
            mech_instance = macanism(
                vectors=vectors,
                origin=origin,
                loops=loops_func,
                pos=motion,
                guess=(guess,) # guess needs to be a tuple for the macanism class
            )
            mech_instance.iterate() # Calculate all positions

            # Find the output joint object to extract its path
            output_joint_obj = None
            for joint_obj in mech_instance.joints:
                if joint_obj.name == output_joint_name_str:
                    output_joint_obj = joint_obj
                    break

            if not output_joint_obj or output_joint_obj.x_positions is None:
                print(f"Error: Output joint '{output_joint_name_str}' not found or path not calculated.")
                continue

            path_x = output_joint_obj.x_positions
            path_y = output_joint_obj.y_positions
            path_xy_coordinates = [[float(x), float(y)] for x, y in zip(path_x, path_y)]

            # Prepare data for JSON
            data_to_save = {
                "mechanism_template_name": template_name,
                "variation_id": variation_id,
                "parameters": current_params,
                "output_joint_name": output_joint_name_str,
                "path_xy_coordinates": path_xy_coordinates,
                # Optionally, include all joint paths for visualization later
                # "all_joint_paths": {
                #    j.name: [[float(x), float(y)] for x,y in zip(j.x_positions, j.y_positions)]
                #    for j in mech_instance.joints if j.x_positions is not None
                # }
            }

            # Save to JSON file
            filepath = os.path.join(output_dir, f"variation_{variation_id}.json")
            with open(filepath, 'w') as f:
                json.dump(data_to_save, f, indent=4)
            print(f"Successfully saved: {filepath}")
            generated_count += 1

        except Exception as e:
            print(f"Error during mechanism solution or data extraction for params {current_params}: {e}")
            # import traceback
            # traceback.print_exc() # For more detailed debugging if needed

    print(f"\nFinished generation for {template_name}. Successfully generated {generated_count} files.")


def generate_crank_slider_variations(base_output_dir="output_db"):
    """
    Generates path data for variations of a simple crank-slider linkage.
    """
    template_name = "CrankSlider_Simple"
    output_dir = os.path.join(base_output_dir, template_name)
    ensure_dir(output_dir)

    param_ranges = {
        "r_crank": np.linspace(0.5, 1.5, 2),  # Crank length
        "r_conrod": np.linspace(2.0, 4.0, 2), # Connecting rod length
    }

    keys, values = zip(*param_ranges.items())
    parameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"\nStarting generation for {template_name}. Total variations: {len(parameter_combinations)}")
    generated_count = 0

    for i, p_combo in enumerate(parameter_combinations):
        current_params = p_combo.copy()
        current_params["output_joint_name"] = "B" # Slider
        current_params["input_revolutions"] = 1
        current_params["num_steps"] = 180

        print(f"\nProcessing CS variation {i+1}/{len(parameter_combinations)}: {current_params}")

        template_result = create_crank_slider_template(current_params)
        if not template_result:
            print(f"Skipping invalid CS parameters: {current_params}")
            continue

        vectors, origin, loops_func, guess, motion, output_joint_name_str = template_result

        param_str = json.dumps(current_params, sort_keys=True)
        variation_id = hashlib.md5(param_str.encode('utf-8')).hexdigest()[:10]

        try:
            mech_instance = macanism(
                vectors=vectors, origin=origin, loops=loops_func,
                pos=motion, guess=(guess,)
            )
            mech_instance.iterate()

            output_joint_obj = next((j for j in mech_instance.joints if j.name == output_joint_name_str), None)

            if not output_joint_obj or output_joint_obj.x_positions is None:
                print(f"Error: CS Output joint '{output_joint_name_str}' not found or path not calculated.")
                continue

            path_xy_coordinates = [[float(x), float(y)] for x, y in zip(output_joint_obj.x_positions, output_joint_obj.y_positions)]

            data_to_save = {
                "mechanism_template_name": template_name,
                "variation_id": variation_id,
                "parameters": current_params,
                "output_joint_name": output_joint_name_str,
                "path_xy_coordinates": path_xy_coordinates,
            }

            filepath = os.path.join(output_dir, f"variation_{variation_id}.json")
            with open(filepath, 'w') as f:
                json.dump(data_to_save, f, indent=4)
            print(f"Successfully saved CS: {filepath}")
            generated_count += 1

        except Exception as e:
            print(f"Error during CS mechanism solution for params {current_params}: {e}")

    print(f"\nFinished generation for {template_name}. Successfully generated {generated_count} files.")


if __name__ == "__main__":
    generate_four_bar_variations()
    # To add more:
    generate_crank_slider_variations()
