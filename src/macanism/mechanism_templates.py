"""
Mechanism Templates

This module provides functions to generate pre-defined mechanism structures
(joints, vectors, loop equations, guesses) based on input parameters.
These templates can be used for systematic generation of mechanism variations
for database creation or for instantiating known mechanism types.
"""
import numpy as np
import math

# Assuming this file is part of the macanism package, use relative imports
from .mechanism import get_joints, Joint
from .vectors import Vector # This is the kinematic Vector

def calculate_distance_and_angle(p1_coords, p2_coords):
    """Helper to calculate distance and angle between two points."""
    dx = p2_coords[0] - p1_coords[0]
    dy = p2_coords[1] - p1_coords[1]
    distance = math.sqrt(dx**2 + dy**2)
    angle = math.atan2(dy, dx)
    return distance, angle

def create_four_bar_linkage_template(params: dict):
    """
    Creates a four-bar linkage mechanism based on specified parameters.

    Args:
        params (dict): A dictionary containing the parameters:
            - "r_O2A": Length of the crank (O2A).
            - "r_AB": Length of the coupler (AB).
            - "r_O4B": Length of the rocker (O4B).
            - "O4_x": X-coordinate of the fixed pivot O4 (O2 is at origin).
            - "O4_y": Y-coordinate of the fixed pivot O4.
            - "input_revolutions": Number of full input crank revolutions (e.g., 1 for 360 deg, 2 for 720 deg). Default 1.
            - "num_steps": Number of steps for the input motion. Default 180.
            - "output_joint_name": Name of the joint whose path is of primary interest (e.g., "B" or a coupler point name).
                                   If a coupler point, "coupler_point_params" should be provided.
            - "coupler_point_params" (optional): dict for a coupler point, e.g.,
                {"name": "P", "ref_joint_A_name": "A", "ref_joint_B_name": "B", "dist_AP": 1.0, "angle_PAB_deg": 30}
                (Note: Coupler point implementation requires adding a vector from A to P, and ensuring P is a joint)


    Returns:
        tuple: (vectors_tuple, origin_joint, loops_func, guess_array, input_motion_array, output_joint_name_str)
               Returns None if parameters are invalid (e.g., O4 coincides with O2).
    """
    O2_coords = (0.0, 0.0)
    O4_coords = (params["O4_x"], params["O4_y"])

    if O2_coords == O4_coords:
        print("Warning: O2 and O4 coincide. Invalid four-bar linkage.")
        return None

    # Define Joints
    # Standard joints for a 4-bar: O2 (origin), A (crank end), B (coupler-rocker), O4 (rocker pivot)
    joint_names = "O2 A B O4"
    coupler_point_name = None
    if "coupler_point_params" in params and params["coupler_point_params"]:
        coupler_point_name = params["coupler_point_params"]["name"]
        if coupler_point_name and coupler_point_name not in "O2ABO4": # Avoid name clashes
             joint_names += f" {coupler_point_name}"

    joints_list = get_joints(joint_names)
    O2, A, B, O4 = joints_list[0], joints_list[1], joints_list[2], joints_list[3]

    output_joint_actual = None
    if params.get("output_joint_name") == "B":
        output_joint_actual = B
    # Handle coupler point if specified
    P = None # Coupler point joint
    vec_AP = None # Vector for coupler point

    if coupler_point_name:
        P = next(j for j in joints_list if j.name == coupler_point_name)
        if params.get("output_joint_name") == coupler_point_name:
            output_joint_actual = P

        cp_params = params["coupler_point_params"]
        # Assuming ref_joint_A_name is 'A' and ref_joint_B_name is 'B' for simplicity here.
        # A more robust implementation would fetch joints by cp_params["ref_joint_A_name"], etc.
        # Vector AP: r = dist_AP, theta = angle of AB + angle_PAB_deg
        # This requires knowing angle of AB, which is an unknown (x[0] in loops).
        # So, vector AP must be defined relative to AB.
        # For now, let's define AP as a fixed vector relative to A, with its angle defined from AB.
        # This means AP's angle is x[0] (angle of AB) + angle_PAB_rad
        # This makes the coupler point definition a bit more complex for the loops.
        # Alternative: define AP as a vector from A, with fixed length and angle relative to AB.
        # The vector from A to P. Its angle will be theta_AB + angle_PAB.
        # This means its angle is x[0] (theta_AB) + angle_PAB_rad
        # This is tricky because x[0] is an unknown.
        # A simpler way for a fixed coupler point on link AB:
        # Define vector AP. Its length is params["dist_AP"]. Its angle is fixed relative to vector AB.
        # Let angle_AB be theta_coupler (unknown x[0]).
        # Angle of AP = theta_coupler + params["angle_PAB_deg"]
        # This means vec_AP = Vector((A, P), r=cp_params["dist_AP"])
        # And its angle in the loop equation would be x[0] + math.radians(cp_params["angle_PAB_deg"])
        # This requires a third loop equation or for P to be determined after A and B are found.

        # For simplicity in this first pass, let's assume the output_joint_name is one of O2, A, B, O4.
        # Full coupler point path generation will require extending the loops or post-calculation.
        # For now, if P is requested, we'll just use B as a placeholder output.
        if output_joint_actual is None and P is not None:
            print(f"Warning: Coupler point {P.name} path calculation not fully implemented in this basic template. Using B as output.")
            output_joint_actual = B
        elif output_joint_actual is None:
             print(f"Warning: Output joint {params.get('output_joint_name')} not recognized. Using B as output.")
             output_joint_actual = B


    # Define Vectors
    r_O2A = params["r_O2A"]
    r_AB = params["r_AB"]
    r_O4B = params["r_O4B"]

    dist_O2O4, angle_O2O4 = calculate_distance_and_angle(O2_coords, O4_coords)

    # Vector definitions based on example:
    # a: crank (O2A)
    # b: ground (O2O4)
    # c: coupler (AB)
    # d: rocker (O4B) - careful with direction for loop closure O2-A-B-O4-O2

    vec_a_crank = Vector((O2, A), r=r_O2A) # Input vector, angle 'inp'
    vec_b_ground = Vector((O2, O4), r=dist_O2O4, theta=angle_O2O4, style='ground') # Fixed ground link
    vec_c_coupler = Vector((A, B), r=r_AB) # Angle is x[0]
    vec_d_rocker = Vector((O4, B), r=r_O4B) # Angle is x[1]

    vectors_list = [vec_a_crank, vec_b_ground, vec_c_coupler, vec_d_rocker]

    # If coupler point P is defined and attached to A, add vec_AP
    # For now, we are not adding vec_AP to the main vector list for fsolve,
    # as its path would be calculated after solving for A and B.

    def loops_four_bar(x, inp):
        """
        Loop closure equations for the four-bar linkage.
        O2 -> A -> B -> O4 -> O2 (vector loop)
        vec_a_crank(inp) + vec_c_coupler(x[0]) - vec_d_rocker(x[1]) - vec_b_ground() = 0
        x[0] is angle of coupler (AB)
        x[1] is angle of rocker (O4B)
        inp is angle of crank (O2A)
        """
        temp = np.zeros((1, 2)) # One loop equation, two components (x, y)

        # Call .get() method which is vector_instance.pos.get(), results are [x,y,z]
        # We need to slice to get [x,y] for 2D calculations.
        term1 = vec_a_crank(inp)[:2]
        term2 = vec_c_coupler(x[0])[:2] # x[0] is theta_AB
        term3 = vec_d_rocker(x[1])[:2] # x[1] is theta_O4B
        term4 = vec_b_ground()[:2]     # Ground link, fixed angle

        # Loop: O2A + AB - O4B - O2O4 = 0  (if O4B is B to O4, then O2A + AB + BO4 - O2O4 = 0)
        # If vec_d_rocker is (O4, B), then its components are from O4 to B.
        # Loop: O2A + AB = O2O4 + O4B
        # O2A(inp) + AB(x[0]) - O4B(x[1]) - O2O4() = 0
        temp[0] = term1 + term2 - term3 - term4
        return temp.flatten()

    # Guess values for unknowns (angles x[0], x[1])
    # A simple guess: coupler roughly horizontal, rocker roughly vertical.
    # These are initial guesses for theta_AB and theta_O4B in radians.
    # This needs to be robust or adaptable.
    # Example from fourbarlinkage.py: np.deg2rad([50, 120])
    # For a general case, this is hard. Let's use a common starting guess.
    # Angle of AB (coupler) relative to horizontal.
    # Angle of O4B (rocker) relative to horizontal.
    # A better guess might be derived from initial geometry if possible.
    # For now, a placeholder:
    guess_theta_AB = math.atan2(O4_coords[1] - r_O2A * math.sin(0), O4_coords[0] - r_O2A * math.cos(0) - dist_O2O4 * math.cos(angle_O2O4)) # very rough
    guess_theta_O4B = math.atan2(0 - O4_coords[1], r_O2A + r_AB - O4_coords[0]) # very rough
    initial_guess = np.array([np.deg2rad(45), np.deg2rad(135)]) # Angles for coupler and rocker

    num_revolutions = params.get("input_revolutions", 1)
    num_steps = params.get("num_steps", 180) # Default 180 steps per revolution
    total_steps = num_steps * num_revolutions
    input_motion = np.linspace(0, num_revolutions * 2 * np.pi, total_steps)

    # Determine the actual output joint object based on name
    # The 'output_joint_actual' was determined earlier.
    # If it's a coupler point not part of the main kinematic chain for fsolve,
    # its path needs to be calculated *after* the main chain is solved.
    # This template currently only supports O2, A, B, O4 as direct outputs from fsolve chain.

    return (tuple(vectors_list), O2, loops_four_bar, initial_guess, input_motion, output_joint_actual.name)

# To add more templates:
# def create_crank_slider_template(params: dict):
#     ...
#     return (vectors_tuple, origin_joint, loops_func, guess_array, input_motion_array, output_joint_name_str)

if __name__ == '__main__':
    # Example usage:
    test_params_4bar = {
        "r_O2A": 1.0, "r_AB": 2.5, "r_O4B": 2.0,
        "O4_x": 2.0, "O4_y": 0.5,
        "output_joint_name": "B",
        "input_revolutions": 1,
        "num_steps": 180
    }
    result = create_four_bar_linkage_template(test_params_4bar)
    if result:
        vectors, origin, loops_func, guess, motion, out_j_name = result
        print(f"Generated template for Four-Bar Linkage. Output joint: {out_j_name}")
        print(f"Number of vectors: {len(vectors)}")
        print(f"Origin: {origin.name}")
        print(f"Initial guess: {guess}")
        print(f"Motion steps: {len(motion)}")

    # Example with a coupler point (conceptual, path calculation for P not fully integrated here)
    test_params_coupler = {
        "r_O2A": 1.0, "r_AB": 3.0, "r_O4B": 2.5,
        "O4_x": 3.0, "O4_y": 0.0,
        "output_joint_name": "P", # Requesting coupler point
        "coupler_point_params": {"name": "P", "dist_AP": 1.5, "angle_PAB_deg": -30},
        "input_revolutions": 1, "num_steps": 10 # fewer steps for quick test
    }
    # result_coupler = create_four_bar_linkage_template(test_params_coupler)
    # if result_coupler:
    #     # Note: The path for 'P' would need post-processing based on A and B's paths.
    #     # The current template returns B's name if P is not directly solvable in the main loop.
    #     print(f"Generated template for Four-Bar with Coupler Point. Output joint (effective): {result_coupler[5]}")
    pass


def create_crank_slider_template(params: dict):
    """
    Creates a simple crank-slider mechanism.
    O2 (origin, crank pivot) is at (0,0). Slider B moves along the X-axis.

    Args:
        params (dict): A dictionary containing the parameters:
            - "r_crank": Length of the crank (O2A).
            - "r_conrod": Length of the connecting rod (AB).
            - "input_revolutions": Number of full input crank revolutions. Default 1.
            - "num_steps": Number of steps for the input motion. Default 180.
            - "output_joint_name": Name of the joint for path output ("A" or "B"). Default "B".

    Returns:
        tuple: (vectors_tuple, origin_joint, loops_func, guess_array, input_motion_array, output_joint_name_str)
               Returns None if parameters are invalid (e.g., conrod too short).
    """
    r_crank = params["r_crank"]
    r_conrod = params["r_conrod"]

    if r_conrod < r_crank:
        # This might still be valid but can lead to full rotation issues if not handled by solver/guess
        print(f"Warning: Crank-slider conrod length ({r_conrod}) is less than crank length ({r_crank}). May have assembly issues.")
    if r_crank <= 0 or r_conrod <= 0:
        print("Warning: Link lengths must be positive.")
        return None

    # Define Joints
    O2, A, B = get_joints("O2 A B")

    # Define Vectors
    vec_crank = Vector((O2, A), r=r_crank)  # Input vector, angle 'inp'
    vec_conrod = Vector((A, B), r=r_conrod) # Angle is x[0] (theta_conrod)
    # Slider position vector: O2 to B. B moves along X-axis.
    # So, this vector has theta=0, and its length is x[1] (slider_displacement_r_O2B)
    vec_slider_pos = Vector((O2, B), theta=0) # r is x[1]

    vectors_list = [vec_crank, vec_conrod, vec_slider_pos]

    def loops_crank_slider(x, inp):
        """
        Loop closure: O2A + AB - O2B = 0 (if O2B is vector from O2 to B)
        x[0]: angle of conrod (AB)
        x[1]: length of slider_pos vector (r_O2B, displacement of B from O2)
        inp: angle of crank (O2A)
        """
        temp = np.zeros((1, 2)) # One loop equation (vector), two components (x, y)

        term1_crank_xy = vec_crank(inp)[:2]
        term2_conrod_xy = vec_conrod(x[0])[:2] # x[0] is theta_conrod
        # For vec_slider_pos, theta is fixed at 0. Its get method is _slip(r_val).
        # So, x[1] is r_val for vec_slider_pos.
        term3_slider_pos_xy = vec_slider_pos(x[1])[:2]

        temp[0] = term1_crank_xy + term2_conrod_xy - term3_slider_pos_xy
        return temp.flatten()

    # Initial guess for unknowns: x[0] (theta_conrod), x[1] (r_O2B)
    # A simple starting configuration: crank at 0 deg, A at (r_crank, 0).
    # If conrod is horizontal, B is at (r_crank + r_conrod, 0).
    # So, initial theta_conrod = 0, initial r_O2B = r_crank + r_conrod.
    # This might not be a great guess for all input angles 'inp'.
    # fsolve usually starts solving from inp[0].
    # Let's assume inp[0] is 0.
    initial_guess_theta_conrod = 0.0
    # If crank is at 0 deg (inp=0), A is at (r_crank, 0).
    # B is on x-axis. If conrod is not shorter than crank:
    if r_conrod >= r_crank:
         initial_guess_r_O2B = r_crank + r_conrod # Max extension
    else: # conrod shorter than crank, slider cannot reach this far.
         initial_guess_r_O2B = r_conrod - r_crank # if crank angle is pi
         # if crank angle is 0, then B is at r_crank - r_conrod (if r_crank > r_conrod)
         # or r_conrod + r_crank (if conrod can flip back)
         # This case is tricky. A better guess:
         # When crank angle is 0, A is at (r_crank, 0).
         # B.x = A.x + r_conrod * cos(theta_conrod)
         # B.y = A.y + r_conrod * sin(theta_conrod) = 0 => sin(theta_conrod)=0 => theta_conrod = 0 or pi
         # If theta_conrod = 0, B.x = r_crank + r_conrod.
         # If theta_conrod = pi, B.x = r_crank - r_conrod.
         initial_guess_r_O2B = r_crank + r_conrod # Simplest guess, fsolve might sort it out.

    initial_guess = np.array([initial_guess_theta_conrod, initial_guess_r_O2B])

    num_revolutions = params.get("input_revolutions", 1)
    num_steps = params.get("num_steps", 180)
    total_steps = num_steps * num_revolutions
    input_motion = np.linspace(0, num_revolutions * 2 * np.pi, total_steps)

    output_joint_name_str = params.get("output_joint_name", "B")
    if output_joint_name_str not in ["A", "B"]:
        print(f"Warning: Invalid output_joint_name '{output_joint_name_str}' for CrankSlider. Defaulting to 'B'.")
        output_joint_name_str = "B"

    return (tuple(vectors_list), O2, loops_crank_slider, initial_guess, input_motion, output_joint_name_str)


if __name__ == '__main__':
    # Example usage for four-bar:
    test_params_4bar = {
        "r_O2A": 1.0, "r_AB": 2.5, "r_O4B": 2.0,
        "O4_x": 2.0, "O4_y": 0.5,
        "output_joint_name": "B",
        "input_revolutions": 1,
        "num_steps": 180
    }
    result_4bar = create_four_bar_linkage_template(test_params_4bar)
    if result_4bar:
        # ... (previous print statements for 4bar) ...
        pass

    # Example usage for crank-slider:
    test_params_cs = {
        "r_crank": 1.0, "r_conrod": 3.0,
        "output_joint_name": "B",
        "input_revolutions": 1, "num_steps": 180
    }
    result_cs = create_crank_slider_template(test_params_cs)
    if result_cs:
        vectors_cs, origin_cs, loops_cs, guess_cs, motion_cs, out_j_name_cs = result_cs
        print(f"\nGenerated template for Crank-Slider. Output joint: {out_j_name_cs}")
        print(f"Number of vectors: {len(vectors_cs)}")
        print(f"Origin: {origin_cs.name}")
        print(f"Initial guess: {guess_cs}")
        print(f"Motion steps: {len(motion_cs)}")

    test_params_cs_short_conrod = {
        "r_crank": 2.0, "r_conrod": 1.5, # Conrod shorter than crank
        "output_joint_name": "B"
    }
    result_cs_short = create_crank_slider_template(test_params_cs_short_conrod)
    if result_cs_short:
        print(f"\nGenerated template for Crank-Slider (short conrod). Output joint: {result_cs_short[5]}")
