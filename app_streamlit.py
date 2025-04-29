"""
Streamlit application for Macanism project - Interactive Demo
"""
import streamlit as st
import matplotlib.pyplot as plt
import math

# Ensure src is in path for Streamlit - this might be needed if running streamlit run app_streamlit.py from root
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from macanism.gears import SpurGear, SpurGearParameters
from macanism.linkages import LinkageSystem, Point, Link, Joint
from macanism.common_elements import Point as CommonPoint # Alias to avoid confusion if used directly

# --- Helper Functions for Plotting (Simplified) ---
def plot_gear(gear: SpurGear, ax):
    """Rudimentary plot of a spur gear."""
    ax.clear()
    # gear_points_sets = gear.get_full_gear_points() # List of lists of Points
    # for tooth_flank_points in gear_points_sets:
    #     if not tooth_flank_points: continue
    #     x_coords = [p.x for p in tooth_flank_points]
    #     y_coords = [p.y for p in tooth_flank_points]
    #     # Closing the flank for a simple look (not a real tooth profile)
    #     # x_coords.append(tooth_flank_points[0].x)
    #     # y_coords.append(tooth_flank_points[0].y)
    #     ax.plot(x_coords, y_coords, 'b-')

    # Simplified: Draw pitch circle and base circle for now as full profile is dummy
    pitch_radius = gear.params.pitch_diameter / 2.0
    base_radius = gear.params.base_diameter / 2.0

    pitch_circle = plt.Circle((0,0), pitch_radius, color='gray', fill=False, linestyle='--', label='Pitch Circle')
    base_circle = plt.Circle((0,0), base_radius, color='red', fill=False, linestyle=':', label='Base Circle')
    ax.add_artist(pitch_circle)
    ax.add_artist(base_circle)

    # Placeholder for actual tooth plotting - the current generate_tooth_profile is too simple
    ax.set_title(f"Simplified Gear: {gear.name}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis('equal')
    ax.grid(True)
    ax.legend()
    max_dim = pitch_radius * 1.2
    ax.set_xlim([-max_dim, max_dim])
    ax.set_ylim([-max_dim, max_dim])

def plot_linkage(linkage: LinkageSystem, ax):
    """Rudimentary plot of a linkage system."""
    ax.clear()
    for link_id, link_data in linkage.links.items():
        p1 = link_data.joints[0].position
        p2 = link_data.joints[1].position
        ax.plot([p1.x, p2.x], [p1.y, p2.y], 'o-', label=link_id, lw=3)

    for joint_id, joint_data in linkage.joints.items():
        ax.plot(joint_data.position.x, joint_data.position.y, 'ks', ms=8, label=f"Joint {joint_id}")

    ax.set_title(f"Linkage: {linkage.name}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis('equal')
    ax.grid(True)
    # ax.legend() # Can get crowded quickly, enable if needed

# --- Streamlit App Layout ---
st.set_page_config(page_title="Macanism Interactive Demo", layout="wide")
st.title("⚙️ Macanism Interactive Demonstrator")
st.caption("Define and visualize basic macanisms. Note: Simulations are currently placeholders.")

# --- Gear Section ---
st.header("Spur Gear Designer")
col1_gear, col2_gear = st.columns([1, 2])

with col1_gear:
    st.subheader("Parameters")
    gear_name = st.text_input("Gear Name", "MySpurGear")
    param_type = st.radio("Define by:", ("Module", "Diametral Pitch"), horizontal=True)

    module_val, dp_val = None, None
    if param_type == "Module":
        module_val = st.number_input("Module (e.g., 2)", min_value=0.1, value=2.0, step=0.1)
    else:
        dp_val = st.number_input("Diametral Pitch (e.g., 10)", min_value=0.1, value=10.0, step=0.1)

    num_teeth_val = st.slider("Number of Teeth", min_value=5, max_value=200, value=30, step=1)
    pressure_angle_val = st.slider("Pressure Angle (degrees)", min_value=10.0, max_value=30.0, value=20.0, step=0.5)

    try:
        gear_params = SpurGearParameters(
            module=module_val,
            diametral_pitch=dp_val,
            num_teeth=num_teeth_val,
            pressure_angle_deg=pressure_angle_val
        )
        current_gear = SpurGear(params=gear_params, name=gear_name)
        # current_gear.generate_tooth_profile() # Profile generation is very basic

        st.subheader("Calculated Properties")
        props = current_gear.get_properties()
        for key, val in props.items():
            st.metric(label=key.replace("_", " ").title(), value=f"{val:.4f}" if isinstance(val, float) else str(val))

    except ValueError as e:
        st.error(f"Error in gear parameters: {e}")
        current_gear = None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        current_gear = None

with col2_gear:
    st.subheader("Gear Visualization (Conceptual)")
    fig_gear, ax_gear = plt.subplots(figsize=(6,6))
    if current_gear:
        plot_gear(current_gear, ax_gear)
        st.pyplot(fig_gear)
    else:
        st.warning("Gear not defined due to parameter error.")

st.divider()

# --- Linkage Section ---
st.header("Linkage Designer (Simple Four-Bar)")
col1_linkage, col2_linkage = st.columns([1, 2])

with col1_linkage:
    st.subheader("Parameters (Four-Bar Linkage)")
    linkage_name = st.text_input("Linkage Name", "MyFourBar", key="linkage_name_input")

    st.markdown("Define initial joint positions for a four-bar:")
    # O2 (Ground Pivot 1) - Fixed at origin
    st.markdown("**Joint O2 (Ground Pivot 1):** Fixed at (0, 0)")
    o2_pos = CommonPoint(0,0)

    # A (Input Crank Pivot)
    a_x = st.number_input("Joint A.x (Input Crank Length)", value=1.0, key="ax")
    a_y = st.number_input("Joint A.y", value=0.0, disabled=True, help="Set by crank angle for simulation")
    a_pos = CommonPoint(a_x, a_y)

    # B (Coupler End / Rocker Start)
    b_x = st.number_input("Joint B.x", value=2.5, key="bx")
    b_y = st.number_input("Joint B.y", value=1.0, key="by")
    b_pos = CommonPoint(b_x, b_y)

    # O4 (Ground Pivot 2)
    o4_x = st.number_input("Joint O4.x (Ground Pivot 2)", value=2.0, key="o4x")
    o4_y = st.number_input("Joint O4.y", value=0.0, key="o4y") # Typically on x-axis for simplicity
    o4_pos = CommonPoint(o4_x, o4_y)

    # Create Linkage System
    current_linkage = LinkageSystem(name=linkage_name)
    try:
        j_o2 = current_linkage.add_joint("O2", o2_pos)
        j_a  = current_linkage.add_joint("A", a_pos)
        j_b  = current_linkage.add_joint("B", b_pos)
        j_o4 = current_linkage.add_joint("O4", o4_pos)

        # Define links based on these joints
        l_crank = current_linkage.add_link("Crank (O2A)", "O2", "A")
        l_coupler = current_linkage.add_link("Coupler (AB)", "A", "B")
        l_rocker = current_linkage.add_link("Rocker (BO4)", "B", "O4")
        l_ground = current_linkage.add_link("Ground (O4O2)", "O4", "O2") # Ground link for completeness

        st.subheader("Linkage Properties")
        st.write(f"**{l_crank.id}:** Length = {l_crank.length:.3f}")
        st.write(f"**{l_coupler.id}:** Length = {l_coupler.length:.3f}")
        st.write(f"**{l_rocker.id}:** Length = {l_rocker.length:.3f}")
        st.write(f"**{l_ground.id}:** Length = {l_ground.length:.3f}")

        input_angle_slider = st.slider("Input Crank Angle (O2A, degrees from X-axis)",
                                       min_value=0.0, max_value=360.0, value=0.0, step=1.0,
                                       key="crank_angle")
        # This part would ideally update j_a's position and call linkage.solve_kinematics()
        # For now, it's just a slider, actual simulation update is not wired yet.
        if st.button("Run Simulation Step (Placeholder)", key="run_linkage_sim"):
            st.info(f"Placeholder: Would simulate for crank angle {input_angle_slider} deg. Kinematic solver not yet implemented.")
            # In a real app:
            # j_a_new_x = l_crank.length * math.cos(math.radians(input_angle_slider))
            # j_a_new_y = l_crank.length * math.sin(math.radians(input_angle_slider))
            # current_linkage.joints["A"].position = CommonPoint(j_a_new_x, j_a_new_y)
            # current_linkage.solve_kinematics(input_angle_slider)
            # And then re-plot

    except ValueError as e:
        st.error(f"Error in linkage definition: {e}")
        current_linkage = None
    except Exception as e:
        st.error(f"An unexpected error occurred in linkage setup: {e}")
        current_linkage = None

with col2_linkage:
    st.subheader("Linkage Visualization (Initial Configuration)")
    fig_linkage, ax_linkage = plt.subplots(figsize=(6,6))
    if current_linkage:
        plot_linkage(current_linkage, ax_linkage)
        st.pyplot(fig_linkage)
    else:
        st.warning("Linkage not defined due to error.")

st.sidebar.info(
    "This is a basic interactive demo for the Macanism project. "
    "Core simulation logic is under development."
)
st.sidebar.markdown("""### How to Run:
1. Ensure Streamlit is installed (`uv pip install streamlit matplotlib` or `pip install streamlit matplotlib`).
2. Save this code as `app_streamlit.py` in the project root.
3. Open your terminal in the project root directory.
4. Run: `streamlit run app_streamlit.py`""")