"""
Streamlit application for Macanism project - Interactive Demo
"""
import streamlit as st
import matplotlib.pyplot as plt
import math
import numpy as np # For 3-bar linkage calculations

# Ensure src is in path for Streamlit - this might be needed if running streamlit run app_streamlit.py from root
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from macanism.gears import SpurGear, SpurGearParameters
from macanism.linkages import LinkageSystem, Point as LinkagePoint, Link, Joint
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
st.set_page_config(page_title="Macanism Demos", layout="wide")
st.title("⚙️ Macanism Interactive Demos")
st.caption("Simplified demonstrations of macanism concepts. Full simulation logic is under development.")

tab1, tab2, tab3, tab4 = st.tabs(["3-Bar Linkage", "4-Bar Linkage", "Cam Demo (Conceptual)", "Spur Gear Demo"])

# --- 3-Bar Linkage Tab ---
with tab1:
    st.header("3-Bar Linkage (Triangle Structure)")
    col1_3bar, col2_3bar = st.columns([1, 2])
    with col1_3bar:
        st.markdown("A 3-bar linkage forms a rigid triangular structure.")
        l1 = st.slider("Length of Link 1 (P0-P1)", 0.1, 5.0, 2.0, key="3bar_l1")
        l2 = st.slider("Length of Link 2 (P1-P2)", 0.1, 5.0, 2.5, key="3bar_l2")
        l3 = st.slider("Length of Link 3 (P2-P0)", 0.1, 5.0, 3.0, key="3bar_l3")

        # Basic check for triangle inequality
        if not (l1 + l2 > l3 and l1 + l3 > l2 and l2 + l3 > l1):
            st.warning("These lengths do not form a valid triangle!")
            p0, p1, p2 = CommonPoint(0,0), CommonPoint(0,0), CommonPoint(0,0)
        else:
            p0 = CommonPoint(0, 0)
            p1 = CommonPoint(l1, 0)
            # Calculate P2 using trigonometry (cosine rule)
            # Angle at P0 for triangle P0,P1,P2
            try:
                angle_at_p0 = math.acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3))
                p2_x = l3 * math.cos(angle_at_p0)
                p2_y = l3 * math.sin(angle_at_p0)
                p2 = CommonPoint(p2_x, p2_y)
            except ValueError: # Math domain error if lengths are impossible despite inequality (e.g. rounding)
                st.error("Could not calculate P2 coordinates - check lengths.")
                p0, p1, p2 = CommonPoint(0,0), CommonPoint(l1,0), CommonPoint(0,0)

        points_3bar = [p0, p1, p2]
        links_3bar = [(0,1), (1,2), (2,0)]

    with col2_3bar:
        fig_3bar, ax_3bar = plt.subplots(figsize=(6,5))
        plot_linkage_demo(points_3bar, links_3bar, ax_3bar, "3-Bar Linkage Structure")
        st.pyplot(fig_3bar)

# --- 4-Bar Linkage Tab ---
with tab2:
    st.header("4-Bar Linkage Demo")
    col1_4bar, col2_4bar = st.columns([1, 2])
    with col1_4bar:
        st.markdown("Define a four-bar linkage by its link lengths and input crank angle.")
        l_crank = st.slider("Crank Length (O2A)", 0.1, 5.0, 1.0, key="4bar_l_crank")
        l_coupler = st.slider("Coupler Length (AB)", 0.1, 5.0, 2.5, key="4bar_l_coupler")
        l_rocker = st.slider("Rocker Length (BO4)", 0.1, 5.0, 2.0, key="4bar_l_rocker")
        l_ground = st.slider("Ground Link Length (O2O4)", 0.1, 5.0, 3.0, key="4bar_l_ground")
        crank_angle_deg = st.slider("Input Crank Angle (degrees)", 0.0, 360.0, 30.0, key="4bar_angle")
        crank_angle_rad = math.radians(crank_angle_deg)

        # Fixed points
        o2 = CommonPoint(0, 0)
        o4 = CommonPoint(l_ground, 0)

        # Crank end point A
        ax = o2.x + l_crank * math.cos(crank_angle_rad)
        ay = o2.y + l_crank * math.sin(crank_angle_rad)
        a = CommonPoint(ax, ay)

        # Placeholder for point B (coupler-rocker joint) - requires solving kinematic equations
        # For demo, we'll estimate B based on simple assumptions or make it fixed if not solvable easily.
        # This is NOT a real kinematic solution.
        # Try to find intersection of two circles (coupler from A, rocker from O4)
        d_ao4 = a.distance_to(o4)
        b_x, b_y = o4.x - l_rocker, o4.y # Default if not solvable
        try:
            if d_ao4 <= l_coupler + l_rocker and d_ao4 >= abs(l_coupler - l_rocker): # Check if solution exists
                # Using cosine rule to find angle at A in triangle AO4B
                alpha = math.acos((d_ao4**2 + l_coupler**2 - l_rocker**2) / (2 * d_ao4 * l_coupler))
                # Angle of line AO4
                theta_ao4 = math.atan2(o4.y - a.y, o4.x - a.x)
                # We need to choose one of two solutions for B (e.g., elbow up/down)
                # For simplicity, let's choose one.
                b_angle = theta_ao4 + alpha # or theta_ao4 - alpha
                b_x = a.x + l_coupler * math.cos(b_angle)
                b_y = a.y + l_coupler * math.sin(b_angle)
            else:
                st.warning("Grashof condition violated or links cannot connect with current crank angle.")
        except (ValueError, ZeroDivisionError):
            st.warning("Could not calculate point B reliably with current link lengths/angle.")

        b = CommonPoint(b_x, b_y)

        points_4bar = [o2, a, b, o4]
        links_4bar = [(0,1), (1,2), (2,3), (3,0)] # O2A, AB, BO4, O4O2(ground)

    with col2_4bar:
        fig_4bar, ax_4bar = plt.subplots(figsize=(6,5))
        plot_linkage_demo(points_4bar, links_4bar, ax_4bar, "4-Bar Linkage (Crank Driven - Simplified Solve)")
        st.pyplot(fig_4bar)

# --- Cam Demo Tab (Conceptual) ---
with tab3:
    st.header("Cam Design (Conceptual Demo)")
    col1_cam, col2_cam = st.columns([1, 2])
    with col1_cam:
        st.markdown("Select conceptual cam parameters.")
        cam_type = st.selectbox("Cam Type", ["Plate Cam", "Grooved Cam", "Cylindrical Cam"])
        follower_type = st.selectbox("Follower Type", ["Roller Follower", "Flat-Faced Follower", "Pointed Follower"])
        motion_type = st.selectbox("Primary Motion Segment", ["Rise (Cycloidal)", "Fall (Harmonic)", "Dwell"])

        st.write(f"**Selected Cam:** {cam_type}")
        st.write(f"**Follower:** {follower_type}")
        st.write(f"**Motion:** {motion_type}")
        st.info("Full cam profile generation and SVAJ analysis are future features.")

    with col2_cam:
        st.markdown("**Conceptual Visualization Placeholder**")
        # Placeholder for a cam image or basic sketch
        fig_cam, ax_cam = plt.subplots(figsize=(5,5))
        ax_cam.text(0.5, 0.5, f"{cam_type}\nwith\n{follower_type}\n({motion_type})\n\n(Visualization TBD)",
                    ha='center', va='center', fontsize=12, wrap=True)
        ax_cam.set_xticks([])
        ax_cam.set_yticks([])
        st.pyplot(fig_cam)

# --- Spur Gear Demo Tab ---
with tab4:
    st.header("Spur Gear Demo")
    col1_gear, col2_gear = st.columns([1, 2])
    with col1_gear:
        st.markdown("Define spur gear parameters.")
        gear_name = st.text_input("Gear Name", "DemoGear", key="gear_name_input")
        param_type = st.radio("Define by:", ("Module", "Diametral Pitch"), horizontal=True, key="gear_param_type")

        module_val, dp_val = None, None
        if param_type == "Module":
            module_val = st.number_input("Module", min_value=0.1, value=2.0, step=0.1, key="gear_module")
        else:
            dp_val = st.number_input("Diametral Pitch", min_value=0.1, value=10.0, step=0.1, key="gear_dp")

        num_teeth_val = st.slider("Number of Teeth", 5, 200, 30, 1, key="gear_teeth")
        pressure_angle_val = st.slider("Pressure Angle (deg)", 10.0, 30.0, 20.0, 0.5, key="gear_pa")

        try:
            gear_params = SpurGearParameters(
                module=module_val, diametral_pitch=dp_val,
                num_teeth=num_teeth_val, pressure_angle_deg=pressure_angle_val
            )
            current_gear_demo = SpurGear(params=gear_params, name=gear_name)
            st.subheader("Calculated Properties")
            props = current_gear_demo.get_properties()
            for key, val in props.items():
                st.write(f"**{key.replace('_', ' ').title()}:** {val:.4f}" if isinstance(val, float) else str(val))
        except ValueError as e:
            st.error(f"Error in gear parameters: {e}")
            current_gear_demo = None
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            current_gear_demo = None

    with col2_gear:
        fig_gear_demo, ax_gear_demo = plt.subplots(figsize=(6,5))
        if current_gear_demo:
            plot_gear_demo(current_gear_demo, ax_gear_demo)
            st.pyplot(fig_gear_demo)
        else:
            st.warning("Gear not defined due to parameter error.")

st.sidebar.info(
    "This is a basic interactive demo for the Macanism project. "
    "Core simulation logic is under development."
)
st.sidebar.markdown("""### How to Run:
1. Ensure Streamlit is installed (`uv pip install streamlit matplotlib` or `pip install streamlit matplotlib`).
2. Save this code as `app_streamlit.py` in the project root.
3. Open your terminal in the project root directory.
4. Run: `streamlit run app_streamlit.py`""")