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

def plot_gear_demo(gear: SpurGear, ax, center_x: float = 0.0, center_y: float = 0.0, rotation_angle_deg: float = 0.0):
    ax.clear()
    pitch_radius = gear.params.pitch_diameter / 2.0
    base_radius = gear.params.base_diameter / 2.0
    rotation_angle_rad = math.radians(rotation_angle_deg)

    # Apply rotation and translation to circle centers for plotting
    # For circles, we just need to transform their center point.
    # If we were plotting individual points of teeth, we'd transform each point.

    pitch_circle = plt.Circle((center_x, center_y), pitch_radius, color='gray', fill=False, linestyle='--', label=f'Pitch (d={pitch_radius*2:.2f})')
    base_circle = plt.Circle((center_x, center_y), base_radius, color='red', fill=False, linestyle=':', label=f'Base (d={base_radius*2:.2f})')
    ax.add_artist(pitch_circle)
    ax.add_artist(base_circle)

    # Simple line to indicate rotation
    if pitch_radius > 0:
        x_rot_line_end = center_x + pitch_radius * math.cos(rotation_angle_rad)
        y_rot_line_end = center_y + pitch_radius * math.sin(rotation_angle_rad)
        ax.plot([center_x, x_rot_line_end], [center_y, y_rot_line_end], color='blue', linestyle='-', lw=1, label="Orientation")

    ax.set_title(f"{gear.name} (Center:({center_x:.1f},{center_y:.1f}), Rot:{rotation_angle_deg:.0f}°)")
    ax.axis('equal')
    ax.grid(True)
    ax.legend(fontsize='small')
    max_dim = pitch_radius * 1.5 if pitch_radius > 0 else 1.0
    # Adjust plot limits based on center and max_dim
    ax.set_xlim([center_x - max_dim, center_x + max_dim])
    ax.set_ylim([center_y - max_dim, center_y + max_dim])

def plot_linkage_demo(points: list[CommonPoint], links_config: list[tuple[int,int]], ax, title="Linkage"):
    ax.clear()
    for i, (idx1, idx2) in enumerate(links_config):
        p1 = points[idx1]
        p2 = points[idx2]
        ax.plot([p1.x, p2.x], [p1.y, p2.y], 'o-', lw=3, label=f"L{i+1} ({p1.distance_to(p2):.2f})")

    for i, p in enumerate(points):
        ax.text(p.x + 0.05 * ax.get_xlim()[1], p.y + 0.05 * ax.get_ylim()[1], f"P{i}", fontsize=9)
        ax.plot(p.x, p.y, 'ks', ms=8)

    ax.set_title(title)
    ax.axis('equal')
    ax.grid(True)
    ax.legend(fontsize='small')
    # Auto-scaling for linkage plot can be tricky; might need manual adjustments or smarter limits
    all_x = [p.x for p in points]
    all_y = [p.y for p in points]
    if all_x and all_y:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        xrange = max_x - min_x if max_x > min_x else 1.0
        yrange = max_y - min_y if max_y > min_y else 1.0
        padding = 0.2 * max(xrange, yrange) # Add some padding
        ax.set_xlim([min_x - padding, max_x + padding])
        ax.set_ylim([min_y - padding, max_y + padding])

# --- Streamlit App Layout ---
st.set_page_config(page_title="Macanism Demos", layout="wide")
st.title("⚙️ Macanism Interactive Demos")
st.caption("Interactive tools for mechanism design and analysis.")

# Import for Phase 2 - Path to Mechanism
import glob # For finding database files
import json # For loading database files
from scipy.interpolate import interp1d # For resampling paths
from scipy.spatial.distance import directed_hausdorff # For path similarity

from macanism.mechanism_templates import create_four_bar_linkage_template, create_crank_slider_template
from macanism import macanism as MechanismSolver # Alias to avoid confusion with any 'macanism' module name
from streamlit_drawable_canvas import st_canvas # For path drawing

# --- Helper function to scan database ---
def scan_path_database(base_dir="output_db"):
    db_summary = {}
    if not os.path.exists(base_dir):
        return db_summary
    for mech_type_dir in os.listdir(base_dir):
        full_mech_type_path = os.path.join(base_dir, mech_type_dir)
        if os.path.isdir(full_mech_type_path):
            variations = [f for f in os.listdir(full_mech_type_path) if f.startswith("variation_") and f.endswith(".json")]
            db_summary[mech_type_dir] = len(variations)
    return db_summary

# --- Global Constants for Path Processing ---
NUM_RESAMPLE_POINTS = 100

# --- Path Preprocessing Function ---
from sklearn.decomposition import PCA

def preprocess_path(path_points_list, num_points_to_resample):
    path_np = np.array(path_points_list, dtype=float)
    if len(path_np) < 2: return None

    # 1. Translate path to be centered at its mean (centroid)
    centroid = np.mean(path_np, axis=0)
    centered_path = path_np - centroid

    # 2. PCA Alignment for Rotation Invariance
    aligned_path = centered_path
    if len(centered_path) >= 2: # PCA needs at least 2 samples for fitting
        try:
            pca = PCA(n_components=2)
            # Fit PCA on the centered path to find principal components
            pca.fit(centered_path)

            # Get the first principal component (the direction of most variance)
            first_principal_component = pca.components_[0]

            # Calculate the angle of this component with the positive x-axis
            angle = np.arctan2(first_principal_component[1], first_principal_component[0])

            # Create a rotation matrix to align this component with the x-axis (rotate by -angle)
            cos_rot = np.cos(-angle)
            sin_rot = np.sin(-angle)
            rotation_matrix = np.array([[cos_rot, -sin_rot],
                                        [sin_rot,  cos_rot]])

            # Apply the rotation to all points in the centered path
            aligned_path = np.dot(centered_path, rotation_matrix.T) # (N,2) @ (2,2)

        except Exception as e_pca:
            st.warning(f"PCA alignment failed: {e_pca}. Using centered path without rotation alignment.")
            # aligned_path remains centered_path if PCA fails

    # 3. Resample the (now centered and aligned) path
    if len(aligned_path) < 2: # Check again in case PCA somehow reduced points (should not happen)
        return None

    distances = np.sqrt(np.sum(np.diff(aligned_path, axis=0)**2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    if cumulative_distances[-1] == 0: return None # Path has no length (all points identical after alignment)

    t_norm = cumulative_distances / cumulative_distances[-1]

    # Ensure t_norm is monotonically increasing for interp1d
    current_path_for_interp = aligned_path
    if not np.all(np.diff(t_norm) >= 0):
        unique_t_norm, unique_indices = np.unique(t_norm, return_index=True)
        current_path_for_interp_unique = aligned_path[unique_indices]
        if len(current_path_for_interp_unique) < 2: return None

        distances = np.sqrt(np.sum(np.diff(current_path_for_interp_unique, axis=0)**2, axis=1))
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
        if cumulative_distances[-1] == 0: return None
        t_norm = cumulative_distances / cumulative_distances[-1]
        current_path_for_interp = current_path_for_interp_unique

        if not np.all(np.diff(t_norm) >= 0) or len(t_norm) < 2:
             st.warning(f"Path preprocessing: t_norm still not monotonic. Length: {len(t_norm)}")
             return None

    fx = interp1d(t_norm, current_path_for_interp[:, 0], kind='linear', fill_value="extrapolate")
    fy = interp1d(t_norm, current_path_for_interp[:, 1], kind='linear', fill_value="extrapolate")

    t_resample = np.linspace(0, 1, num_points_to_resample)
    x_resample = fx(t_resample)
    y_resample = fy(t_resample)

    resampled_aligned_path = np.vstack((x_resample, y_resample)).T

    # 4. Scale the aligned, resampled path by the maximum absolute coordinate value
    #    (This matches `normalize_curve2` from reference, which scales PCA-projected points)
    max_abs_coord = np.max(np.abs(resampled_aligned_path))
    if max_abs_coord > 1e-9: # Avoid division by zero for null paths
        final_processed_path = resampled_aligned_path / max_abs_coord
    else:
        final_processed_path = resampled_aligned_path # Path is essentially zero

    return final_processed_path


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "3-Bar Linkage", "4-Bar Linkage", "Cam Demo (Conceptual)", "Spur Gear Demo",
    "Path-to-Mechanism Search"
])

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

    # Use session state to keep track of animation status and current plot
    if 'animation_running' not in st.session_state:
        st.session_state.animation_running = False
    if 'fig_4bar' not in st.session_state:
        st.session_state.fig_4bar, st.session_state.ax_4bar = plt.subplots(figsize=(6,5))

    with col1_4bar:
        st.markdown("Define a four-bar linkage. O2 is at (0,0).")
        l_crank = st.slider("Crank Length (O2A)", 0.1, 5.0, 1.0, key="4bar_l_crank")
        l_coupler = st.slider("Coupler Length (AB)", 0.1, 5.0, 2.5, key="4bar_l_coupler")
        l_rocker = st.slider("Rocker Length (BO4)", 0.1, 5.0, 2.0, key="4bar_l_rocker")

        st.markdown("**Move Ground Pivot O4:**")
        o4x = st.number_input("O4.x", -5.0, 5.0, 3.0, 0.1, key="4bar_o4x")
        o4y = st.number_input("O4.y", -5.0, 5.0, 0.0, 0.1, key="4bar_o4y")

        # Static crank angle slider for manual control
        # We will override this with animation if running
        if not st.session_state.animation_running:
            crank_angle_deg_manual = st.slider("Input Crank Angle (degrees)", 0.0, 360.0, 30.0, key="4bar_angle_manual")
        else:
            # Keep a disabled display or hide when animating
            st.slider("Input Crank Angle (degrees)", 0.0, 360.0, 30.0, key="4bar_angle_manual_disabled", disabled=True, help="Animation in progress")
            crank_angle_deg_manual = 0 # Default during animation, actual angle set by loop

        if st.button("Run Animation", key="run_animation_4bar"):
            st.session_state.animation_running = True
            # If user clicks stop, this will be set to False

        if st.button("Stop Animation", key="stop_animation_4bar"):
            st.session_state.animation_running = False

        # Fixed points
        o2 = CommonPoint(0, 0)
        o4 = CommonPoint(o4x, o4y)
        l_ground = o2.distance_to(o4)
        st.metric("Ground Link Length (O2O4)", f"{l_ground:.3f}")

    # --- Calculation and Plotting Column ---
    with col2_4bar:
        # Placeholder for the plot that will be updated
        plot_placeholder = st.empty()

        if st.session_state.animation_running:
            import time
            num_frames = 120 # Number of frames for a full 360-degree animation
            delay_between_frames = 0.05 # seconds

            for i in range(num_frames):
                if not st.session_state.animation_running: # Check if stop was clicked
                    break

                current_crank_angle_deg = (360.0 / num_frames) * i
                crank_angle_rad = math.radians(current_crank_angle_deg)

                # Crank end point A
                ax_coord = o2.x + l_crank * math.cos(crank_angle_rad)
                ay_coord = o2.y + l_crank * math.sin(crank_angle_rad)
                a = CommonPoint(ax_coord, ay_coord)

                # Kinematic solution for point B
                b_x, b_y = o4.x - l_rocker, o4.y # Default B
                d_ao4 = a.distance_to(o4)
                if d_ao4 <= l_coupler + l_rocker and d_ao4 >= abs(l_coupler - l_rocker) and d_ao4 > 1e-6:
                    try:
                        cos_alpha_val = (l_coupler**2 + d_ao4**2 - l_rocker**2) / (2 * l_coupler * d_ao4)
                        cos_alpha_val = max(-1.0, min(1.0, cos_alpha_val))
                        alpha_at_A_in_AO4B = math.acos(cos_alpha_val)
                        theta_ao4 = math.atan2(o4.y - a.y, o4.x - a.x)
                        b_angle_from_A = theta_ao4 - alpha_at_A_in_AO4B
                        b_x = a.x + l_coupler * math.cos(b_angle_from_A)
                        b_y = a.y + l_coupler * math.sin(b_angle_from_A)
                    except (ValueError, ZeroDivisionError):
                        pass # Keep default B if math error

                b = CommonPoint(b_x, b_y)
                points_4bar = [o2, a, b, o4]
                links_4bar = [(0,1), (1,2), (2,3), (3,0)]

                plot_linkage_demo(points_4bar, links_4bar, st.session_state.ax_4bar, f"4-Bar Animation (Angle: {current_crank_angle_deg:.1f}°)")
                with plot_placeholder:
                    st.pyplot(st.session_state.fig_4bar)

                time.sleep(delay_between_frames)

            st.session_state.animation_running = False # Animation finished or stopped
            # After animation, show the final frame or revert to manual slider control
            # For now, let's ensure one final plot update with the last state or manual slider state
            # This will effectively fall through to the static plot logic below if animation_running is false

        # Static plot / plot based on manual slider (when not animating)
        if not st.session_state.animation_running:
            crank_angle_rad_manual = math.radians(crank_angle_deg_manual)
            ax_manual = o2.x + l_crank * math.cos(crank_angle_rad_manual)
            ay_manual = o2.y + l_crank * math.sin(crank_angle_rad_manual)
            a_manual = CommonPoint(ax_manual, ay_manual)

            b_x_manual, b_y_manual = o4.x - l_rocker, o4.y # Default B
            d_ao4_manual = a_manual.distance_to(o4)
            if d_ao4_manual <= l_coupler + l_rocker and d_ao4_manual >= abs(l_coupler - l_rocker) and d_ao4_manual > 1e-6:
                try:
                    cos_alpha_val_manual = (l_coupler**2 + d_ao4_manual**2 - l_rocker**2) / (2 * l_coupler * d_ao4_manual)
                    cos_alpha_val_manual = max(-1.0, min(1.0, cos_alpha_val_manual))
                    alpha_manual = math.acos(cos_alpha_val_manual)
                    theta_ao4_manual = math.atan2(o4.y - a_manual.y, o4.x - a_manual.x)
                    b_angle_manual = theta_ao4_manual - alpha_manual
                    b_x_manual = a_manual.x + l_coupler * math.cos(b_angle_manual)
                    b_y_manual = a_manual.y + l_coupler * math.sin(b_angle_manual)
                except (ValueError, ZeroDivisionError):
                    pass # Keep default B

            b_manual = CommonPoint(b_x_manual, b_y_manual)
            points_4bar_manual = [o2, a_manual, b_manual, o4]
            links_4bar_manual = [(0,1), (1,2), (2,3), (3,0)]

            plot_linkage_demo(points_4bar_manual, links_4bar_manual, st.session_state.ax_4bar, "4-Bar Linkage (Crank Driven - Simplified Solve)")
            with plot_placeholder:
                st.pyplot(st.session_state.fig_4bar)

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
        gear_name = st.text_input("Gear Name", "DemoGear", key="gear_name_input_v2") # Changed key

        st.markdown("**Move Gear Center:**")
        gear_center_x = st.number_input("Center X", -5.0, 5.0, 0.0, 0.1, key="gear_center_x")
        gear_center_y = st.number_input("Center Y", -5.0, 5.0, 0.0, 0.1, key="gear_center_y")
        gear_rotation_deg = st.slider("Rotation (degrees)", 0.0, 360.0, 0.0, 1.0, key="gear_rotation")

        param_type = st.radio("Define by:", ("Module", "Diametral Pitch"), horizontal=True, key="gear_param_type_v2") # Changed key

        module_val, dp_val = None, None
        if param_type == "Module":
            module_val = st.number_input("Module", min_value=0.1, value=2.0, step=0.1, key="gear_module_v2") # Changed key
        else:
            dp_val = st.number_input("Diametral Pitch", min_value=0.1, value=10.0, step=0.1, key="gear_dp_v2") # Changed key

        num_teeth_val = st.slider("Number of Teeth", 5, 200, 30, 1, key="gear_teeth_v2") # Changed key
        pressure_angle_val = st.slider("Pressure Angle (deg)", 10.0, 30.0, 20.0, 0.5, key="gear_pa_v2") # Changed key

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
            plot_gear_demo(current_gear_demo, ax_gear_demo, gear_center_x, gear_center_y, gear_rotation_deg)
            st.pyplot(fig_gear_demo)
        else:
            st.warning("Gear not defined due to parameter error.")

st.sidebar.info("Macanism Project Demo")

# --- Path-to-Mechanism Search Tab (Phase 2) ---
with tab5:
    st.header("Path-to-Mechanism Search")
    st.markdown("""
    **Objective:** Draw a desired output path, and the system will search its database
    for mechanisms that can generate a similar path.
    """)

    col1_search, col2_search = st.columns([1, 2])

    with col1_search:
        st.subheader("1. Database Status")
        db_summary = scan_path_database()
        if db_summary:
            st.write("Available mechanism types in database:")
            for mech_type, count in db_summary.items():
                st.write(f"- **{mech_type}**: {count} variations")
        else:
            st.warning("Mechanism path database ('output_db') not found or empty. Please generate it first.")

        st.subheader("2. Draw Your Desired Path")

        # Simplified Drawing canvas setup for testing
        st.markdown("--- DEBUG: Simplified Canvas ---")
        canvas_result = st_canvas(
            stroke_width=3,
            stroke_color="#000000",
            background_color="#EEEEEE",
            width=400,
            height=300,
            drawing_mode="freedraw",
            key="path_canvas_debug", # Changed key to ensure it's a new instance
        )
        st.markdown("--- END DEBUG ---")

        # Keep original canvas code commented out for now
        # stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3, key="stroke_width_path_draw")
        # stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#000", key="stroke_color_path_draw")
        # bg_color = st.sidebar.color_picker("Background color hex: ", "#eee", key="bg_color_path_draw")
        # drawing_mode = "freedraw"
        # realtime_update = st.sidebar.checkbox("Update in realtime", True, key="realtime_update_path_draw")
        # canvas_width = 400
        # canvas_height = 300
        # canvas_result_original = st_canvas(
        #     fill_color="rgba(255, 165, 0, 0.3)",
        #     stroke_width=stroke_width,
        #     stroke_color=stroke_color,
        #     background_color=bg_color,
        #     update_streamlit=realtime_update,
        #     width=canvas_width,
        #     height=canvas_height,
        #     drawing_mode=drawing_mode,
        #     key="path_canvas_original",
        # )


        if st.button("Process Drawn Path"):
            st.session_state.user_drawn_path = []
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data.get("objects", [])
                if objects and objects[0]["type"] == "path": # Assuming the first drawn object is the path
                    # Path data is a list of segments like ['M', x, y, 'L', x, y, ...] or ['Q', cx, cy, x, y, ...]
                    # For 'freedraw', it's typically a single "path" object with many segments.
                    # We need to extract the (x,y) coordinates from these segments.
                    path_segments = objects[0].get("path", [])
                    current_path_points = []
                    for i in range(len(path_segments)):
                        segment = path_segments[i]
                        if segment[0] in ['M', 'L']: # MoveTo or LineTo
                            current_path_points.append((segment[1], segment[2]))
                        elif segment[0] == 'Q': # Quadratic Bezier
                            # For simplicity, just take the end point of Q. Could interpolate if needed.
                            current_path_points.append((segment[3], segment[4]))
                        elif segment[0] == 'C': # Cubic Bezier
                            current_path_points.append((segment[5], segment[6]))

                    if current_path_points:
                        # Normalize Y to be positive upwards, canvas Y is often downwards
                        # And scale to a more mechanism-like coordinate system if canvas pixels are too large
                        # This normalization/scaling step is crucial and might need adjustment.
                        # For now, let's assume canvas coordinates are roughly usable.
                        # A simple Y-flip if canvas origin is top-left:
                        # normalized_path = [(x, canvas_height - y) for x, y in current_path_points]
                        # st.session_state.user_drawn_path = normalized_path
                        st.session_state.user_drawn_path = current_path_points # Use raw for now
                        st.success(f"Path with {len(st.session_state.user_drawn_path)} points loaded from canvas.")
                    else:
                        st.warning("No path data found on canvas or path is empty.")
                else:
                    st.warning("No path drawn or unrecognized drawing object. Please use 'freedraw'.")
            else:
                st.warning("Canvas is empty. Please draw a path.")

        # Display number of points if path exists
        if 'user_drawn_path' in st.session_state and st.session_state.user_drawn_path:
            st.write(f"Number of points in current path: {len(st.session_state.user_drawn_path)}")


        st.subheader("3. Search Parameters")
        num_matches_to_show = st.slider("Number of top matches to display", 1, 10, 3)

        if st.button("Search for Matching Mechanisms"):
            if 'user_drawn_path' in st.session_state and st.session_state.user_drawn_path:
                user_path = np.array(st.session_state.user_drawn_path)
                if len(user_path) < 2:
                    st.error("Drawn path must have at least 2 points.")
                else:
                    st.write("Searching...")

                    # 1. Load database paths
                    db_files = glob.glob(os.path.join("output_db", "*", "*.json"))
                    all_db_mechanisms = []
                    for f_path in db_files:
                        try:
                            with open(f_path, 'r') as f:
                                data = json.load(f)
                                all_db_mechanisms.append(data)
                        except Exception as e:
                            st.error(f"Error loading {f_path}: {e}")

                    if not all_db_mechanisms:
                        st.warning("No mechanism data found in database.")
                        st.session_state.search_results = []
                    else:
                        # 2. Preprocess paths and calculate similarity
                        num_resample_points = 100 # Standard number of points for comparison

                        # preprocess_path function is now global
                        processed_user_path = preprocess_path(user_path.tolist(), num_resample_points)

                        if processed_user_path is None:
                            st.error("Could not process user path (e.g., too short or no length).")
                            st.session_state.search_results = []
                        else:
                            results = []
                            for db_mech_data in all_db_mechanisms:
                                db_path_points = db_mech_data.get("path_xy_coordinates")
                                if not db_path_points: continue

                                processed_db_path = preprocess_path(db_path_points, num_resample_points)
                                if processed_db_path is None: continue

                                # Calculate similarity (e.g., Hausdorff distance)
                                # Lower is better. directed_hausdorff returns (dist, idx1, idx2)
                                similarity_score_u_to_db = directed_hausdorff(processed_user_path, processed_db_path)[0]
                                similarity_score_db_to_u = directed_hausdorff(processed_db_path, processed_user_path)[0]
                                # Use max of the two for a symmetric measure
                                score = max(similarity_score_u_to_db, similarity_score_db_to_u)

                                results.append({
                                    "score": score,
                                    "template_name": db_mech_data.get("mechanism_template_name"),
                                    "variation_id": db_mech_data.get("variation_id"),
                                    "parameters": db_mech_data.get("parameters"),
                                    "db_path_original_len": len(db_path_points),
                                    "db_path_processed": processed_db_path.tolist() # For potential plotting
                                })

                            results.sort(key=lambda r: r["score"])
                            st.session_state.search_results = results[:num_matches_to_show]
                            st.success(f"Search complete. Found {len(results)} potential matches. Displaying top {num_matches_to_show}.")
            else:
                st.warning("Please draw or enter and process a path first.")

    with col2_search:
        st.subheader("Search Results & Visualization")

        if 'user_drawn_path' in st.session_state and st.session_state.user_drawn_path:
            st.markdown("**Your Drawn Path:**")
            fig_drawn_path, ax_drawn_path = plt.subplots(figsize=(6,5))
            user_path_x = [p[0] for p in st.session_state.user_drawn_path]
            user_path_y = [p[1] for p in st.session_state.user_drawn_path]
            ax_drawn_path.plot(user_path_x, user_path_y, 'g.-', label="User's Path")
            ax_drawn_path.set_title("User Drawn Path")
            ax_drawn_path.axis('equal')
            ax_drawn_path.grid(True)
            ax_drawn_path.legend()
            st.pyplot(fig_drawn_path)

        if 'search_results' in st.session_state and st.session_state.search_results:
            st.markdown("---")
            st.subheader("Top Matches:")
            for i, res in enumerate(st.session_state.search_results):
                st.markdown(f"**Match {i+1}: Score = {res['score']:.4f}**")
                st.write(f"Type: {res['template_name']}, Variation ID: {res['variation_id']}")
                with st.expander("View Parameters"):
                    st.json(res['parameters'])

                # Placeholder for mechanism visualization button/action
                if st.button(f"Visualize Match {i+1}", key=f"viz_match_{i}"):
                    st.write(f"Visualizing {res['template_name']} - {res['variation_id']} (placeholder)")
                    # Mechanism reconstruction and plotting
                    st.session_state.current_visualization_data = res # Store selected match data

                    template_name_str = res['template_name']
                    parameters = res['parameters']

                    template_functions = {
                        "FourBarLinkage_TypeA": create_four_bar_linkage_template,
                        "CrankSlider_Simple": create_crank_slider_template
                        # Add other templates here as they are created
                    }

                    if template_name_str not in template_functions:
                        st.error(f"Template function for '{template_name_str}' not found.")
                    else:
                        template_func = template_functions[template_name_str]
                        mech_definition = template_func(parameters)

                        if mech_definition is None:
                            st.error(f"Could not generate mechanism definition for {template_name_str} with params {parameters}")
                        else:
                            vectors, origin, loops_func, guess, motion, output_joint_name = mech_definition

                            try:
                                mech_instance = MechanismSolver(
                                    vectors=vectors, origin=origin, loops=loops_func,
                                    pos=motion, guess=(guess,)
                                )
                                mech_instance.iterate()

                                # Plotting
                                fig_viz, ax_viz = plt.subplots(figsize=(7,6))

                                # 1. Plot the mechanism structure (links at first position)
                                # iterate() has already populated x_positions, y_positions.
                                # We use the first entry [0] for the initial configuration plot.
                                # No need to call mech_instance.calculate() again if iterate() was successful.
                                for vec_obj in mech_instance.vectors:
                                    if not vec_obj.pos.show: continue
                                    j1, j2 = vec_obj.joints
                                    # Get first position from stored arrays
                                    x1, y1 = j1.x_positions[0], j1.y_positions[0]
                                    x2, y2 = j2.x_positions[0], j2.y_positions[0]
                                    ax_viz.plot([x1, x2], [y1, y2], color=vec_obj.pos.kwargs.get('color', 'k'),
                                                linestyle=vec_obj.pos.kwargs.get('linestyle', '-'),
                                                marker=vec_obj.pos.kwargs.get('marker', 'o'),
                                                label=f"Link {j1.name}-{j2.name}")

                                # 2. Plot the generated path of the output joint
                                output_j_obj = next((j for j in mech_instance.joints if j.name == output_joint_name), None)
                                if output_j_obj:
                                    ax_viz.plot(output_j_obj.x_positions, output_j_obj.y_positions, 'b--', lw=2, label=f"Mech Path ({output_j_obj.name})")

                                # 3. Overlay user's drawn path (processed)
                                # Use global NUM_RESAMPLE_POINTS for calls from this section
                                user_path_processed_plot = np.array(preprocess_path(st.session_state.user_drawn_path, NUM_RESAMPLE_POINTS))
                                if user_path_processed_plot is not None:
                                     # The user path was processed (translated to origin, scaled).
                                     # The mechanism path is in its own coordinate system.
                                     # For a fair comparison on the same plot, we need to:
                                     # A) Plot mechanism path as is, and try to transform user path to match its scale/pos. (Hard)
                                     # B) Process the mechanism path similarly to user path and plot both normalized. (Easier for now)

                                    mech_output_path_raw = np.array(list(zip(output_j_obj.x_positions, output_j_obj.y_positions)))
                                    mech_output_path_processed = preprocess_path(mech_output_path_raw.tolist(), NUM_RESAMPLE_POINTS)

                                    if mech_output_path_processed is not None:
                                        ax_viz.plot(mech_output_path_processed[:,0], mech_output_path_processed[:,1], 'm:', lw=2, label="Mech Path (Processed)")
                                    ax_viz.plot(user_path_processed_plot[:,0], user_path_processed_plot[:,1], 'g.-', label="User Path (Processed)")


                                ax_viz.set_title(f"Visualizing: {res['template_name']} ({res['variation_id']})\nOutput: {output_joint_name}")
                                ax_viz.axis('equal')
                                ax_viz.grid(True)
                                ax_viz.legend(fontsize='small')
                                st.pyplot(fig_viz) # Show static plot first

                                # Add animation
                                st.markdown("---")
                                st.subheader("Mechanism Animation")
                                try:
                                    # Ensure the output joint's path is followed in animation
                                    if output_j_obj:
                                        output_j_obj.follow = True
                                        # Also, ensure other joints that might have been set to follow by default
                                        # in templates are respected, or explicitly set follow for all/none.
                                        # For now, just ensuring the main output joint is followed.

                                    # Make sure all joints have their iterable arrays correctly sized for animation
                                    # This should be handled by iterate() already.

                                    # Check if pos is an array for animation frames
                                    if not isinstance(mech_instance.pos, np.ndarray) or mech_instance.pos.ndim == 0:
                                         st.warning("Mechanism 'pos' attribute is not a suitable array for animation. Skipping animation.")
                                    elif len(mech_instance.pos) <= 1:
                                        st.warning("Mechanism 'pos' array has insufficient frames for animation. Skipping animation.")
                                    else:
                                        with st.spinner("Generating animation..."):
                                            # Ensure player key_bindings are off for streamlit
                                            ani, fig_ani, ax_ani = mech_instance.get_animation(key_bindings=False)

                                            animation_path = "temp_matched_animation.mp4"
                                            # Ensure a clean figure for saving animation
                                            # fig_ani.canvas.draw_idle() # Update figure state

                                            # Check if writer is available
                                            if plt.rcParams['animation.writer'] == 'ffmpeg':
                                                ani.save(animation_path, writer='ffmpeg', fps=30, dpi=150)
                                                st.video(animation_path)
                                            else:
                                                st.warning("FFmpeg writer not available for MP4. Trying GIF format (might be slow/large).")
                                                gif_path = "temp_matched_animation.gif"
                                                try:
                                                    ani.save(gif_path, writer='pillow', fps=15, dpi=100) # Pillow for GIF
                                                    st.image(gif_path)
                                                except Exception as e_gif:
                                                    st.error(f"Could not save animation as GIF: {e_gif}")

                                            # Close the animation figure to free memory
                                            plt.close(fig_ani)


                                except Exception as e_anim:
                                    st.error(f"Error generating animation: {e_anim}")
                                    # import traceback
                                    # st.text(traceback.format_exc())


                                # Store for potential interactive modification (Phase 3)
                                st.session_state.current_mech_instance = mech_instance
                                st.session_state.current_mech_params = parameters
                                st.session_state.current_mech_template_name = template_name_str


                            except Exception as e_viz:
                                st.error(f"Error visualizing mechanism: {e_viz}")
                                # import traceback
                                # st.text(traceback.format_exc())


        elif 'search_results' in st.session_state and not st.session_state.search_results:
             st.info("No matching mechanisms found or database is empty.")
