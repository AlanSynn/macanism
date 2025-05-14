import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.macanism import *
import numpy as np
import matplotlib.pyplot as plt

# Joint.follow_all = True  # You can trace the path of all joints by uncommenting this

O, A, B, C, D = get_joints('O A B C D')
D.follow = True
B.follow = True

a = Vector((O, A), r=5)
# b = Vector((O, C), r=8, theta=np.deg2rad(90), style='ground')  # Use this vector for a cusp output
b = Vector((O, C), r=8, theta=0, style='ground')
c = Vector((A, B), r=8)
d = Vector((C, B), r=9)
e = Vector((A, D), r=4)
f = Vector((O, D), show=False)


def loops(x, inp):
    # Note: The way this is structured with the second loop equation is only appropriate for position analysis.
    temp = np.zeros((2, 2))

    # Each vector's .get() method (called via __call__) returns an array [x, y, z].
    # For these 2D loop equations, we only need the x and y components.
    vec_a_xy = a(inp)[:2]
    vec_b_xy = b()[:2] # b has fixed r, theta, so b.get is _neither()
    vec_c_xy = c(x[0])[:2] # c has fixed r, theta varies (x[0] is theta for c), c.get is _tangent(theta)
    vec_d_xy = d(x[1])[:2] # d has fixed r, theta varies (x[1] is theta for d), d.get is _tangent(theta)
    # e has fixed r, theta varies (x[0]+deg2rad(30) is theta for e), e.get is _tangent(theta)
    vec_e_xy = e(x[0] + np.deg2rad(30))[:2]
    # f has r and theta varying (x[2] is r for f, x[3] is theta for f), f.get is _both(r, theta)
    vec_f_xy = f(x[2], x[3])[:2]

    temp[0] = vec_a_xy + vec_c_xy - vec_d_xy - vec_b_xy
    temp[1] = vec_a_xy + vec_e_xy - vec_f_xy
    return temp.flatten()


t2 = np.linspace(0, 6*np.pi, 300)
guess = np.concatenate((np.deg2rad([50, 120]), np.array([5]), np.deg2rad([50])))
macanism = macanism(vectors=(a, b, c, d, e, f), origin=O, pos=t2, guess=(guess, ),
                      loops=loops)

macanism.iterate()
ani, fig, ax = macanism.get_animation()

ax.set_title('Four Bar Linkage')

# --- Save path data using the new method ---
output_dir = "output_path_data" # Define an output directory

# Save in CSV format
macanism.save_paths(directory=output_dir,
                    base_filename="fourbar_paths",
                    mechanism_name="FourBarLinkage",
                    variation_name="default_params",
                    format="csv")

# Save in JSON format
macanism.save_paths(directory=output_dir,
                    base_filename="fourbar_paths",
                    mechanism_name="FourBarLinkage",
                    variation_name="default_params",
                    format="json")

# Save in NPY format
macanism.save_paths(directory=output_dir,
                    base_filename="fourbar_paths",
                    mechanism_name="FourBarLinkage",
                    variation_name="default_params",
                    format="npy")

print(f"\nPath data saved in various formats to the '{output_dir}' directory.")
# --- End of save path data ---

plt.show()

# ani.save('../animations/fourbarlinkage.mp4', dpi=300)
