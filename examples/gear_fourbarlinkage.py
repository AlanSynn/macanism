# Four-bar linkage representation of gears
from macanism import *
import numpy as np
import matplotlib.pyplot as plt

# Define gear inputs
pd = 8  # Diametral pitch
N_gear = 45  # Number of teeth on gear
N_pinion = 15  # Number of teeth on pinion

m = N_gear/N_pinion  # gear ratio (by definition, gear is always bigger than the pinion)

# Define joints
O, A, B, C = get_joints('O A B C')
A.follow, B.follow = True, True

# Define vectors and loop equation
a = Vector((O, A), r=N_pinion/(pd*2))
b = Vector((A, B), style='dotted')
c = Vector((C, B), r=N_gear/(pd*2))
d = Vector((O, C), r=N_gear/(pd*2) + N_pinion/(pd*2), theta=0, show=False)


def loops(x, inp):
    temp = a(inp) + b(x[0], x[1]) - c(x[2]) - d()  # equations 1 and 2
    temp = np.concatenate((temp, [a.pos.theta + m*c.pos.theta]))  # equation 3 appended
    return temp


# Make array of known inputs
thetas = np.linspace(0, m*2*np.pi, 300)  # angles for pinion

# Guess values
pos_guess = np.array([N_gear/pd, 0, 0])

mech = macanism(vectors=(a, b, c, d), origin=O, loops=loops, pos=thetas, guess=(pos_guess, ))
mech.iterate()

ani, fig, ax = mech.get_animation()
ax.text(0, N_gear/(2*pd), f'Gear Ratio: {m:.0f}', backgroundcolor='white', ha='center', va='center')
ax.set_title('Four-bar Linkage Representation for Gears')
# ani.save('../animations/gear_fourbarlinkage.mp4', dpi=240)
plt.show()
