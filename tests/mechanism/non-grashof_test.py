# Should run with no errors to get the same answer as in the homework. See engine_test.pdf.
from macanism import Vector, get_joints, macanism
import numpy as np
import matplotlib.pyplot as plt

O2, O4, O6, A, B, C, D, E, F, G = get_joints('O2 O4 O6 A B C D E F G')
a = Vector((O4, B), r=2.5)
b = Vector((B, A), r=8.4)
c = Vector((O4, O2), r=12.5, theta=0, style='ground')
d = Vector((O2, A), r=5)
e = Vector((C, A), r=2.4, show=False)
f = Vector((C, E), r=8.9)
g = Vector((O6, E), r=3.2)
h = Vector((O2, O6), r=10.5, theta=np.pi/2, style='ground')
i = Vector((D, E), r=3, show=False)
j = Vector((D, F), r=6.4)
k = Vector((G, F), theta=np.deg2rad(150), style='dotted')
l = Vector((O6, G), r=1.2, theta=np.pi/2, style='dotted')

guess1 = np.concatenate((np.deg2rad([120, 20, 70, 170, 120]), np.array([7])))
guess2 = np.array([15, 15, 30, 12, 30, 3])
guess3 = np.array([10, 10, 30, -30, 20, 10])


def loops(x, inp):
    temp = np.zeros((3, 2))
    temp[0] = a(inp) + b(x[1]) - d(x[0]) - c()
    temp[1] = f(x[2]) - g(x[3]) - h() + d(x[0]) - e(x[1])
    temp[2] = j(x[4]) - k(x[5]) - l() + g(x[3]) - i(x[2])
    return temp.flatten()


macanism = macanism(vectors=(a, b, c, d, e, f, g, h, i, j, k, l), origin=O4, loops=loops,
                      pos=np.deg2rad(52.92024014972946), vel=-30, acc=0, guess=(guess1, guess2, guess3))

macanism.calculate()
# macanism.tables(acceleration=True, velocity=True, position=True)
fig1, ax1 = macanism.plot(cushion=2, show_joints=True)
fig2, ax2 = macanism.plot(cushion=2, velocity=True, acceleration=True)
ax2.set_title('Showing Velocity and Acceleration')

assert f'{abs(k.vel.r_dot):.5f}' == '6.39467'
assert f'{abs(k.acc.r_ddot):.5f}' == '2121.04337'

plt.show()
