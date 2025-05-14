import unittest
import numpy as np
from src.macanism.vectors import VectorBase, Vector, Position, Velocity, Acceleration

# Mock Joint class for testing VectorBase methods that interact with joints
class MockJoint:
    def __init__(self, name=""):
        self.name = name
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.ax = 0.0
        self.ay = 0.0
        self.az = 0.0

    def _fix_position(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z

    def _fix_velocity(self, vx, vy, vz=0.0):
        self.vx, self.vy, self.vz = vx, vy, vz

    def _fix_acceleration(self, ax, ay, az=0.0):
        self.ax, self.ay, self.az = ax, ay, az

    def __repr__(self):
        return self.name


class TestVectorBase(unittest.TestCase):
    def test_initialization_xyz(self):
        v = VectorBase(x=1, y=2, z=3)
        self.assertEqual(v.x, 1)
        self.assertEqual(v.y, 2)
        self.assertEqual(v.z, 3)

    def test_initialization_r_theta_z(self):
        v = VectorBase(r=5, theta=np.pi/2, z=1) # Should point along y-axis in XY plane
        # self.get() selection logic
        self.assertTrue(callable(v.get))
        # Values are not directly converted to x,y in __init__ unless x,y are None and r,theta are not.
        # This test primarily checks if z is stored.
        self.assertEqual(v.z, 1)
        # x,y are None initially if r,theta are given
        self.assertIsNone(v.x)
        self.assertIsNone(v.y)


    def test_reverse(self):
        j1 = MockJoint("J1")
        j2 = MockJoint("J2")
        v = VectorBase(joints=(j1, j2), x=1, y=2, z=3)
        v_rev = v._reverse()
        self.assertEqual(v_rev.x, -1)
        self.assertEqual(v_rev.y, -2)
        self.assertEqual(v_rev.z, -3)
        self.assertEqual(v_rev.joints[0], j2)
        self.assertEqual(v_rev.joints[1], j1)

    def test_get_mag(self):
        v = VectorBase(x=3, y=4, z=0)
        mag, angle = v._get_mag()
        self.assertAlmostEqual(mag, 5.0)
        self.assertAlmostEqual(angle, np.arctan2(4,3))

        v_3d = VectorBase(x=1, y=2, z=2) # Mag = sqrt(1+4+4) = 3
        mag_3d, angle_3d_xy = v_3d._get_mag()
        self.assertAlmostEqual(mag_3d, 3.0)
        self.assertAlmostEqual(angle_3d_xy, np.arctan2(2,1))

    def test_add(self):
        j1, j2, j3 = MockJoint("J1"), MockJoint("J2"), MockJoint("J3")
        v1 = VectorBase(joints=(j1,j2), x=1, y=2, z=3)
        v2 = VectorBase(joints=(j2,j3), x=4, y=5, z=6)
        v_sum = v1 + v2
        self.assertEqual(v_sum.x, 5)
        self.assertEqual(v_sum.y, 7)
        self.assertEqual(v_sum.z, 9)
        self.assertEqual(v_sum.joints[0], j1)
        self.assertEqual(v_sum.joints[1], j3)


    def test_sub(self):
        j1, j2, j3 = MockJoint("J1"), MockJoint("J2"), MockJoint("J3")
        v1 = VectorBase(joints=(j1,j2), x=1, y=2, z=3)
        v2 = VectorBase(joints=(j2,j3), x=4, y=5, z=6) # Note: joints for subtraction might be conceptually different
        v_diff = v1 - v2
        self.assertEqual(v_diff.x, -3)
        self.assertEqual(v_diff.y, -3)
        self.assertEqual(v_diff.z, -3)
        self.assertEqual(v_diff.joints[0], j1)
        self.assertEqual(v_diff.joints[1], j3)


    def test_dot_product(self):
        v1 = VectorBase(x=1, y=2, z=3)
        v2 = VectorBase(x=4, y=-5, z=6)
        # 1*4 + 2*(-5) + 3*6 = 4 - 10 + 18 = 12
        self.assertEqual(v1.dot(v2), 12)

    def test_cross_product(self):
        v1 = VectorBase(x=1, y=0, z=0) # i
        v2 = VectorBase(x=0, y=1, z=0) # j
        v_cross = v1.cross(v2) # i x j = k
        self.assertEqual(v_cross.x, 0)
        self.assertEqual(v_cross.y, 0)
        self.assertEqual(v_cross.z, 1)

        v3 = VectorBase(x=1, y=2, z=3)
        v4 = VectorBase(x=4, y=5, z=6)
        # (2*6 - 3*5, 3*4 - 1*6, 1*5 - 2*4)
        # (12 - 15, 12 - 6, 5 - 8)
        # (-3, 6, -3)
        v_cross_2 = v3.cross(v4)
        self.assertEqual(v_cross_2.x, -3)
        self.assertEqual(v_cross_2.y, 6)
        self.assertEqual(v_cross_2.z, -3)

    def test_fix_global_position(self):
        j1 = MockJoint("J1")
        j2 = MockJoint("J2")
        v = VectorBase(joints=(j1, j2), x=10, y=20, z=30)
        v._fix_global_position()
        self.assertEqual(j2.x, 10)
        self.assertEqual(j2.y, 20)
        self.assertEqual(j2.z, 30)

    def test_fix_global_velocity(self):
        j1 = MockJoint("J1")
        j2 = MockJoint("J2")
        # For Velocity, x,y,z of VectorBase instance represent vx,vy,vz
        v = VectorBase(joints=(j1, j2), x=1, y=2, z=3) # vx, vy, vz
        v._fix_global_velocity()
        self.assertEqual(j2.vx, 1)
        self.assertEqual(j2.vy, 2)
        self.assertEqual(j2.vz, 3)

    def test_fix_global_acceleration(self):
        j1 = MockJoint("J1")
        j2 = MockJoint("J2")
        # For Acceleration, x,y,z of VectorBase instance represent ax,ay,az
        v = VectorBase(joints=(j1, j2), x=4, y=5, z=6) # ax, ay, az
        v._fix_global_acceleration()
        self.assertEqual(j2.ax, 4)
        self.assertEqual(j2.ay, 5)
        self.assertEqual(j2.az, 6)


class TestPosition(unittest.TestCase):
    def test_init_xyz(self):
        p = Position(x=1, y=2, z=3)
        np.testing.assert_array_almost_equal(p._neither(), [1, 2, 3])

    def test_init_r_theta_z(self):
        p = Position(r=1, theta=np.pi/2, z=5) # x=0, y=1, z=5
        np.testing.assert_array_almost_equal(p._neither(), [0, 1, 5])

    def test_both_method(self):
        p = Position(x=0,y=0,z=0)
        res = p._both(r_val=2, theta_val=np.pi, z_val=10) # x=-2, y=0, z=10
        np.testing.assert_array_almost_equal(res, [-2, 0, 10])
        self.assertEqual(p.x, -2)
        self.assertAlmostEqual(p.y, 0) # Use assertAlmostEqual for floating point
        self.assertEqual(p.z, 10)

    def test_tangent_method(self):
        p = Position(r=3, z=1) # r is fixed
        res = p._tangent(theta_val=np.pi/2, z_val=5) # x=0, y=3, z=5
        np.testing.assert_array_almost_equal(res, [0, 3, 5])
        self.assertAlmostEqual(p.x, 0) # Use assertAlmostEqual
        self.assertEqual(p.y, 3)
        self.assertEqual(p.z, 5)

    def test_slip_method(self):
        p = Position(theta=np.pi, z=2) # theta is fixed
        res = p._slip(r_val=4, z_val=7) # x=-4, y=0, z=7
        np.testing.assert_array_almost_equal(res, [-4, 0, 7])
        self.assertEqual(p.x, -4)
        self.assertAlmostEqual(p.y, 0) # Use assertAlmostEqual
        self.assertEqual(p.z, 7)


class TestVelocity(unittest.TestCase):
    def test_init_xyz_is_vx_vy_vz(self):
        # When x,y,z are passed to Velocity, they are vx,vy,vz
        vel = Velocity(x=1, y=2, z=3)
        self.assertEqual(vel.x, 1) # vx
        self.assertEqual(vel.y, 2) # vy
        self.assertEqual(vel.z, 3) # vz
        # r, theta, and positional z are context, should be 0 or None if not set
        self.assertEqual(vel.r, 0.0)
        self.assertEqual(vel.theta, 0.0)
        # self.z in VectorBase is used for vz, positional z context is also self.z for Velocity
        # This is a bit confusing, but the logic in _both, _tangent, _slip uses self.r, self.theta as context
        # and self.z (from init) as vz.

    def test_neither_method(self):
        # Context for velocity calculation (r, theta from position)
        vel = Velocity(r=1, theta=np.pi/2) # Positional context
        vel.r_dot, vel.omega, vel.z_dot = 0,0,0 # Explicitly set rates to 0
        res = vel._neither() # Should result in vx=0, vy=0, vz=0
        np.testing.assert_array_almost_equal(res, [0,0,0])
        self.assertEqual(vel.x, 0)
        self.assertEqual(vel.y, 0)
        self.assertEqual(vel.z, 0) # vz

    def test_both_method(self):
        # Positional context: r=2, theta=pi/4. Velocity context: z_pos=5
        vel = Velocity(r=2, theta=np.pi/4)
        # Inputs: r_dot=1, omega=0.5, z_dot=3 (vz)
        # x = r_dot*cos(th) - r*omega*sin(th) = 1*cos(pi/4) - 2*0.5*sin(pi/4) = cos(pi/4) - sin(pi/4) = 0
        # y = r_dot*sin(th) + r*omega*cos(th) = 1*sin(pi/4) + 2*0.5*cos(pi/4) = sin(pi/4) + cos(pi/4) = sqrt(2)
        # z = z_dot = 3
        res = vel._both(r_dot_val=1, omega_val=0.5, z_dot_val=3)
        np.testing.assert_array_almost_equal(res, [0, np.sqrt(2), 3])
        self.assertAlmostEqual(vel.x, 0)
        self.assertAlmostEqual(vel.y, np.sqrt(2))
        self.assertEqual(vel.z, 3) # vz
        self.assertEqual(vel.z_dot, 3)


    def test_tangent_method(self):
        # Positional context: r=3, theta=pi. Velocity context: z_pos=1
        vel = Velocity(r=3, theta=np.pi)
        # Inputs: omega=2, z_dot=4 (vz)
        # r_dot is 0
        # x = -r*omega*sin(th) = -3*2*sin(pi) = 0
        # y =  r*omega*cos(th) =  3*2*cos(pi) = -6
        # z = z_dot = 4
        res = vel._tangent(omega_val=2, z_dot_val=4)
        np.testing.assert_array_almost_equal(res, [0, -6, 4])
        self.assertAlmostEqual(vel.x, 0)
        self.assertAlmostEqual(vel.y, -6)
        self.assertEqual(vel.z, 4) # vz
        self.assertEqual(vel.z_dot, 4)

    def test_slip_method(self):
        # Positional context: r=?, theta=pi/2. Velocity context: z_pos=2
        vel = Velocity(theta=np.pi/2)
        # Inputs: r_dot=5, z_dot=6 (vz)
        # omega is 0
        # x = r_dot*cos(th) = 5*cos(pi/2) = 0
        # y = r_dot*sin(th) = 5*sin(pi/2) = 5
        # z = z_dot = 6
        res = vel._slip(r_dot_val=5, z_dot_val=6)
        np.testing.assert_array_almost_equal(res, [0, 5, 6])
        self.assertAlmostEqual(vel.x, 0)
        self.assertAlmostEqual(vel.y, 5)
        self.assertEqual(vel.z, 6) # vz
        self.assertEqual(vel.z_dot, 6)


class TestAcceleration(unittest.TestCase):
    def test_init_xyz_is_ax_ay_az(self):
        # When x,y,z are passed to Acceleration, they are ax,ay,az
        acc = Acceleration(x=1, y=2, z=3)
        self.assertEqual(acc.x, 1) # ax
        self.assertEqual(acc.y, 2) # ay
        self.assertEqual(acc.z, 3) # az
        # Context attributes
        self.assertEqual(acc.r, 0.0)
        self.assertEqual(acc.theta, 0.0)
        self.assertEqual(acc.r_dot, 0.0)
        self.assertEqual(acc.omega, 0.0)
        self.assertEqual(acc.z_dot, 0.0)


    def test_neither_method(self):
        # Positional context: r=1, theta=pi/2. Velocity context: r_dot=0, omega=0, z_dot=0
        acc = Acceleration(r=1, theta=np.pi/2, r_dot=0, omega=0, z_dot=0)
        # Inputs: r_ddot=0, alpha=0, z_ddot=0
        res = acc._neither() # ax=0, ay=0, az=0
        np.testing.assert_array_almost_equal(res, [0,0,0])
        self.assertEqual(acc.x, 0) # ax
        self.assertEqual(acc.y, 0) # ay
        self.assertEqual(acc.z, 0) # az

    def test_both_method(self):
        # Context: r=1, theta=0, r_dot=1, omega=1, z_dot=1 (vz)
        acc = Acceleration(r=1, theta=0, r_dot=1, omega=1, z_dot=1)
        # Inputs: r_ddot=1, alpha=1, z_ddot=1 (az)
        # x = (r_ddot*cos(th) - 2*r_dot*omega*sin(th) - r*alpha*sin(th) - r*omega^2*cos(th))
        #   = (1*cos(0) - 2*1*1*sin(0) - 1*1*sin(0) - 1*1^2*cos(0))
        #   = (1*1 - 0 - 0 - 1*1*1) = 1 - 1 = 0
        # y = (r_ddot*sin(th) + 2*r_dot*omega*cos(th) + r*alpha*cos(th) - r*omega^2*sin(th))
        #   = (1*sin(0) + 2*1*1*cos(0) + 1*1*cos(0) - 1*1^2*sin(0))
        #   = (0 + 2*1*1*1 + 1*1*1 - 0) = 2 + 1 = 3
        # z = z_ddot = 1
        res = acc._both(r_ddot_val=1, alpha_val=1, z_ddot_val=1)
        np.testing.assert_array_almost_equal(res, [0, 3, 1])
        self.assertAlmostEqual(acc.x, 0) # ax
        self.assertAlmostEqual(acc.y, 3) # ay
        self.assertEqual(acc.z, 1) # az
        self.assertEqual(acc.z_ddot, 1)


    def test_tangent_method(self):
        # Context: r=2, theta=pi/2, r_dot=0 (implicit for tangent), omega=1, z_dot=1 (vz)
        acc = Acceleration(r=2, theta=np.pi/2, r_dot=0, omega=1, z_dot=1)
        # Inputs: alpha=1, z_ddot=2 (az)
        # r_ddot is 0
        # x = (r*alpha*-sin(th) - r*omega^2*cos(th))
        #   = (2*1*-sin(pi/2) - 2*1^2*cos(pi/2))
        #   = (2*1*-1 - 2*1*0) = -2
        # y = (r*alpha*cos(th) - r*omega^2*sin(th))
        #   = (2*1*cos(pi/2) - 2*1^2*sin(pi/2))
        #   = (2*1*0 - 2*1*1) = -2
        # z = z_ddot = 2
        res = acc._tangent(alpha_val=1, z_ddot_val=2)
        np.testing.assert_array_almost_equal(res, [-2, -2, 2])
        self.assertAlmostEqual(acc.x, -2)
        self.assertAlmostEqual(acc.y, -2)
        self.assertEqual(acc.z, 2)
        self.assertEqual(acc.z_ddot, 2)


    def test_slip_method(self):
        # Context: r=?, theta=0, r_dot=?, omega=0 (implicit for slip), z_dot=1 (vz)
        acc = Acceleration(theta=0, omega=0, z_dot=1)
        # Inputs: r_ddot=3, z_ddot=4 (az)
        # alpha is 0
        # x = r_ddot*cos(th) = 3*cos(0) = 3
        # y = r_ddot*sin(th) = 3*sin(0) = 0
        # z = z_ddot = 4
        res = acc._slip(r_ddot_val=3, z_ddot_val=4)
        np.testing.assert_array_almost_equal(res, [3, 0, 4])
        self.assertAlmostEqual(acc.x, 3)
        self.assertAlmostEqual(acc.y, 0)
        self.assertEqual(acc.z, 4)
        self.assertEqual(acc.z_ddot, 4)


class TestVectorClass(unittest.TestCase):
    def test_vector_initialization_and_components(self):
        j1, j2 = MockJoint("J1"), MockJoint("J2")
        vec = Vector(joints=(j1, j2), x=1, y=2, z=3, r=np.sqrt(1**2+2**2), theta=np.arctan2(2,1))

        # Position component
        self.assertIsInstance(vec.pos, Position)
        self.assertEqual(vec.pos.x, 1)
        self.assertEqual(vec.pos.y, 2)
        self.assertEqual(vec.pos.z, 3)
        # Check that r, theta are also set if provided for pos
        self.assertAlmostEqual(vec.pos.r, np.sqrt(5))
        self.assertAlmostEqual(vec.pos.theta, np.arctan2(2,1))


        # Velocity component - x,y,z passed to Vector are for Position.
        # Velocity's x,y,z (vx,vy,vz) would be derived or set separately.
        # Default init of Velocity from Vector will have x,y,z as None or 0 if not derived.
        # Let's test if context r, theta, z are passed.
        self.assertIsInstance(vec.vel, Velocity)
        self.assertAlmostEqual(vec.vel.r, vec.pos.r) # Context r
        self.assertAlmostEqual(vec.vel.theta, vec.pos.theta) # Context theta
        # vec.vel.z is tricky. If x,y,z were passed to Velocity's init, it would be vz.
        # Here, it's context z from position.
        self.assertEqual(vec.vel.z, vec.pos.z)


        # Acceleration component
        self.assertIsInstance(vec.acc, Acceleration)
        self.assertAlmostEqual(vec.acc.r, vec.pos.r) # Context r
        self.assertAlmostEqual(vec.acc.theta, vec.pos.theta) # Context theta
        self.assertEqual(vec.acc.z, vec.pos.z) # Context z from position

    def test_vector_zero_and_set_data(self):
        vec = Vector()
        s = 10 # size of arrays
        vec._zero(s)

        self.assertEqual(len(vec.pos.rs), s)
        self.assertEqual(len(vec.pos.thetas), s)
        self.assertEqual(len(vec.pos.zs), s)
        self.assertTrue(np.all(vec.pos.zs == 0))

        self.assertEqual(len(vec.vel.r_dots), s)
        self.assertEqual(len(vec.vel.omegas), s)
        self.assertEqual(len(vec.vel.z_dots), s)
        self.assertTrue(np.all(vec.vel.z_dots == 0))


        self.assertEqual(len(vec.acc.r_ddots), s)
        self.assertEqual(len(vec.acc.alphas), s)
        self.assertEqual(len(vec.acc.z_ddots), s)
        self.assertTrue(np.all(vec.acc.z_ddots == 0))

        # Set some data
        vec.pos.r, vec.pos.theta, vec.pos.z = 1, 0.5, 10
        vec._set_position_data(0)
        self.assertEqual(vec.pos.rs[0], 1)
        self.assertEqual(vec.pos.thetas[0], 0.5)
        self.assertEqual(vec.pos.zs[0], 10)

        vec.vel.r_dot, vec.vel.omega, vec.vel.z_dot = 0.1, 0.2, 0.3
        vec._set_velocity_data(0)
        self.assertEqual(vec.vel.r_dots[0], 0.1)
        self.assertEqual(vec.vel.omegas[0], 0.2)
        self.assertEqual(vec.vel.z_dots[0], 0.3)

        vec.acc.r_ddot, vec.acc.alpha, vec.acc.z_ddot = 0.01, 0.02, 0.03
        vec._set_acceleration_data(0)
        self.assertEqual(vec.acc.r_ddots[0], 0.01)
        self.assertEqual(vec.acc.alphas[0], 0.02)
        self.assertEqual(vec.acc.z_ddots[0], 0.03)

    def test_update_velocity_and_acceleration(self):
        vec = Vector()
        vec.pos.r, vec.pos.theta, vec.pos.z = 1, np.pi, 5
        vec._update_velocity()
        self.assertEqual(vec.vel.r, 1)
        self.assertEqual(vec.vel.theta, np.pi)
        self.assertEqual(vec.vel.z, 5) # z from position for context

        vec.vel.r_dot, vec.vel.omega, vec.vel.z_dot = 0.1, 0.2, 0.3 # These are actual rates
        vec._update_acceleration()
        self.assertEqual(vec.acc.r, vec.vel.r)
        self.assertEqual(vec.acc.theta, vec.vel.theta)
        self.assertEqual(vec.acc.z, vec.vel.z) # z from velocity (which was z from pos) for context
        self.assertEqual(vec.acc.r_dot, 0.1)
        self.assertEqual(vec.acc.omega, 0.2)
        self.assertEqual(vec.acc.z_dot, 0.3) # z_dot from velocity for context


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
