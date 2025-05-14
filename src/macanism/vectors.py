import numpy as np
import json
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APPEARANCE = os.path.join(THIS_DIR, 'appearance.json')


class VectorBase:
    def __init__(self, joints=None, r=None, theta=None, x=None, y=None, z=None, show=True, style=None, **kwargs):
        """
        :param joints: tup; A tuple of Joint objects. The first Joint is the tail, and the second is the head.
        :param r: int, float; The length of the vector (typically in the XY plane or total magnitude if z is also derived).
            If specified, r_dot and r_ddot might be zero.
        :param theta: int, float; The angle of the vector in radians from the positive x-axis (typically in the XY plane).
            If specified, omega and alpha might be zero.
        :param x: int, float; The value of the x component of the vector.
        :param y: int, float; The value of the y component of the vector.
        :param z: int, float; The value of the z component of the vector. Defaults to 0 if not provided.
        :param show: bool; If True, then the vector will be present in plots and animations.
        :param style: str; Applies a certain style passed to plt.plot().
            Options:
                ground - a dashed black line for grounded link
                dotted - a black dotted line
        :param kwargs: Extra arguments that are passed to plt.plot(). If not specified, the line will be maroon with a
            marker style = 'o'

        Instance Variables
        ------------------
        rs: An ndarray of r values.
        thetas: An ndarray of theta values.
        xs: An ndarray of x component values. (Potentially for future use if Cartesian history is stored)
        ys: An ndarray of y component values. (Potentially for future use if Cartesian history is stored)
        zs: An ndarray of z component values.
        r_dots: An ndarray of r_dot values.
        omegas: An ndarray of omega values.
        z_dots: An ndarray of z_dot values (dz/dt).
        r_ddots: An ndarray of r_ddot values.
        alphas: An ndarray of alpha values.
        z_ddots: An ndarray of z_ddot values (d^2z/dt^2).
        get: A function that returns the x, y, and z components of the vector. The arguments for this function depend
            on what is specified at the initialization of the object.
        """
        self.joints, self.r, self.theta, self.show = joints, r, theta, show
        self.x, self.y, self.z = x, y, (z if z is not None else 0.0)

        with open(APPEARANCE, 'r') as f:
            appearance = json.load(f)

        if style:
            self.kwargs = appearance['macanism_plot'][style]
        elif kwargs:
            self.kwargs = kwargs
        else:
            self.kwargs = appearance['macanism_plot']['default']

        # Initialize derivative attributes from kwargs, defaulting to None if not provided.
        # These might be overwritten by specific logic below or used by subclasses.
        self.r_dot = kwargs.get('r_dot', None)
        self.omega = kwargs.get('omega', None)
        self.z_dot = kwargs.get('z_dot', None)
        self.r_ddot = kwargs.get('r_ddot', None)
        self.alpha = kwargs.get('alpha', None)
        self.z_ddot = kwargs.get('z_ddot', None)

        # Determination of how get() works based on fixed/variable r, theta (primarily for 2D kinematics)
        # For 3D, x,y,z are primary. r, theta might define XY projection.
        # z_dot and z_ddot are assumed 0 if r,theta are fixed, unless specified otherwise.
        if self.r is not None and self.theta is not None: # r, theta fixed
            self.r_dot = 0.0 if self.r_dot is None else self.r_dot # Keep if provided, else 0
            self.omega = 0.0 if self.omega is None else self.omega
            self.r_ddot = 0.0 if self.r_ddot is None else self.r_ddot
            self.alpha = 0.0 if self.alpha is None else self.alpha
            self.z_dot = 0.0 if self.z_dot is None else self.z_dot
            self.z_ddot = 0.0 if self.z_ddot is None else self.z_ddot
            self.get = self._neither
        elif self.r is not None and self.theta is None: # r fixed, theta varies
            self.r_dot = 0.0 if self.r_dot is None else self.r_dot
            self.r_ddot = 0.0 if self.r_ddot is None else self.r_ddot
            self.z_dot = 0.0 if self.z_dot is None else self.z_dot
            self.z_ddot = 0.0 if self.z_ddot is None else self.z_ddot
            # self.omega will be input (or remains None if not provided via kwargs)
            self.get = self._tangent
        elif self.r is None and self.theta is not None: # theta fixed, r varies
            self.omega = 0.0 if self.omega is None else self.omega
            self.alpha = 0.0 if self.alpha is None else self.alpha
            self.z_dot = 0.0 if self.z_dot is None else self.z_dot
            self.z_ddot = 0.0 if self.z_ddot is None else self.z_ddot
            # self.r_dot will be input (or remains None if not provided via kwargs)
            self.get = self._slip
        else: # r, theta, x, y, z might all be variable or defined by components
            # r_dot, omega, z_dot, etc., will be inputs if applicable (or remain None if not provided via kwargs)
            # If they are None here, subclasses are responsible for handling or setting them.
            self.get = self._both

    def _neither(self):
        pass

    def _both(self, _, __):
        pass

    def _slip(self, _):
        pass

    def _tangent(self, _):
        pass

    def _fix_global_position(self):
        """
        Fixes the position of the head joint by making its position the x and y components of the current instance.
        The Z component is ignored for the 2D Joint class in mechanism.py.
        """
        # Joint class in mechanism.py is 2D, so only pass x and y.
        _x = self.x if self.x is not None else 0.0
        _y = self.y if self.y is not None else 0.0
        self.joints[1]._fix_position(_x, _y)

    def _fix_global_velocity(self):
        """
        Fixes the velocity of the head joint by making its velocity the x, y components of the current instance.
        The Z component is ignored for the 2D Joint class in mechanism.py.
        """
        # Joint class in mechanism.py is 2D, so only pass x and y velocity components.
        _vx = self.x if self.x is not None else 0.0 # self.x here is vx for a Velocity object
        _vy = self.y if self.y is not None else 0.0 # self.y here is vy
        self.joints[1]._fix_velocity(_vx, _vy)

    def _fix_global_acceleration(self):
        """
        Fixes the acceleration of the head joint by making its acceleration the x, y components of the current
        instance. The Z component is ignored for the 2D Joint class in mechanism.py.
        """
        # Joint class in mechanism.py is 2D, so only pass x and y acceleration components.
        _ax = self.x if self.x is not None else 0.0 # self.x here is ax for an Acceleration object
        _ay = self.y if self.y is not None else 0.0 # self.y here is ay
        self.joints[1]._fix_acceleration(_ax, _ay)

    def _reverse(self):
        """
        :return: A VectorBase object that is reversed. The joints get reversed as well as the x, y, and z components.
        """
        _x = -self.x if self.x is not None else 0.0
        _y = -self.y if self.y is not None else 0.0
        _z = -self.z if self.z is not None else 0.0
        return VectorBase(joints=(self.joints[1], self.joints[0]), x=_x, y=_y, z=_z, style=self.kwargs.get('label')) # Preserve style

    def _get_mag(self):
        """
        :return: A tuple consisting of the 3D magnitude of the current instance and the 2D angle in the XY plane.
        """
        _x = self.x if self.x is not None else 0.0
        _y = self.y if self.y is not None else 0.0
        _z = self.z if self.z is not None else 0.0
        mag = np.sqrt(_x**2 + _y**2 + _z**2)

        angle = 0.0
        if _x == 0.0 and _y == 0.0:
            angle = 0.0 # Or undefined, defaults to 0
        elif _x > 0 and _y >= 0:
            angle = np.arctan(_y/_x)
        elif _x < 0 and _y >= 0:
            angle = np.arctan(_y/_x) + np.pi
        elif _x < 0 and _y <= 0:
            angle = np.arctan(_y/_x) + np.pi
        elif _x > 0 and _y <= 0:
            angle = np.arctan(_y/_x) + 2*np.pi
        elif _x == 0 and _y > 0: # y must be non-zero here
            angle = np.pi/2
        elif _x == 0 and _y < 0: # y must be non-zero here
            angle = 3*np.pi/2

        return mag, angle

    def __add__(self, other):
        if not isinstance(other, VectorBase):
            raise TypeError("Can only add VectorBase to another VectorBase.")
        _x1 = self.x if self.x is not None else 0.0
        _y1 = self.y if self.y is not None else 0.0
        _z1 = self.z if self.z is not None else 0.0
        _x2 = other.x if other.x is not None else 0.0
        _y2 = other.y if other.y is not None else 0.0
        _z2 = other.z if other.z is not None else 0.0
        x_sum, y_sum, z_sum = _x1 + _x2, _y1 + _y2, _z1 + _z2
        return VectorBase(joints=(self.joints[0], other.joints[1]), x=x_sum, y=y_sum, z=z_sum, style=self.kwargs.get('label'))

    def __sub__(self, other):
        if not isinstance(other, VectorBase):
            raise TypeError("Can only subtract VectorBase from another VectorBase.")
        _x1 = self.x if self.x is not None else 0.0
        _y1 = self.y if self.y is not None else 0.0
        _z1 = self.z if self.z is not None else 0.0
        _x2 = other.x if other.x is not None else 0.0
        _y2 = other.y if other.y is not None else 0.0
        _z2 = other.z if other.z is not None else 0.0
        x_diff, y_diff, z_diff = _x1 - _x2, _y1 - _y2, _z1 - _z2
        return VectorBase(joints=(self.joints[0], other.joints[1]), x=x_diff, y=y_diff, z=z_diff, style=self.kwargs.get('label'))

    def dot(self, other):
        """
        Computes the dot product with another VectorBase object.
        :param other: VectorBase; The other vector.
        :return: float; The dot product.
        """
        if not isinstance(other, VectorBase):
            raise TypeError("Can only compute dot product with another VectorBase.")
        _x1 = self.x if self.x is not None else 0.0
        _y1 = self.y if self.y is not None else 0.0
        _z1 = self.z if self.z is not None else 0.0
        _x2 = other.x if other.x is not None else 0.0
        _y2 = other.y if other.y is not None else 0.0
        _z2 = other.z if other.z is not None else 0.0
        return _x1 * _x2 + _y1 * _y2 + _z1 * _z2

    def cross(self, other):
        """
        Computes the cross product with another VectorBase object.
        :param other: VectorBase; The other vector.
        :return: VectorBase; The cross product vector. Joints are set to None.
        """
        if not isinstance(other, VectorBase):
            raise TypeError("Can only compute cross product with another VectorBase.")
        _x1 = self.x if self.x is not None else 0.0
        _y1 = self.y if self.y is not None else 0.0
        _z1 = self.z if self.z is not None else 0.0
        _x2 = other.x if other.x is not None else 0.0
        _y2 = other.y if other.y is not None else 0.0
        _z2 = other.z if other.z is not None else 0.0

        cross_x = _y1 * _z2 - _z1 * _y2
        cross_y = _z1 * _x2 - _x1 * _z2
        cross_z = _x1 * _y2 - _y1 * _x2
        return VectorBase(joints=None, x=cross_x, y=cross_y, z=cross_z, style=self.kwargs.get('label'))


class Vector:
    def __init__(self, joints=None, r=None, theta=None, x=None, y=None, z=None, show=True, style=None, **kwargs):
        """
        See the VectorBase class for details regarding the parameters. The purpose of this class is to group Position,
        Velocity, and Acceleration objects.

        Instance Variables
        ------------------
        pos: Position object which is a subclass of VectorBase. Does not include the r_dot, omega, r_ddot, and alpha
            attributes.
        vel: Velocity object which is a subclass of VectorBase. Does not include the r_ddot, alpha, and z_ddot attributes.
        acc: Acceleration object which is a subclass of VectorBase.
        """
        self.pos = Position(joints=joints, r=r, theta=theta, x=x, y=y, z=z, show=show, style=style, **kwargs)
        self.vel = Velocity(joints=joints, r=r, theta=theta, x=x, y=y, z=z, show=show, style=style, **kwargs) # z here is vz
        self.acc = Acceleration(joints=joints, r=r, theta=theta, x=x, y=y, z=z, show=show, style=style, **kwargs) # z here is az

        self.get = self.pos.get # Main get method for the composed vector (usually position)
        self.joints = joints

    def _update_velocity(self):
        """
        Updates the velocity object to include the length, r, and the angle ,theta.
        """
        # Update Velocity's r, theta, z based on Position's current state
        self.vel.r = self.pos.r
        self.vel.theta = self.pos.theta
        self.vel.z = self.pos.z # z component of position, for context in velocity calculation

    def _update_acceleration(self):
        """
        Updates the acceleration object to include r, theta, r_dot, and omega.
        """
        # Update Acceleration's r, theta, z, r_dot, omega, z_dot based on Velocity's current state
        self.acc.r = self.vel.r
        self.acc.theta = self.vel.theta
        self.acc.z = self.vel.z # z component of velocity (vz), for context
        self.acc.r_dot = self.vel.r_dot
        self.acc.omega = self.vel.omega
        self.acc.z_dot = self.vel.z_dot # Pass z_dot for acceleration calculation

    def _zero(self, s):
        """
        Zeros all the ndarray attributes at a certain size, s.

        :param s: int; The size of the data
        """
        self.pos.rs = np.zeros(s)
        self.pos.thetas = np.zeros(s)
        self.pos.zs = np.zeros(s) # New
        self.vel.rs = self.pos.rs
        self.vel.thetas = self.pos.thetas
        self.vel.zs = self.pos.zs # Stores z of position for vel calc context
        self.vel.r_dots = np.zeros(s)
        self.vel.omegas = np.zeros(s)
        self.vel.z_dots = np.zeros(s) # New: stores vz history
        self.acc.rs = self.vel.rs
        self.acc.thetas = self.vel.thetas
        self.acc.zs = self.vel.zs # Stores z of velocity for acc calc context
        self.acc.r_dots = self.vel.r_dots
        self.acc.omegas = self.vel.omegas
        self.acc.z_dots = self.vel.z_dots # Pass z_dot history
        self.acc.r_ddots = np.zeros(s)
        self.acc.alphas = np.zeros(s)
        self.acc.z_ddots = np.zeros(s) # New: stores az history

    def _set_position_data(self, i):
        """
        Sets position data at index, i.

        :param i: Index
        """
        self.pos.rs[i] = self.pos.r if self.pos.r is not None else 0.0
        self.pos.thetas[i] = self.pos.theta if self.pos.theta is not None else 0.0
        self.pos.zs[i] = self.pos.z if self.pos.z is not None else 0.0

    def _set_velocity_data(self, i):
        """
        Sets velocity data at index, i.

        :param i: Index
        """
        self.vel.r_dots[i] = self.vel.r_dot if self.vel.r_dot is not None else 0.0
        self.vel.omegas[i] = self.vel.omega if self.vel.omega is not None else 0.0
        self.vel.z_dots[i] = self.vel.z_dot if hasattr(self.vel, 'z_dot') and self.vel.z_dot is not None else 0.0 # Store actual z velocity component

    def _set_acceleration_data(self, i):
        """
        Sets acceleration data at index, i.

        :param i: Index
        """
        self.acc.r_ddots[i] = self.acc.r_ddot if self.acc.r_ddot is not None else 0.0
        self.acc.alphas[i] = self.acc.alpha if self.acc.alpha is not None else 0.0
        self.acc.z_ddots[i] = self.acc.z_ddot if hasattr(self.acc, 'z_ddot') and self.acc.z_ddot is not None else 0.0 # Store actual z acceleration component

    def __call__(self, *args):
        return self.get(*args)

    def __repr__(self):
        return f'{self.joints[0]}{self.joints[1]}'


class Position(VectorBase):
    def __init__(self, **kwargs):
        # z is handled by VectorBase __init__
        VectorBase.__init__(self, **kwargs)
        # Position does not have rates of change of r, theta, or z as primary attributes
        if hasattr(self, 'r_dot'): del self.r_dot
        if hasattr(self, 'r_ddot'): del self.r_ddot
        if hasattr(self, 'omega'): del self.omega
        if hasattr(self, 'alpha'): del self.alpha
        if hasattr(self, 'r_dots'): del self.r_dots
        if hasattr(self, 'omegas'): del self.omegas
        if hasattr(self, 'r_ddots'): del self.r_ddots
        if hasattr(self, 'alphas'): del self.alphas
        # Also remove z derivatives specific to velocity/acceleration from Position instance
        if hasattr(self, 'z_dot'): del self.z_dot
        if hasattr(self, 'z_dots'): del self.z_dots
        if hasattr(self, 'z_ddot'): del self.z_ddot
        if hasattr(self, 'z_ddots'): del self.z_ddots


    def _both(self, r_val, theta_val, z_val=None): # Corresponds to "get = self._both"
        # When r, theta, and optionally z are provided to define/update the position
        self.x, self.y = r_val * np.cos(theta_val), r_val * np.sin(theta_val)
        if z_val is not None:
            self.z = z_val
        # If z_val is None, self.z retains its value from __init__ or previous calls
        self.r, self.theta = r_val, theta_val
        return np.array([self.x, self.y, self.z if self.z is not None else 0.0])

    def _neither(self): # Corresponds to "get = self._neither" (r, theta, z assumed fixed from init)
        if self.x is None and self.y is None and self.r is not None and self.theta is not None:
            self.x, self.y = self.r * np.cos(self.theta), self.r * np.sin(self.theta)
        # self.z is already set during __init__ or defaults in VectorBase
        _x = self.x if self.x is not None else 0.0
        _y = self.y if self.y is not None else 0.0
        _z = self.z if self.z is not None else 0.0
        return np.array([_x, _y, _z])

    def _tangent(self, theta_val, z_val=None): # Corresponds to "get = self._tangent" (r fixed, theta input)
        if self.r is None: self.r = 0.0 # Default r if not set
        self.x, self.y = self.r * np.cos(theta_val), self.r * np.sin(theta_val)
        if z_val is not None:
            self.z = z_val
        self.theta = theta_val
        return np.array([self.x, self.y, self.z if self.z is not None else 0.0])

    def _slip(self, r_val, z_val=None): # Corresponds to "get = self._slip" (theta fixed, r input)
        if self.theta is None: self.theta = 0.0 # Default theta if not set
        self.x, self.y = r_val * np.cos(self.theta), r_val * np.sin(self.theta)
        if z_val is not None:
            self.z = z_val
        self.r = r_val
        return np.array([self.x, self.y, self.z if self.z is not None else 0.0])

    def __repr__(self):
        return f'Position(joints={self.joints}, r={self.r}, theta={self.theta}, x={self.x}, y={self.y}, z={self.z})'

    def __str__(self):
        return f'R_{self.joints[0]}{self.joints[1]}'


class Velocity(VectorBase):
    def __init__(self, **kwargs):
        VectorBase.__init__(self, **kwargs)
        # Velocity does not have second derivatives of r, theta, or z
        if hasattr(self, 'r_ddot'): del self.r_ddot
        if hasattr(self, 'alpha'): del self.alpha
        if hasattr(self, 'r_ddots'): del self.r_ddots
        if hasattr(self, 'alphas'): del self.alphas
        if hasattr(self, 'z_ddot'): del self.z_ddot
        if hasattr(self, 'z_ddots'): del self.z_ddots
        # Ensure r, theta, z (positional context) are present, default if necessary
        self.r = self.r if hasattr(self, 'r') and self.r is not None else 0.0
        self.theta = self.theta if hasattr(self, 'theta') and self.theta is not None else 0.0
        # self.z here is for positional context, not vz. vz will be stored in self.x, self.y, self.z of Velocity.
        # VectorBase init sets self.z (which for Velocity object means vz, if x,y,z were passed to init)

    def _neither(self): # r_dot=0, omega=0, z_dot=0
        self.x, self.y, self.z = 0.0, 0.0, 0.0 # vx, vy, vz are zero
        self.r_dot, self.omega, self.z_dot = 0.0, 0.0, 0.0
        return np.array([self.x, self.y, self.z])

    def _both(self, r_dot_val, omega_val, z_dot_val=None): # All rates r_dot, omega, z_dot are inputs
        # r, theta for context are from the associated Position vector (updated via _update_velocity)
        _r = self.r if self.r is not None else 0.0
        _theta = self.theta if self.theta is not None else 0.0
        self.x = r_dot_val * np.cos(_theta) - _r * omega_val * np.sin(_theta)
        self.y = r_dot_val * np.sin(_theta) + _r * omega_val * np.cos(_theta)
        self.z = z_dot_val if z_dot_val is not None else 0.0 # This is vz
        self.r_dot, self.omega = r_dot_val, omega_val
        self.z_dot = z_dot_val if z_dot_val is not None else 0.0 # Store instantaneous z_dot
        return np.array([self.x, self.y, self.z])

    def _tangent(self, omega_val, z_dot_val=None): # r_dot=0; omega and z_dot are inputs
        _r = self.r if self.r is not None else 0.0
        _theta = self.theta if self.theta is not None else 0.0
        self.x = -_r * omega_val * np.sin(_theta) # Simplified: self.r * omega_val * -np.sin()
        self.y = _r * omega_val * np.cos(_theta)
        self.z = z_dot_val if z_dot_val is not None else 0.0
        self.r_dot = 0.0 # r is constant
        self.omega = omega_val
        self.z_dot = z_dot_val if z_dot_val is not None else 0.0
        return np.array([self.x, self.y, self.z])

    def _slip(self, r_dot_val, z_dot_val=None): # omega=0; r_dot and z_dot are inputs
        _theta = self.theta if self.theta is not None else 0.0
        self.x = r_dot_val * np.cos(_theta)
        self.y = r_dot_val * np.sin(_theta)
        self.z = z_dot_val if z_dot_val is not None else 0.0
        self.r_dot = r_dot_val
        self.omega = 0.0 # theta is constant
        self.z_dot = z_dot_val if z_dot_val is not None else 0.0
        return np.array([self.x, self.y, self.z])

    def __repr__(self):
        return f'Velocity(joints={self.joints}, r_dot={self.r_dot}, omega={self.omega}, z_dot={self.z_dot}, vx={self.x}, vy={self.y}, vz={self.z})'

    def __str__(self):
        return f'V_{self.joints[0]}{self.joints[1]}'


class Acceleration(VectorBase):
    def __init__(self, **kwargs):
        VectorBase.__init__(self, **kwargs)
        # Acceleration uses all derivative attributes (r_ddot, alpha, z_ddot)
        # Ensure context attributes are present (r, theta, z from position; r_dot, omega, z_dot from velocity)
        self.r = self.r if hasattr(self, 'r') and self.r is not None else 0.0
        self.theta = self.theta if hasattr(self, 'theta') and self.theta is not None else 0.0
        # self.z here is positional context, not az. az stored in self.x,y,z of Accel object.
        self.r_dot = self.r_dot if hasattr(self, 'r_dot') and self.r_dot is not None else 0.0
        self.omega = self.omega if hasattr(self, 'omega') and self.omega is not None else 0.0
        self.z_dot = self.z_dot if hasattr(self, 'z_dot') and self.z_dot is not None else 0.0


    def _neither(self): # r_ddot=0, alpha=0, z_ddot=0
        self.x, self.y, self.z = 0.0, 0.0, 0.0 # ax, ay, az are zero
        self.r_ddot, self.alpha, self.z_ddot = 0.0, 0.0, 0.0
        return np.array([self.x, self.y, self.z])

    def _both(self, r_ddot_val, alpha_val, z_ddot_val=None): # All second rates are inputs
        # Context r, theta, r_dot, omega, z_dot from associated Velocity (updated via _update_acceleration)
        _r = self.r if self.r is not None else 0.0
        _theta = self.theta if self.theta is not None else 0.0
        _r_dot = self.r_dot if self.r_dot is not None else 0.0
        _omega = self.omega if self.omega is not None else 0.0
        # z_dot from velocity context

        self.x = (r_ddot_val * np.cos(_theta) - 2 * _r_dot * _omega * np.sin(_theta) -
                   _r * alpha_val * np.sin(_theta) - _r * _omega**2 * np.cos(_theta))
        self.y = (r_ddot_val * np.sin(_theta) + 2 * _r_dot * _omega * np.cos(_theta) +
                   _r * alpha_val * np.cos(_theta) - _r * _omega**2 * np.sin(_theta))
        self.z = z_ddot_val if z_ddot_val is not None else 0.0 # This is az
        self.r_ddot, self.alpha = r_ddot_val, alpha_val
        self.z_ddot = z_ddot_val if z_ddot_val is not None else 0.0 # Store instantaneous z_ddot
        return np.array([self.x, self.y, self.z])

    def _tangent(self, alpha_val, z_ddot_val=None): # r_ddot=0 (implies r_dot=0 for this formula), alpha and z_ddot are inputs
        # Assumes r_dot = 0 (constant r, so only tangential and centripetal acceleration in XY)
        _r = self.r if self.r is not None else 0.0
        _theta = self.theta if self.theta is not None else 0.0
        _omega = self.omega if self.omega is not None else 0.0 # omega from velocity

        self.x = (_r * alpha_val * -np.sin(_theta) - _r * _omega**2 * np.cos(_theta))
        self.y = (_r * alpha_val * np.cos(_theta) - _r * _omega**2 * np.sin(_theta))
        self.z = z_ddot_val if z_ddot_val is not None else 0.0
        self.r_ddot = 0.0 # r is constant, r_dot is zero
        self.alpha = alpha_val
        self.z_ddot = z_ddot_val if z_ddot_val is not None else 0.0
        return np.array([self.x, self.y, self.z])

    def _slip(self, r_ddot_val, z_ddot_val=None): # alpha=0 (implies omega=0 for this formula), r_ddot and z_ddot are inputs
        # Assumes omega = 0 (constant theta, so only radial acceleration in XY)
        _theta = self.theta if self.theta is not None else 0.0

        self.x = r_ddot_val * np.cos(_theta) # Purely radial acceleration if omega, alpha are zero
        self.y = r_ddot_val * np.sin(_theta)
        self.z = z_ddot_val if z_ddot_val is not None else 0.0
        self.r_ddot = r_ddot_val
        self.alpha = 0.0 # theta is constant, omega is zero
        self.z_ddot = z_ddot_val if z_ddot_val is not None else 0.0
        return np.array([self.x, self.y, self.z])

    def __repr__(self):
        return f'Acceleration(joints={self.joints}, r_ddot={self.r_ddot}, alpha={self.alpha}, z_ddot={self.z_ddot}, ax={self.x}, ay={self.y}, az={self.z})'

    def __str__(self):
        return f'A_{self.joints[0]}{self.joints[1]}'
