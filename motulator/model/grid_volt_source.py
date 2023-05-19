# pylint: disable=C0103
"""
This module contains continuous-time models for 3-phase voltage source
of a grid.

"""
import numpy as np

from motulator.helpers import (
    complex2abc,
    abc2complex
    )

# %%
class StiffSource:
    """
    Grid subsystem.

    This model is a constant frequency 3-phase voltage source of the AC grid.

    Parameters
    ----------
    U_gN : float
        nominal voltage peak value (phase to ground)
    w_g : float
        grid constant frequency
    e_g_abs : function
        3-phase grid voltage magnitude
    """
    
    def __init__(self, w_N=2*np.pi*50,
                 e_g_abs=lambda t: 400*np.sqrt(2/3)):
        self.w_N = w_N
        self.e_g_abs = e_g_abs

    def voltages(self, t):
        """
        Compute the voltage in stationary frame at the grid output:
           
        Parameters
        ----------
        t : float
            Time.

        Returns
        -------
        e_gs: complex
            grid complex voltage.

        """
        # Integration of frequency to obtain the angle
        theta = self.w_N*t        
        
        # Calculation of the three-phase voltage
        e_g_a = self.e_g_abs(t)*np.cos(theta)
        e_g_b = self.e_g_abs(t)*np.cos(theta-2*np.pi/3)
        e_g_c = self.e_g_abs(t)*np.cos(theta-4*np.pi/3)
        
        
        e_gs = abc2complex([e_g_a, e_g_b, e_g_c])
        return e_gs


    def meas_voltages(self, t):
        """
        Measure the phase voltages at the end of the sampling period.

        Returns
        -------
        e_g_abc : 3-tuple of floats
            Phase voltages.

        """
        # Grid voltage
        e_g_abc = complex2abc(self.voltages(t))  # + noise + offset ...
        return e_g_abc

# %%
class FlexSource:
    """
    Grid subsystem.
    This models the 3-phase voltage source of the AC grid while taking into
    account the electromechanical dynamics of a typical grid generated by the 
    synchronous generators.
    
    More information about the model can be found in [1].
    
    [1] : ENTSO-E, Documentation on Controller Tests in Test Grid
    Configurations, Technical Report, 26.11.2013.
    Parameters
    ----------
    U_gN : float
        Voltage peak value (phase to ground)
    w_N : float
        grid constant frequency
    """

    def __init__(
            self, T_D=10,
            T_N=3,
            H_g=3,
            D_g=0,
            r_d=.05,
            T_gov=0.5,
            U_gN=400*np.sqrt(2/3),
            w_N=2*np.pi*50,
            S_grid =500e6,
            e_g_abs=lambda t: 400*np.sqrt(2/3),
            p_m_ref=lambda t: 0,
            p_e=lambda t: 0):
        self.U_gN=U_gN
        self.e_g_abs=e_g_abs
        self.w_N=w_N
        self.S_grid=S_grid
        self.T_D=T_D
        self.T_N=T_N
        self.H_g=H_g
        self.D_g=D_g
        self.r_d=r_d*w_N/S_grid
        self.T_gov=T_gov
        self.p_e=p_e
        self.p_m_ref=p_m_ref
        # Initial values
        self.w_g0 = w_N
        self.err_w_g0, self.p_gov0, self.x_turb0, self.theta_g0 = 0, 0, 0, 0

    def f(self, t, err_w_g, p_gov, x_turb):
        """
        Compute the state derivative.
        Parameters
        ----------
        t : float
            Time.
        err_w_g : float
            grid angular speed deviation (in mechanical rad/s).
        p_e : float
            Electric power.
        Returns
        -------
        list, length 2
            Time derivative of the state vector.
        """
        # calculation of mechanical power from the turbine output
        p_m = (self.T_N/self.T_D)*p_gov + (1-(self.T_N/self.T_D))*x_turb        
        # swing equation
        p_diff = (p_m - self.p_e(t))/self.S_grid # in per units
        derr_w_g = self.w_N*(p_diff - self.D_g*err_w_g)/(2*self.H_g)
        # governor dynamics  
        dp_gov = (self.p_m_ref(t) - (1/self.r_d)*err_w_g - p_gov) / self.T_gov
        # turbine dynamics (lead-lag)
        dx_turb = (p_gov - x_turb)/self.T_D
        # integration of the angle
        dtheta_g = self.w_N + err_w_g
        return [derr_w_g, dp_gov, dx_turb, dtheta_g]
    
    def voltages(self, t, theta_g):
        """
        Compute the voltage in stationary frame at the grid output:
           
        Parameters
        ----------
        t : float
            Time.

        Returns
        -------
        e_gs: complex
            grid complex voltage.

        """ 
        
        # Calculation of the three-phase voltage
        e_g_a = self.e_g_abs(t)*np.cos(theta_g)
        e_g_b = self.e_g_abs(t)*np.cos(theta_g-2*np.pi/3)
        e_g_c = self.e_g_abs(t)*np.cos(theta_g-4*np.pi/3)
        
        
        e_gs = abc2complex([e_g_a, e_g_b, e_g_c])
        return e_gs


    def meas_voltages(self, t):
        """
        Measure the phase voltages at the end of the sampling period.

        Returns
        -------
        e_g_abc : 3-tuple of floats
            Phase voltages.

        """
        # Grid voltage
        e_g_abc = complex2abc(self.voltages(t))  # + noise + offset ...
        return e_g_abc
    
    
    def meas_freq(self):
        """
        Measure the grid frequency.
        This returns the grid frequency at the end of the sampling period.
        Returns
        -------
        w_g0 : float
            Grid angular speed (in rad/s).
        """
        w_g0 = self.w_g0
        return w_g0

    def meas_angle(self):
        """
        Measure the grid angle.
        This returns the grid angle at the end of the sampling period.
        Returns
        -------
        theta_g0 : float
            grid electrical angle (in rad).
        """
        return self.theta_g0