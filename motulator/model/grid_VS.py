# pylint: disable=C0103
"""
This module contains continuous-time models for grid models using a constant voltage approximation.

"""
from __future__ import annotations
import numpy as np
from motulator.helpers import (
    complex2abc,
    abc2complex
    )

import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# %%
class Grid:
    """
    Grid subsystem.

    This models the 3-phase voltage source of the AC grid.

    Parameters
    ----------
    U_gN : float
        nominal voltage peak value (phase to ground)
    w_g : float
        grid constant frequency
    u_g_abs : function
        3-phase grid voltage magnitude (rms)
    """
    
    def __init__(self, w_N=2*np.pi*50,
                 u_g_abs=lambda t: 400*np.sqrt(2/3)):
        self.w_N = w_N
        self.u_g_abs = u_g_abs

    def voltages(self, t):
        """
        Compute the voltage in stationary frame at the grid output:
           
        Parameters
        ----------
        t : float
            Time.

        Returns
        -------
        u_gs: complex
            grid complex voltage.

        """
        # Integration of frequency to obtain the angle
        theta = self.w_N*t        
        
        # Calculation of the three-phase voltage
        u_g_a = self.u_g_abs(t)*np.cos(theta)
        u_g_b = self.u_g_abs(t)*np.cos(theta-2*np.pi/3)
        u_g_c = self.u_g_abs(t)*np.cos(theta-4*np.pi/3)
        
        
        u_gs = abc2complex([u_g_a, u_g_b, u_g_c])
        return u_gs


    def meas_voltages(self, t):
        """
        Measure the phase voltages at the end of the sampling period.

        Returns
        -------
        u_g_abc : 3-tuple of floats
            Phase voltages.

        """
        # Grid voltage
        u_g_abc = complex2abc(self.voltages(t))  # + noise + offset ...
        return u_g_abc