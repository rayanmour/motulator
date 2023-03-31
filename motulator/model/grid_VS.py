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