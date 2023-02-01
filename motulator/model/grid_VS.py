# pylint: disable=C0103
"""
This module contains continuous-time models for grid models using a constant voltage approximation.

"""
from __future__ import annotations
from collections.abc import Callable
import numpy as np
from dataclasses import dataclass, field
from motulator.helpers import (
    complex2abc,
    abc2complex
    )

import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# %%
@dataclass
class Grid:
    """
    Grid subsystem.

    This models the 3-phase voltage source of the AC grid.

    Parameters
    ----------
    U_gN : float
        Voltage peak value (phase to ground)
    w_g : float
        grid constant frequency

    """
    
    U_gN: float = 400*np.sqrt(2/3) #in volts
    w_g: float = 2*np.pi*50 #in rad/s
    u_g_abs_A: Callable[[float], float] = field(repr=False,
                                                default=lambda t: 400*np.sqrt(2/3)) #in volts
    u_g_abs_B: Callable[[float], float] = field(repr=False,
                                                default=lambda t: 400*np.sqrt(2/3)) #in volts
    u_g_abs_C: Callable[[float], float] = field(repr=False,
                                                default=lambda t: 400*np.sqrt(2/3)) #in volts

    def voltages(self, t):
        """
        Compute the voltage in stator frame at the grid output:
           (remark: I think it could also be possible to implement this function using theta_0 instead)

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
        theta = self.w_g*t        
        
        # Calculation of the three-phase voltage
        u_g_a = self.u_g_abs_A(t)*np.cos(theta)
        u_g_b = self.u_g_abs_B(t)*np.cos(theta-2*np.pi/3)
        u_g_c = self.u_g_abs_C(t)*np.cos(theta-4*np.pi/3)
        
        
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