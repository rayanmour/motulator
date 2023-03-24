# pylint: disable=C0103
"""
This module contains continuous-time models for first order dynamic model of
an RL line.

"""
from __future__ import annotations
import numpy as np
from motulator.helpers import complex2abc


import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# %%
class InductiveGrid:
    """
    Inductive grid model where one of the output voltage is controllable.

    An inductive grid model is built using a simple inductance model where the
    two output voltages are imposed and the current can be calculated using
    dynamic equations.

    Parameters
    ----------
    L_g : float
        Grid inductance (in H)
    R_g : float
        Grid resistance (in Ohm)
    u_cs : function
        External voltage at the impedance outputs, `u_cs(t)`.

    """
    
    def __init__(self, L_g=10e-3, R_g=0, u_cs=lambda t: 0j):
        self.L_g = L_g
        self.R_g = R_g
        self.u_cs = u_cs
        # Initial values
        self.i_gs0 = 0j

    
    def f(self, t, i_gs, u_gs):
        # pylint: disable=R0913
        """
        Compute the state derivatives.

        Parameters
        ----------
        t: real
            Time.
        i_gs : complex
            Line current.
        u_gs : complex
            Grid voltage.

        Returns
        -------
        di_gs: complex
            Time derivative of the state vector, igs (line current)

        """
        di_gs = (self.u_cs(t) - u_gs - self.R_g*i_gs)/self.L_g
        return di_gs
    
    def meas_currents(self):
        """
        Measure the phase currents at the end of the sampling period.

        Returns
        -------
        i_g_abc : 3-tuple of floats
            Phase currents.

        """
        # Line current space vector in stationary coordinates
        i_g_abc = complex2abc(self.i_gs0)  # + noise + offset ...
        return i_g_abc

# %%
class InverterToInductiveGrid:
    """
    Inductive grid model with a connection made to the inverter outputs.

    An inductive grid model is built using a simple inductance model where the
    two output voltages are imposed and the current can be calculated using
    dynamic equations.

    Parameters
    ----------
    L_f : float
        Filter inductance (in H)
    R_f : float
        Filter resistance (in Ohm)
    L_g : float
        Grid inductance (in H)
    R_g : float
        Grid resistance (in Ohm)

    """
    def __init__(self, U_gN=400*np.sqrt(2/3), L_f = 6e-3, R_f = 0, L_g=30e-3, R_g=0):
        self.L_f = L_f
        self.R_f = R_f
        self.L_g = L_g
        self.R_g = R_g
        # Storing the voltage from the derivative function
        self.u_cs0 = U_gN + 0j
        self.u_gs0 = U_gN + 0j
        # Initial values
        self.i_gs0 = 0j

    
    def f(self, i_gs, u_cs, u_gs):
        # pylint: disable=R0913
        """
        Compute the state derivatives.

        Parameters
        ----------
        i_gs : complex
            Line current (A).
        u_cs : complex
            Input voltage (V).
        u_gs : complex
            Output voltage (V).

        Returns
        -------
        di_gs: complex
            Time derivative of the complex state i_gs (line current, in A)

        """
        # Calculation of the total impedance
        L_t = self.L_f + self.L_g
        R_t = self.R_f + self.R_g
        
        di_gs = (u_cs - u_gs - R_t*i_gs)/L_t
        
        return di_gs
    
    def input_voltages(self, i_gs, u_gs, w_g):
        """
        Compute the voltage at the input of the inductor based on the output
        voltage value and the line current in the stationary frame.
           
        Parameters
        ----------
        i_gs : complex
            Line current (A).
        u_gs : complex
            Output voltage (V).
        w_g : float
            Grid angular speed (in rad/s).

        Returns
        -------
        u_cs : complex
            Input voltage (V).

        """
        
        # computation of input voltage
        u_cs = u_gs + 1j*w_g*(self.L_f + self.L_g)*i_gs
        
        return u_cs

    def meas_currents(self):
        """
        Measure the phase currents at the end of the sampling period.

        Returns
        -------
        i_g_abc : 3-tuple of floats
            Phase currents.

        """
        # Line current space vector in stationary coordinates
        i_g_abc = complex2abc(self.i_gs0)  # + noise + offset ...
        return i_g_abc
    
    
    def meas_pcc_voltage(self):
        """
        Measure the phase currents at the end of the sampling period.

        Returns
        -------
        u_pcc_abc : 3-tuple of floats
            Phase voltage at the point of common coupling (PCC).

        """
        # PCC voltage in alpha-beta coordinates (neglecting resistive effects)
        u_pccs = (self.L_g /(self.L_f + self.L_g))*self.u_cs0 + (self.L_f /(self.L_f + self.L_g))*self.u_gs0
        
        u_pcc_abc = complex2abc(u_pccs)  # + noise + offset ...
        return u_pcc_abc
