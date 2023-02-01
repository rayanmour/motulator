# pylint: disable=C0103
"""
This module contains continuous-time models for first order dynamic model of
an RL line.

"""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass, field
import numpy as np
from motulator.helpers import complex2abc


import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# %%
@dataclass
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
    u_c_s : function
        External voltage at the impedance outputs, `ucs(t)`.

    Returns
    -------
    complex list, length 1
        Line current in stator frame, igs

    """
    L_g: float = 10e-3
    R_g: float = 0
    u_cs: Callable[[complex], complex] = field(repr=False,
                                                default=lambda t: 0+1j*0)
    # Initial values
    i_gs0: complex = field(repr=False, default=0j)

    
    def f(self, t, i_gs, u_gs, w_g):
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
        w_g : float
            Grid angular speed (in mechanical rad/s).

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
        # Stator current space vector in stator coordinates
        i_g_abc = complex2abc(self.i_gs0)  # + noise + offset ...
        return i_g_abc

# %%
@dataclass
class InverterToInductiveGrid:
    """
    Inductive grid model with a connection made to the inverter outputs.

    An inductive grid model is built using a simple inductance model where the
    two output voltages are imposed and the current can be calculated using
    dynamic equations.

    Parameters
    ----------
    L_g : float
        Grid inductance (in H)
    R_g : float
        Grid resistance (in Ohm)

    Returns
    -------
    complex list, length 1
        Line current in stator frame, igs

    """
    L_g: float = 10e-3
    R_g: float = 0
    # Initial values
    i_gs0: complex = field(repr=False, default=0j)

    
    def f(self, i_gs, u_cs, u_gs, w_g):
        # pylint: disable=R0913
        """
        Compute the state derivatives.

        Parameters
        ----------
        i_gs : complex
            Line current.
        u_cs : complex
            Point of Common Coupling (PCC) voltage.
        u_gs : complex
            Grid voltage.
        w_g : float
            Grid angular speed (in mechanical rad/s).

        Returns
        -------
        di_gs: complex
            Time derivative of the state vector, igs (line current)

        """
        di_gs = (u_cs - u_gs - self.R_g*i_gs)/self.L_g
        return di_gs

    def meas_currents(self):
        """
        Measure the phase currents at the end of the sampling period.

        Returns
        -------
        i_g_abc : 3-tuple of floats
            Phase currents.

        """
        # Stator current space vector in stator coordinates
        i_g_abc = complex2abc(self.i_gs0)  # + noise + offset ...
        return i_g_abc
