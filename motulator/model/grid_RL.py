# pylint: disable=C0103
"""
This module contains continuous-time models for first order dynamic model of
an RL line.

"""
import numpy as np

from motulator.helpers import complex2abc


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
    def __init__(self, U_gN=400*np.sqrt(2/3), L_f = 6e-3, R_f=0, L_g=0, R_g=0):
        self.L_f = L_f
        self.R_f = R_f
        self.L_g = L_g
        self.R_g = R_g
        # Storing the PCC voltage value
        self.u_gs0 = U_gN + 0j
        # Initial values
        self.i_gs0 = 0j



    def pcc_voltages(self, i_gs, u_cs, e_gs):
        """
        Compute the PCC voltage, located in-between the filter and the line
        impedances

        Parameters
        ----------
        i_gs : complex
            Line current (A).
        u_cs : complex
            Converter-side voltage (V).
        e_gs : complex
            Grid-side voltage (V).

        Returns
        -------
        u_gs : complex
            Voltage at the point of common coupling (PCC).

        """
        # calculation of voltage-related term
        v_tu = ((self.L_g/(self.L_g+self.L_f))*u_cs + 
            (self.L_f/(self.L_g+self.L_f))*e_gs)
        # calculation of current-related term
        v_ti = (
            ((self.R_g*self.L_f - self.R_f*self.L_g)/(self.L_g+self.L_f))*i_gs)
        
        # PCC voltage in alpha-beta coordinates
        u_gs = v_tu + v_ti
        
        return u_gs
    
    
    def f(self, i_gs, u_cs, e_gs):
        # pylint: disable=R0913
        """
        Compute the state derivatives.

        Parameters
        ----------
        i_gs : complex
            Line current (A).
        u_cs : complex
            Converter-side voltage (V).
        e_gs : complex
            Grid-side voltage (V).

        Returns
        -------
        di_gs: complex
            Time derivative of the complex state i_gs (line current, in A)

        """
        # Calculation of the total impedance
        L_t = self.L_f + self.L_g
        R_t = self.R_f + self.R_g
        
        di_gs = (u_cs - e_gs - R_t*i_gs)/L_t
        
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
    
    
    def meas_pcc_voltage(self):
        """
        Measure the PCC voltages at the end of the sampling period.

        Returns
        -------
        u_g_abc : 3-tuple of floats
            Phase voltage at the point of common coupling (PCC).

        """  
        # PCC voltage space vector in stationary coordinates
        u_g_abc = complex2abc(self.u_gs0)  # + noise + offset ...
        return u_g_abc
