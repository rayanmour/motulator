# pylint: disable=C0103
"""
This module contains dc grid model.

The dc grid model defined here is used if the dc bus is modelled as a current 
source and the capacitance dynamics need to be modeled.

"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from motulator.helpers import (
    complex2abc,
    abc2complex
    )

# %%
@dataclass
class DcGrid:
    """
    DC grid

    This model is used to compute the DC grid dynamics, represented by a first
    order system with the DC-bus capacitance dynamics.

     Parameters
     ----------
     C_dc : float
         DC grid capacitance (in Farad)
     G_dc : float
         DC grid conductance (in Siemens)
    i_dc : function
        External dc current, seen as disturbance, `idc(t)`.

    Returns
    -------
    real list, length 1
        DC voltage at the capacitance level udc

     """
    C_dc: float = 1e-3
    G_dc: float = 0
    i_dc: Callable[[float], float] = field(repr=False, default=lambda t: 0)
    # Initial values
    u_dc0: float = field(repr=False, default=540) #same value as converter.py
    # Initial value of the DC-bus voltage
    
    def dc_current(self, i_c_abc, q):
        """
        Compute the DC-side converter current, used to model the DC-bus voltage
        dynamics.
    
        Parameters
        ----------
       i_c_abc : ndarray, shape (3,)
           Phase currents.
       q : complex ndarray, shape (3,)
           Switching state vectors corresponding to the switching instants.
           For example, the switching state q[1] is applied at the interval
           t_n_sw[1].
    
        Returns
        -------
        i_L: float
            dc-side converter voltage
    
        """
        # Duty ratio back into three-phase ratios
        q_abc = complex2abc(q)
        
        # Dot product
        i_L = q_abc[0]*i_c_abc[0] + q_abc[1]*i_c_abc[1] + q_abc[2]*i_c_abc[2]
        
        return i_L
    
    
    def f(self, t, u_dc, i_c_abc, q):
        # pylint: disable=R0913
        """
        Compute the state derivatives.

        Parameters
        ----------
        t : float
            Time.
        u_dc: float
            DC bus voltage (in V)
       i_c_abc : ndarray, shape (3,)
           Phase currents.
       q : complex ndarray, shape (3,)
           Switching state vectors corresponding to the switching instants.
           For example, the switching state q[1] is applied at the interval
           t_n_sw[1].
        Returns
        -------
        du_dc: float
            Time derivative of the state variable, udc (DC capacitance voltage)

        """
        
        # Calculation of the dc-current
        i_L = self.dc_current(i_c_abc, q)
          
        # State derivatives
        du_dc = (self.i_dc(t) - i_L - self.G_dc*u_dc)/self.C_dc
                
        return du_dc

    def meas_dc_voltage(self):
        """
        Measure the DC voltage at the end of the sampling period
    
        Returns
        -------
        u_dc: float
            DC bus voltage (in V)
    
        """
        return self.u_dc0  # + noise + offset ...
