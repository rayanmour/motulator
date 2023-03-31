# pylint: disable=C0103
"""
This module contains continuous-time models for grid connected converters.

Peak-valued complex space vectors are used. The space vector models are
implemented in stationary coordinates.

"""
from __future__ import annotations
import numpy as np
from motulator.helpers import Bunch
from motulator.helpers import (
    complex2abc,
    abc2complex
    )

# %%
class GridCompleteModel:
    """
    Continuous-time model for a grid model with an RL impedance model.

    This interconnects the subsystems of a converter with a grid and provides
    an interface to the solver. More complicated systems could be modeled using
    a similar template.

    Parameters
    ----------
    rl_model : InductiveGrid
        RL line dynamic model.
    grid_model : Grid
        Constant voltage source model.
    conv : Inverter | PWMInverter
        Inverter model.

    """
    
    def __init__(self, rl_model=None, grid_model=None, conv=None):
        self.rl_model = rl_model
        self.grid_model = grid_model
        self.conv = conv

        # Initial time
        self.t0 = 0

        # Store the solution in these lists
        self.data = Bunch()
        self.data.t, self.data.q = [], []
        self.data.i_gs = []

        
    def get_initial_values(self):
        """
        Get the initial values.

        Returns
        -------
        x0 : complex list, length 1
            Initial values of the state variables.

        """
        x0 = [self.rl_model.i_gs0]

        return x0

    def set_initial_values(self, t0, x0):
        """
        Set the initial values.

        Parameters
        ----------
        x0 : complex ndarray
            Initial values of the state variables.

        """
        self.t0 = t0
        self.rl_model.i_gs0 = x0[0]
        # calculation of converter-side voltage
        u_cs0 = self.conv.ac_voltage(self.conv.q, self.conv.u_dc0)
        # calculation of grid-side voltage
        e_gs0 = self.grid_model.voltages(t0)
        # update pcc voltage
        self.rl_model.u_gs0 = self.rl_model.pcc_voltages(x0[0], u_cs0, e_gs0)

    def f(self, t, x):
        """
        Compute the complete state derivative list for the solver.

        Parameters
        ----------
        t : float
            Time.
        x : complex ndarray
            State vector.

        Returns
        -------
        complex list
            State derivatives.

        """
        # Unpack the states
        i_gs = x
        # Interconnections: outputs for computing the state derivatives
        u_cs = self.conv.ac_voltage(self.conv.q, self.conv.u_dc0)
        e_gs = self.grid_model.voltages(t)
        # State derivatives
        rl_f = self.rl_model.f(i_gs, u_cs, e_gs)
        # List of state derivatives 
        return rl_f

    def save(self, sol):
        """
        Save the solution.

        Parameters
        ----------
        sol : Bunch object
            Solution from the solver.

        """
        self.data.t.extend(sol.t)
        self.data.i_gs.extend(sol.y[0])
        self.data.q.extend(sol.q)

                                    
    def post_process(self):
        """
        Transform the lists to the ndarray format and post-process them.

        """
        # From lists to the ndarray
        self.data.t = np.asarray(self.data.t)
        self.data.i_gs = np.asarray(self.data.i_gs)
        self.data.q = np.asarray(self.data.q)

        # Some useful variables
        self.data.e_gs = self.grid_model.voltages(self.data.t)
        self.data.theta = np.mod(self.data.t*self.grid_model.w_N, 2*np.pi)
        self.data.u_cs = self.conv.ac_voltage(self.data.q, self.conv.u_dc0)
        self.data.u_gs = self.rl_model.pcc_voltages(self.data.i_gs,self.data.u_cs,self.data.e_gs)


# %%
class ACDCGridCompleteModel:
    """
    Continuous-time model for a grid model with an RL impedance model.

    This interconnects the subsystems of a converter with a grid and provides
    an interface to the solver. More complicated systems could be modeled using
    a similar template.

    Parameters
    ----------
    rl_model : InductiveGrid
        RL line dynamic model.
    grid_model : Grid
        Constant voltage source model.
    dc_model : DcGrid
        DC grid voltage dynamics (capacitance model)
    conv : Inverter | PWMInverter
        Inverter model.

    """
    
    def __init__(self, rl_model=None, grid_model=None, dc_model=None, conv=None):
        self.rl_model = rl_model
        self.grid_model = grid_model
        self.dc_model = dc_model
        self.conv = conv

        # Initial time
        self.t0 = 0

        # Store the solution in these lists
        self.data = Bunch()
        self.data.t, self.data.q = [], []
        self.data.i_gs = []
        self.data.u_dc = [] 
        self.data.i_L = []
        
    def get_initial_values(self):
        """
        Get the initial values.

        Returns
        -------
        x0 : complex list, length 1
            Initial values of the state variables.

        """
        x0 = [self.rl_model.i_gs0, self.dc_model.u_dc0]
        return x0

    def set_initial_values(self, t0, x0):
        """
        Set the initial values.

        Parameters
        ----------
        x0 : complex ndarray
            Initial values of the state variables.

        """
        self.t0 = t0
        self.rl_model.i_gs0 = x0[0]
        self.dc_model.u_dc0 = x0[1].real
        self.conv.u_dc0 = x0[1].real
        # calculation of converter-side voltage
        u_cs0 = self.conv.ac_voltage(self.conv.q, x0[1].real)
        # calculation of grid-side voltage
        e_gs0 = self.grid_model.voltages(t0)
        # update pcc voltage
        self.rl_model.u_gs0 = self.rl_model.pcc_voltages(x0[0], u_cs0, e_gs0)

    def f(self, t, x):
        """
        Compute the complete state derivative list for the solver.

        Parameters
        ----------
        t : float
            Time.
        x : complex ndarray
            State vector.

        Returns
        -------
        complex list
            State derivatives.

        """
        # Unpack the states
        i_gs, u_dc = x
        # Interconnections: outputs for computing the state derivatives
        u_cs = self.conv.ac_voltage(self.conv.q, u_dc)
        e_gs = self.grid_model.voltages(t)
        q = self.conv.q
        i_g_abc = complex2abc(i_gs)
        # State derivatives
        rl_f = self.rl_model.f(i_gs, u_cs, e_gs)
        dc_f = self.dc_model.f(t, u_dc, i_g_abc, q)
        # List of state derivatives
        return [rl_f, dc_f]

    def save(self, sol):
        """
        Save the solution.

        Parameters
        ----------
        sol : Bunch object
            Solution from the solver.

        """
        self.data.t.extend(sol.t)
        self.data.i_gs.extend(sol.y[0])
        self.data.u_dc.extend(sol.y[1].real)
        self.data.q.extend(sol.q)
        q_abc=complex2abc(np.asarray(sol.q))
        i_c_abc=complex2abc(sol.y[0])
        self.data.i_L.extend(q_abc[0]*i_c_abc[0] + q_abc[1]*i_c_abc[1] + q_abc[2]*i_c_abc[2])
                                    
    def post_process(self):
        """
        Transform the lists to the ndarray format and post-process them.

        """
        # From lists to the ndarray
        self.data.t = np.asarray(self.data.t)
        self.data.i_gs = np.asarray(self.data.i_gs)
        self.data.u_dc = np.asarray(self.data.u_dc)
        self.data.q = np.asarray(self.data.q)
        #self.data.theta = np.asarray(self.data.theta)
        # Some useful variables
        self.data.i_L = np.asarray(self.data.i_L)
        self.data.e_gs = self.grid_model.voltages(self.data.t)
        self.data.theta = np.mod(self.data.t*self.grid_model.w_N, 2*np.pi)
        self.data.u_cs = self.conv.ac_voltage(self.data.q, self.conv.u_dc0)
        self.data.u_gs = self.rl_model.pcc_voltages(self.data.i_gs,self.data.u_cs,self.data.e_gs)
