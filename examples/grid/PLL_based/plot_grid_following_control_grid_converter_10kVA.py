"""
Example simulation script: 10kVA grid-following controlled converter connected
to a perfect AC voltage source (grid) with an L filter.

    
The grid-following control (PLL-based, current source mode) includes
    - (optional) DC-bus voltage controller;
    - Phase-Locked Loop (PLL) to synchronize with the grid;
    - dq current reference generation;
    - vector current controller based on PI.

"""


# %%
# Import the packages.

import numpy as np
import motulator as mt

# To check the computation time of the program
import time
start_time = time.time()

# %%
# Compute base values based on the nominal values (just for figures).

base = mt.BaseValues(
    U_nom=370, I_nom=15.5, f_nom=105.8, tau_nom=20.1, P_nom=6.9e3, p=2)


# %%
# Configure the system model (grid model)
rl_model = mt.InverterToInductiveGrid(L_g=10e-3, R_g=0)
grid_model = mt.Grid(U_gN=np.sqrt(2/3)*400, w_g=2*np.pi*50)
dc_model = None
conv = mt.Inverter(u_dc=650)
"""
REMARK:
    if you do not want to simulate any DC grid, you should define
    dc_model = None. This would make the DC voltage constant, using the
    value given in the converter model.
    Do not forget also to activate/desactivate the dc-bus control
"""
    
if dc_model == None:
    mdl = mt.GridCompleteModel(rl_model, grid_model, conv)
else:
    mdl = mt.ACDCGridCompleteModel(
        rl_model, grid_model, dc_model, conv
        )


pars = mt.GridFollowingCtrlPars(
            L_f=10e-3,
            R_f=0,
            C_dc = 1e-3,
            on_v_dc=False,
            S_base = 10e3,
            i_max = 1.5,
            )
ctrl = mt.GridFollowingCtrl(pars)


# %%

# Set the active and reactive power references
ctrl.p_g_ref = lambda t: (t > .02)*(5e3)
ctrl.q_g_ref = lambda t: (t > .04)*(4e3)

# AC-voltage magnitude (to simulate voltage dips or short-circuits)
u_g_abs_var =  lambda t: np.sqrt(2/3)*400
mdl.grid_model.u_g_abs_A = u_g_abs_var #phase a
mdl.grid_model.u_g_abs_B = u_g_abs_var #phase b
mdl.grid_model.u_g_abs_C = u_g_abs_var #phase c


# Create the simulation object and simulate it
sim = mt.simulation.Simulation(mdl, ctrl, pwm=True)
sim.simulate(t_stop = .1)

# Print the execution time
print('\nExecution time: {:.2f} s'.format((time.time() - start_time)))

# Plot results in per unit values
mt.plot_grid(sim)