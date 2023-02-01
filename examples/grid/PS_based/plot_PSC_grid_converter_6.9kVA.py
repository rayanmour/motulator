"""
Example simulation script: 6.9kVA grid-forming controlled converter connected
to a perfect AC voltage source (grid) with an L filter.

    
The grid-forming control (power synchronization control) includes
    - (optional) DC-bus voltage controller --> not tested yet;
    - inner current controller used to damp the current oscillations;
    - power synchronization loop;
    - (optional) reference-feedforward d-axis current reference injection.

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
rl_model = mt.InverterToInductiveGrid(L_g=73.8e-3, R_g=0)
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


pars = mt.PSCtrlPars(
        L_f=73.8e-3,
        R_f=0,
        C_dc = 1e-3,
        on_rf=False,
        on_v_dc=False,
        S_base=base.P_nom,
        i_max = 1.5,
        w_0_cc = 2*np.pi*5,
        R_a = 4.64)
ctrl = mt.PSCtrl(pars)

# %%

# Set the active and reactive power references
ctrl.p_g_ref = lambda t: (t > .2)*(2.3e3) + (t > .5)*(2.3e3) + (t > .8)*(2.3e3) - (t > 1.2)*(6.9e3)
ctrl.q_g_ref = lambda t: 0 # this one is not used in PSC but required

#DC-side current (only if dc model is used)
if dc_model != None:
    mdl.dc_model.idc = lambda t: (t > .06)*(10)

#AC-voltage magnitude (to simulate voltage dips or short-circuits)
u_g_abs_var =  lambda t: np.sqrt(2/3)*400
mdl.grid_model.u_g_abs_A = u_g_abs_var #phase a
mdl.grid_model.u_g_abs_B = u_g_abs_var #phase b
mdl.grid_model.u_g_abs_C = u_g_abs_var #phase c

#Dc voltage reference
ctrl.u_dc_ref = lambda t: 600 + (t > .02)*(50)


# Create the simulation object and simulate it
sim = mt.simulation.Simulation(mdl, ctrl, pwm=True)
sim.simulate(t_stop = 1.5)

# Print the execution time
print('\nExecution time: {:.2f} s'.format((time.time() - start_time)))

# Plot results in per unit values
mt.plot_grid(sim)
#plotting.plot_extra_pu(sim,(0.015, 0.025))
#plotting.plot_extra_pu(sim,(0.035, 0.045))