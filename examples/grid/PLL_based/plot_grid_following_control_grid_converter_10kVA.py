"""
Example simulation script: 10-kVA grid converter connected to a symmetrical
three-phase AC voltage source (grid) through an inductive filter.
    
The control system includes
    - Phase-Locked Loop (PLL) to synchronize with the grid;
    - Current reference generation;
    - Proportional-integral (PI) vector current controller
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
base_values = mt.BaseValuesElectrical(
    U_nom=400, I_nom=14.5, f_nom=50.0, P_nom=10e3)


# %%
# Configure the system model
rl_model = mt.InverterToInductiveGrid(L_g=10e-3, R_g=0)
grid_model = mt.Grid(U_gN=np.sqrt(2/3)*400, w_g=2*np.pi*50)
dc_model = None
conv = mt.Inverter(u_dc=650)
"""
REMARK:
    if you do not want to simulate any DC bus, you should define
    dc_model = None. This would make the DC voltage constant, using the
    value given in the converter model.
    Do not forget also to activate/desactivate the dc-bus control
"""
    
if dc_model == None:
    mdl = mt.GridCompleteModel(rl_model, grid_model, conv)
else:
    mdl = mt.ACDCGridCompleteModel(rl_model, grid_model, dc_model, conv)

pars = mt.GridFollowingCtrlPars(
            L_f=10e-3,
            R_f=0,
            #C_dc = 1e-3,  # why this is needed if dc-voltage is not controlled? (could there be default value)
            on_v_dc=False,  # what is this?
            S_base = 10e3,
            i_max = 1.5,
            )
ctrl = mt.GridFollowingCtrl(pars)


# %%

# Set the active and reactive power references
# that are inputs to the control system
ctrl.p_g_ref = lambda t: (t > .02)*(5e3)
ctrl.q_g_ref = lambda t: (t > .04)*(4e3)

# AC-voltage magnitude (to simulate voltage dips or short-circuits)
u_g_abs_var =  lambda t: np.sqrt(2/3)*400
mdl.grid_model.u_g_abs_A = u_g_abs_var # a-phase
mdl.grid_model.u_g_abs_B = u_g_abs_var # b-phase
mdl.grid_model.u_g_abs_C = u_g_abs_var # c-phase


# Create the simulation object and simulate it
sim = mt.Simulation(mdl, ctrl, pwm=True)
sim.simulate(t_stop = .1)

# Print the execution time
print('\nExecution time: {:.2f} s'.format((time.time() - start_time)))

# Plot results in per unit values
mt.plot_grid(sim)
#mt.plot_grid(sim, base=base_values)