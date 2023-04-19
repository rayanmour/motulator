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
    U_nom=400, I_nom=18, f_nom=50.0, P_nom=12.5e3)


# %%
# Configure the system model
grid_filter = mt.LFilter(L_f=6e-3, L_g=10e-3, R_g=0)
grid_model = mt.DynGrid(w_N=2*np.pi*50, S_grid=500e3)
conv = mt.Inverter(u_dc=650)

mdl = mt.ACGridLFilterModel(grid_filter, grid_model, conv)
    
pars = mt.GridFollowingCtrlPars(
            L_f=10e-3,
            R_f=0,
            f_sw = 4e3,
            T_s = 1/(8e3),
            I_max = 1.5*(3/2)*base_values.i,
            )
ctrl = mt.GridFollowingCtrl(pars)


# %%

# Set the active power reference
ctrl.p_g_ref = lambda t: ((t > .2)*(6.25e3))

# AC-voltage magnitude (to simulate voltage dips or short-circuits)
e_g_abs_var =  lambda t: np.sqrt(2/3)*400
mdl.grid_model.e_g_abs = e_g_abs_var # grid voltage magnitude

mdl.grid_model.p_e = lambda t: (t > .5)*(.5) # in per units
mdl.grid_model.p_m_ref = lambda t: 0

# Create the simulation object and simulate it
sim = mt.simulation.Simulation(mdl, ctrl, pwm=False)
sim.simulate(t_stop = 10)

# Print the execution time
print('\nExecution time: {:.2f} s'.format((time.time() - start_time)))

# Plot results in SI or per unit values
#mt.plot_grid(sim)
mt.plot_grid(sim, base=base_values,plot_pcc_voltage=True, plot_w=True)
