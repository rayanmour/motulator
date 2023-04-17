"""
Example simulation script: 6.9kVA grid-forming controlled converter connected
to a perfect AC voltage source (grid) with an LCL filter.
    
The control system includes:
    - Power synchronization loop;
    - Inner current controller used to damp the current oscillations.
    - DC-bus controller (optional)
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
# Configure the system model (grid model)
rl_model = mt.LCLGrid(L_fc = 3e-3, C_f = 10e-6, L_fg = 3e-3, L_g=30e-3, R_g=0)
grid_model = mt.Grid(w_N=2*np.pi*50)
dc_model = None
conv = mt.Inverter(u_dc=650)

mdl = mt.ACDCGridLCLModel(rl_model, grid_model, dc_model, conv)

pars = mt.PSCtrlPars(
        L_f=3e-3,
        R_f=0,
        f_sw = 4e3,
        T_s = 1/(8e3),
        on_rf=False,
        on_v_dc=False,
        I_max = 1.5*(3/2)*base_values.i,
        w_0_cc = 2*np.pi*5,
        R_a = .2*base_values.Z)
ctrl = mt.PSCtrl(pars)

# %%

# Set the active power reference
ctrl.p_g_ref = lambda t: ((t > .2)*(4.17e3) + (t > .5)*(4.17e3) + 
    (t > .8)*(4.17e3) - (t > 1.2)*(12.5e3))

# AC-voltage magnitude (to simulate voltage dips or short-circuits)
e_g_abs_var =  lambda t: np.sqrt(2/3)*400
mdl.grid_model.e_g_abs = e_g_abs_var # grid voltage magnitude

# Create the simulation object and simulate it
sim = mt.simulation.Simulation(mdl, ctrl, pwm=False)
sim.simulate(t_stop = 1.5)

# Print the execution time
print('\nExecution time: {:.2f} s'.format((time.time() - start_time)))

# Plot results in per unit values
#mt.plot_grid(sim)
mt.plot_grid(sim, base=base_values, plot_pcc_voltage=True)