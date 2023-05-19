"""
Example simulation script: 12.5-kVA grid-forming controlled converter connected
to an AC grid with electromechanical dynamics through an LCL filter.
    
The control system includes:
    - Reference-feedforward d-axis current reference injection.
    - Power synchronization loop;
    - Inner current controller used to damp the current oscillations.
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
grid_filter = mt.LCLFilter(L_fc=3e-3, C_f=10e-6, L_fg=3e-3, L_g=20e-3, R_g=0)
grid_model = mt.FlexSource(w_N=2*np.pi*50, S_grid=500e3, H_g=2, r_d = 0.05)
dc_model = None
conv = mt.Inverter(u_dc=650)
    
mdl = mt.AcFlexSourceLCLFilterModel(grid_filter, grid_model, conv)

pars = mt.PSCtrlPars(
        L_f=3e-3,
        R_f=0,
        f_sw = 5e3,
        T_s = 1/(10e3),
        on_rf=True,
        on_v_dc=False,
        I_max = 1.5*base_values.i,
        w_0_cc = 2*np.pi*5,
        R_a = .2*base_values.Z)
ctrl = mt.PSCtrl(pars)

# %%

# Set the active power reference
ctrl.p_g_ref = lambda t: ((t > .2)*(6.25e3))

# AC-voltage magnitude (to simulate voltage dips or short-circuits)
e_g_abs_var =  lambda t: np.sqrt(2/3)*400
mdl.grid_model.e_g_abs = e_g_abs_var # grid voltage magnitude

# AC grid electromechanical model
mdl.grid_model.p_e = lambda t: (t > .4)*50e3 # load disturbance in the AC grid
mdl.grid_model.p_m_ref = lambda t: 0 # mechanical power reference

# Create the simulation object and simulate it
sim = mt.simulation.Simulation(mdl, ctrl, pwm=False)
sim.simulate(t_stop = 6)

# Print the execution time
print('\nExecution time: {:.2f} s'.format((time.time() - start_time)))

# Plot results in per unit values
#mt.plot_grid(sim)
mt.plot_grid(sim, plot_pcc_voltage=True, plot_w = True)
