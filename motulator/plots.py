# pylint: disable=invalid-name
"""Example plotting scripts."""

# %%
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

from motulator.helpers import Bunch, complex2abc

# Plotting parameters
plt.rcParams['axes.prop_cycle'] = cycler(color='brgcmyk')
plt.rcParams['lines.linewidth'] = 1.
plt.rcParams['axes.grid'] = True
plt.rcParams.update({"text.usetex": False})


# %%
def plot(sim, t_span=None, base=None):
    """
    Plot example figures.

    Plots figures in per-unit values, if the base values are given. Otherwise
    SI units are used.

    Parameters
    ----------
    sim : Simulation object
        Should contain the simulated data.
    t_span : 2-tuple, optional
        Time span. The default is (0, sim.ctrl.t[-1]).
    base : BaseValues, optional
        Base values for scaling the waveforms.

    """
    # pylint: disable=too-many-statements
    mdl = sim.mdl.data  # Continuous-time data
    ctrl = sim.ctrl.data  # Discrete-time data

    # Check if the time span was given
    if t_span is None:
        t_span = (0, ctrl.t[-1])

    # Check if the base values were given
    if base is None:
        pu_vals = False
        base = Bunch(w=1, u=1, i=1, psi=1, tau=1)  # Unity base values
    else:
        pu_vals = True

    # Recognize the motor type by checking if the rotor flux data exist
    try:
        if mdl.psi_Rs is not None:
            motor_type = 'im'
    except AttributeError:
        motor_type = 'sm'

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(8, 10))

    # Subplot 1: angular speeds
    ax1.step(ctrl.t, ctrl.w_m_ref/base.w, '--', where='post')
    ax1.plot(mdl.t, mdl.w_m/base.w)
    try:
        ax1.step(ctrl.t, ctrl.w_m/base.w, where='post')
    except AttributeError:
        pass
    ax1.legend([
        r'$\omega_\mathrm{m,ref}$',
        r'$\omega_\mathrm{m}$',
        r'$\hat \omega_\mathrm{m}$',
    ])
    ax1.set_xlim(t_span)
    ax1.set_xticklabels([])

    # Subplot 2: torques
    ax2.plot(mdl.t, mdl.tau_L/base.tau, '--')
    ax2.plot(mdl.t, mdl.tau_M/base.tau)
    try:
        ax2.step(ctrl.t, ctrl.tau_M_ref_lim/base.tau)  # Limited torque ref
    except AttributeError:
        pass
    ax2.legend([
        r'$\tau_\mathrm{L}$',
        r'$\tau_\mathrm{M}$',
        r'$\tau_\mathrm{M,ref}$',
    ])
    ax2.set_xlim(t_span)
    ax2.set_xticklabels([])

    # Subplot 3: currents
    ax3.step(ctrl.t, ctrl.i_s.real/base.i, where='post')
    ax3.step(ctrl.t, ctrl.i_s.imag/base.i, where='post')
    try:
        ax3.step(ctrl.t, ctrl.i_s_ref.real/base.i, '--', where='post')
        ax3.step(ctrl.t, ctrl.i_s_ref.imag/base.i, '--', where='post')
    except AttributeError:
        pass
    ax3.legend([
        r'$i_\mathrm{sd}$',
        r'$i_\mathrm{sq}$',
        r'$i_\mathrm{sd,ref}$',
        r'$i_\mathrm{sq,ref}$',
    ])
    ax3.set_xlim(t_span)
    ax3.set_xticklabels([])

    # Subplot 4: voltages
    ax4.step(ctrl.t, np.abs(ctrl.u_s)/base.u, where='post')
    ax4.step(ctrl.t, ctrl.u_dc/np.sqrt(3)/base.u, '--', where='post')
    ax4.legend([r'$u_\mathrm{s}$', r'$u_\mathrm{dc}/\sqrt{3}$'])
    ax4.set_xlim(t_span)
    ax4.set_xticklabels([])

    # Subplot 5: flux linkages
    if motor_type == 'sm':
        ax5.plot(mdl.t, np.abs(mdl.psi_s)/base.psi)
        try:
            ax5.step(ctrl.t, np.abs(ctrl.psi_s)/base.psi, where='post')
        except AttributeError:
            pass
        ax5.legend([r'$\psi_\mathrm{s}$', r'$\hat\psi_\mathrm{s}$'])
    else:
        ax5.plot(mdl.t, np.abs(mdl.psi_ss)/base.psi)
        ax5.plot(mdl.t, np.abs(mdl.psi_Rs)/base.psi)
        try:
            ax5.plot(ctrl.t, np.abs(ctrl.psi_s)/base.psi)
        except AttributeError:
            pass
        ax5.legend([
            r'$\psi_\mathrm{s}$',
            r'$\psi_\mathrm{R}$',
            r'$\hat \psi_\mathrm{s}$',
        ])
    ax5.set_xlim(t_span)

    # Add axis labels
    if pu_vals:
        ax1.set_ylabel('Speed (p.u.)')
        ax2.set_ylabel('Torque (p.u.)')
        ax3.set_ylabel('Current (p.u.)')
        ax4.set_ylabel('Voltage (p.u.)')
        ax5.set_ylabel('Flux linkage (p.u.)')
    else:
        ax1.set_ylabel('Speed (rad/s)')
        ax2.set_ylabel('Torque (Nm)')
        ax3.set_ylabel('Current (A)')
        ax4.set_ylabel('Voltage (V)')
        ax5.set_ylabel('Flux linkage (Vs)')
    ax5.set_xlabel('Time (s)')
    fig.align_ylabels()

    plt.tight_layout()
    plt.show()


# %%
def plot_extra(sim, t_span=(1.1, 1.125), base=None):
    """
    Plot extra waveforms for a motor drive with a diode bridge.

    Parameters
    ----------
    sim : Simulation object
        Should contain the simulated data.
    t_span : 2-tuple, optional
        Time span. The default is (1.1, 1.125).
    base : BaseValues, optional
        Base values for scaling the waveforms.

    """
    mdl = sim.mdl.data  # Continuous-time data
    ctrl = sim.ctrl.data  # Discrete-time data

    # Check if the base values were iven
    if base is not None:
        pu_vals = True
    else:
        pu_vals = False
        base = Bunch(w=1, u=1, i=1, psi=1, tau=1)  # Unity base values

    # Quantities in stator coordinates
    ctrl.u_ss = np.exp(1j*ctrl.theta_s)*ctrl.u_s
    ctrl.i_ss = np.exp(1j*ctrl.theta_s)*ctrl.i_s

    fig1, (ax1, ax2) = plt.subplots(2, 1)

    # Subplot 1: voltages
    ax1.plot(mdl.t, mdl.u_ss.real/base.u)
    ax1.plot(ctrl.t, ctrl.u_ss.real/base.u)
    ax1.set_xlim(t_span)
    ax1.legend([r'$u_\mathrm{sa}$', r'$\hat u_\mathrm{sa}$'])
    ax1.set_xticklabels([])

    # Subplot 2: currents
    ax2.plot(mdl.t, complex2abc(mdl.i_ss).T/base.i)
    ax2.step(ctrl.t, ctrl.i_ss.real/base.i, where='post')
    ax2.set_xlim(t_span)
    ax2.legend([r'$i_\mathrm{sa}$', r'$i_\mathrm{sb}$', r'$i_\mathrm{sc}$'])

    # Add axis labels
    if pu_vals:
        ax1.set_ylabel('Voltage (p.u.)')
        ax2.set_ylabel('Current (p.u.)')
    else:
        ax1.set_ylabel('Voltage (V)')
        ax2.set_ylabel('Current (A)')
    ax2.set_xlabel('Time (s)')
    fig1.align_ylabels()

    # Plots the DC bus and grid-side variables (if exist)
    try:
        mdl.i_L
    except AttributeError:
        mdl.i_L = None
    if mdl.i_L is not None:

        fig2, (ax1, ax2) = plt.subplots(2, 1)

        # Subplot 1: voltages
        ax1.plot(mdl.t, mdl.u_di/base.u)
        ax1.plot(mdl.t, mdl.u_dc/base.u)
        ax1.plot(mdl.t, complex2abc(mdl.u_g).T/base.u)
        ax1.legend(
            [r'$u_\mathrm{di}$', r'$u_\mathrm{dc}$', r'$u_\mathrm{ga}$'])
        ax1.set_xlim(t_span)
        ax1.set_xticklabels([])

        # Subplot 2: currents
        ax2.plot(mdl.t, mdl.i_L/base.i)
        ax2.plot(mdl.t, mdl.i_dc/base.i)
        ax2.plot(mdl.t, mdl.i_g.real/base.i)
        ax2.legend([r'$i_\mathrm{L}$', r'$i_\mathrm{dc}$', r'$i_\mathrm{ga}$'])
        ax2.set_xlim(t_span)

    # Add axis labels
    if pu_vals:
        ax1.set_ylabel('Voltage (p.u.)')
        ax2.set_ylabel('Current (p.u.)')
    else:
        ax1.set_ylabel('Voltage (V)')
        ax2.set_ylabel('Current (A)')
    ax2.set_xlabel('Time (s)')
    fig2.align_ylabels()

    plt.tight_layout()
    plt.show()

# %%
"""
--------------------------------------------------------------------------
-----------------        GRID APPLICATIONS       -------------------------
--------------------------------------------------------------------------
"""

# %%
def plot_grid(
        sim, t_range=None, base=None, plot_pcc_voltage=False, plot_w=False):
    """
    Plot example figures of grid converter simulations.

    Plots figures in per-unit values, if the base values are given. Otherwise
    SI units are used.

    Parameters
    ----------
    sim : Simulation object
        Should contain the simulated data.
    t_range : 2-tuple, optional
        Time range. The default is (0, sim.ctrl.t[-1]).
    base : BaseValues, optional
        Base values for scaling the waveforms.
    plot_pcc_voltage : Boolean, optional
        'True' if the user wants to plot the 3-phase waveform at the PCC. This
        is an optional feature and the grid voltage is plotted by default.

    """
    FS = 16 # Font size of the plots axis
    FL = 16 # Font size of the legends only
    LW = 3 # Line width in plots
    
    
    mdl = sim.mdl.data      # Continuous-time data
    ctrl = sim.ctrl.data    # Discrete-time data
    
    # Check if the time span was given
    if t_range is None:
        t_range = (0, mdl.t[-1]) # Time span

    # Check if the base values were given
    if base is None:
        pu_vals = False
        # Scaling with unity base values except for power use kW
        base = Bunch(w=1, u=1, i=1, p=1000)
    else:
        pu_vals = True
        
    # 3-phase quantities
    i_g_abc = complex2abc(mdl.i_gs).T
    u_g_abc = complex2abc(mdl.u_gs).T
    e_g_abc = complex2abc(mdl.e_gs).T
    
    # grid voltage magnitude
    abs_e_g = np.abs(mdl.e_gs)
    
    # Calculation of active and reactive powers
    # p_g = 1.5*np.asarray(np.real(ctrl.u_g*np.conj(ctrl.i_c)))
    # q_g = 1.5*np.asarray(np.imag(ctrl.u_g*np.conj(ctrl.i_c)))
    p_g = 1.5*np.asarray(np.real(mdl.u_gs*np.conj(mdl.i_gs)))
    q_g = 1.5*np.asarray(np.imag(mdl.u_gs*np.conj(mdl.i_gs)))
    p_g_ref = np.asarray(ctrl.p_g_ref)
    q_g_ref = np.asarray(ctrl.q_g_ref)
    
    # %%
    fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(10, 7))

    if sim.ctrl.on_v_dc==False:
        if plot_pcc_voltage == False:
            # Subplot 1: Grid voltage
            ax1.plot(mdl.t, e_g_abc/base.u, linewidth=LW)
            ax1.legend([r'$e_g^a$',r'$e_g^b$',r'$e_g^c$'],
                       prop={'size': FL}, loc= 'upper right')
            ax1.set_xlim(t_range)
            ax1.set_xticklabels([])
        else:
            # Subplot 1: PCC voltage
            ax1.plot(mdl.t, u_g_abc/base.u, linewidth=LW)
            ax1.legend([r'$u_g^a$',r'$u_g^b$',r'$u_g^c$'],
                       prop={'size': FL}, loc= 'upper right')
            ax1.set_xlim(t_range)
            ax1.set_xticklabels([])
            #ax1.set_ylabel('Grid voltage (V)')
    else:
        # Subplot 1: DC-bus voltage
        ax1.plot(mdl.t, mdl.u_dc/base.u, linewidth=LW)
        ax1.plot(ctrl.t, ctrl.u_dc_ref/base.u, '--', linewidth=LW)
        ax1.legend([r'$u_{dc}$',r'$u_{dc,ref}$'],
                   prop={'size': FL}, loc= 'upper right')
        ax1.set_xlim(t_range)
        ax1.set_xticklabels([])
        #ax1.set_ylabel('DC-bus voltage (V)')
    
    # Subplot 2: Converter currents
    ax2.plot(mdl.t, i_g_abc/base.i, linewidth=LW)
    ax2.legend([r'$i_g^a$',r'$i_g^b$',r'$i_g^c$']
               ,prop={'size': FL}, loc= 'upper right')
    ax2.set_xlim(t_range)
    ax2.set_xticklabels([])
    
    if plot_w:
        # Subplot 3: Grid and converter frequencies
        ax3.plot(mdl.t, mdl.w_g/base.w, linewidth=LW)
        ax3.plot(ctrl.t, ctrl.w_c/base.w, '--', linewidth=LW)
        ax3.legend([r'$\omega_{g}$',r'$\omega_{c}$']
                   ,prop={'size': FL}, loc= 'upper right')
        ax3.set_xlim(t_range)
    else:
        # Subplot 3: Phase angles
        ax3.plot(mdl.t, mdl.theta, linewidth=LW)
        ax3.plot(ctrl.t, ctrl.theta_c, '--', linewidth=LW)
        ax3.legend([r'$\theta_{g}$',r'$\theta_{c}$']
                   ,prop={'size': FL}, loc= 'upper right')
        ax3.set_xlim(t_range)

    # Add axis labels
    if pu_vals:
        ax1.set_ylabel('Voltage (p.u.)')
        ax2.set_ylabel('Current (p.u.)')
    else:
        ax1.set_ylabel('Voltage (V)')
        ax2.set_ylabel('Current (A)')
    if plot_w==False: 
        ax3.set_ylabel('Angle (rad)')
    elif plot_w and pu_vals:
        ax3.set_ylabel('Frequency (p.u.)')
    elif pu_vals == False:
        ax3.set_ylabel('Frequency (rad/s)')
    ax3.set_xlabel('Time (s)')

    # Change font size
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
             ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(FS)

    for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
             ax2.get_xticklabels() + ax2.get_yticklabels()):
        item.set_fontsize(FS)

    for item in ([ax3.title, ax3.xaxis.label, ax3.yaxis.label] +
             ax3.get_xticklabels() + ax3.get_yticklabels()):
        item.set_fontsize(FS)

    fig.align_ylabels()
    plt.tight_layout()
    plt.grid()
    ax3.grid()
    #plt.show()

    
    # %%
    # Second figure
    fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(10, 7))

    # Subplot 1: Active and reactive power
    ax1.plot(mdl.t, p_g/base.p, linewidth=LW)
    ax1.plot(mdl.t, q_g/base.p, linewidth=LW)
    ax1.plot(ctrl.t, (p_g_ref/base.p), '--', linewidth=LW)
    ax1.plot(ctrl.t, (q_g_ref/base.p), '--', linewidth=LW)
    ax1.legend([r'$p_{g}$',r'$q_{g}$',r'$p_{g,ref}$',r'$q_{g,ref}$'],
               prop={'size': FL}, loc= 'upper right')
    ax1.set_xlim(t_range)
    ax1.set_xticklabels([])
    
    # Subplot 2: Converter currents
    ax2.plot(ctrl.t, np.real(ctrl.i_c/base.i), linewidth=LW)
    ax2.plot(ctrl.t, np.imag(ctrl.i_c/base.i), linewidth=LW)
    ax2.plot(ctrl.t, np.real(ctrl.i_c_ref/base.i), '--', linewidth=LW)
    ax2.plot(ctrl.t, np.imag(ctrl.i_c_ref/base.i), '--', linewidth=LW)
    #ax2.plot(mdl.t, mdl.iL, linewidth=LW) converter-side dc current for debug
    ax2.legend([r'$i_{c}^d$',r'$i_{c}^q$',r'$i_{c,ref}^d$',r'$i_{c,ref}^q$'],
               prop={'size': FL}, loc= 'upper right')
    ax2.set_xlim(t_range)
    ax2.set_xticklabels([])

    # Subplot 3: Converter voltage reference and grid voltage
    ax3.plot(ctrl.t,np.real(ctrl.u_c_ref_lim/base.u), 
            ctrl.t,np.imag(ctrl.u_c_ref_lim/base.u), linewidth=LW)
    ax3.plot(mdl.t,np.real(abs_e_g/base.u),'k--', 
             linewidth=LW)
    ax3.legend([r'$u_{c,ref}^d$', r'$u_{c,ref}^q$', r'$|e_{g}|$'], 
                prop={'size': FS}, loc= 'upper right')
    ax3.set_xlim(t_range)
    
    # Change font size
    for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
             ax2.get_xticklabels() + ax2.get_yticklabels()):
        item.set_fontsize(FS)

    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
             ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(FS)

    for item in ([ax3.title, ax3.xaxis.label, ax3.yaxis.label] +
             ax3.get_xticklabels() + ax3.get_yticklabels()):
        item.set_fontsize(FS)

    # Add axis labels
    if pu_vals:
        ax1.set_ylabel('Power (p.u.)')
        ax2.set_ylabel('Current (p.u.)')
        ax3.set_ylabel('Voltage (p.u.)')
    else:
        ax1.set_ylabel('Power (kW, kVar)')
        ax2.set_ylabel('Current (A)')
        ax3.set_ylabel('Voltage (V)')
    ax3.set_xlabel('Time (s)')

    fig.align_ylabels()
    plt.tight_layout()
    plt.grid()
    ax3.grid()
    plt.show()    



# %%
def save_plot(name):
    """
    Save figures.

    This saves figures in a folder "figures" in the current directory. If the
    folder doesn't exist, it is created.

    Parameters
    ----------
    name : string
        Name for the figure
    plt : object
        Handle for the figure to be saved

    """
    plt.savefig(name + '.pdf')
