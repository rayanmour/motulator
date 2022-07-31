# pylint: disable=invalid-name
"""
Continuous-time models for induction motors.

Peak-valued complex space vectors are used. The space vector models are
implemented in rotor coordinates. The default values correspond to a 2.2-kW
induction motor.

"""
import numpy as np

from motulator.helpers import complex2abc


# %%
class InductionMotor:
    """
    Γ-equivalent model of an induction motor.

    An induction motor is modeled using the Γ-equivalent model [1]_. The model
    is implemented in rotor coordinates.

    Parameters
    ----------
    p : int
        Number of pole pairs.
    R_s : float
        Stator resistance.
    R_r : float
        Rotor resistance.
    L_ell : float
        Leakage inductance.
    L_s : float
        Stator inductance.

    Notes
    -----
    The Γ model is chosen here since it can be extended with the magnetic
    saturation model in a staightforward manner. If the magnetic saturation is
    omitted, the Γ model is mathematically identical to the inverse-Γ and T
    models [1]_. Rotor coordinates are chosen since this choice also allows
    extending the model with deep-bar model in a straightforward manner.

    References
    ----------
    .. [1] Slemon, "Modelling of induction machines for electric drives," IEEE
       Trans. Ind. Appl., 1989, https://doi.org/10.1109/28.44251.

    """

    def __init__(self, p=2, R_s=3.7, R_r=2.5, L_ell=.023, L_s=.245, mech=None):
        # pylint: disable=too-many-arguments
        self.p = p
        self.R_s, self.R_r = R_s, R_r
        self.L_ell, self.L_s = L_ell, L_s
        # Initial values
        self.psi_s0, self.psi_r0 = 0j, 0j
        # For the coordinate transformation
        self._mech = mech

    def currents(self, psi_s, psi_r):
        """
        Compute the stator and rotor currents.

        Parameters
        ----------
        psi_s : complex
            Stator flux linkage.
        psi_r : complex
            Rotor flux linkage.

        Returns
        -------
        i_s : complex
            Stator current.
        i_r : complex
            Rotor current.

        """
        i_r = (psi_r - psi_s)/self.L_ell
        i_s = psi_s/self.L_s - i_r
        return i_s, i_r

    def torque(self, psi_s, i_s):
        """
        Compute the electromagnetic torque.

        Parameters
        ----------
        psi_s : complex
            Stator flux linkage.
        i_s : complex
            Stator current.

        Returns
        -------
        tau_M : float
            Electromagnetic torque.

        """
        tau_M = 1.5*self.p*np.imag(i_s*np.conj(psi_s))
        return tau_M

    def f(self, psi_s, psi_r, u_s, w_M):
        """
        Compute the state derivatives.

        Parameters
        ----------
        psi_s : complex
            Stator flux linkage.
        psi_r : complex
            Rotor flux linkage.
        u_s : complex
            Stator voltage.
        w_M : float
            Rotor angular speed (in mechanical rad/s).

        Returns
        -------
        complex list, length 2
            Time derivative of the state vector, [dpsi_s, dpsi_r]

        """
        i_s, i_r = self.currents(psi_s, psi_r)
        dpsi_s = u_s - self.R_s*i_s - 1j*self.p*w_M*psi_s
        dpsi_r = -self.R_r*i_r
        return [dpsi_s, dpsi_r]

    def meas_currents(self):
        """
        Measure the phase currents at the end of the sampling period.

        Returns
        -------
        i_s_abc : 3-tuple of floats
            Phase currents.

        """
        # Current space vector in rotor coordinates
        i_s0, _ = self.currents(self.psi_s0, self.psi_r0)
        theta_m0 = self.p*self._mech.theta_M0
        # Current space vector is stator coordinates
        i_ss0 = np.exp(1j*theta_m0)*i_s0
        i_s_abc = complex2abc(i_ss0)  # + noise + offset ...
        return i_s_abc


# %%
class InductionMotorSaturated(InductionMotor):
    """
    Γ-equivalent model of an induction motor model with main-flux saturation.

    This extends the InductionMotor class with a main-flux magnetic saturation
    model [2]_::

        L_s(psi_ss) = L_su/(1 + (beta*abs(psi_ss)**S)

    Parameters
    ----------
    p : int
        Number of pole pairs.
    R_s : float
        Stator resistance.
    R_r : float
        Rotor resistance.
    L_ell : float
        Leakage inductance.
    L_su : float
        Unsaturated stator inductance.
    beta : float
        Positive coefficient.
    S : float
        Positive coefficient.

    References
    ----------
    .. [2] Qu, Ranta, Hinkkanen, Luomi, "Loss-minimizing flux level control of
       induction motor drives," IEEE Trans. Ind. Appl., 2012,
       https://doi.org/10.1109/TIA.2012.2190818

    """

    def __init__(
            self, p=2, R_s=3.7, R_r=2.5, L_ell=.023, L_su=.34, beta=.84, S=7):
        # pylint: disable=too-many-arguments
        super().__init__(p=p, R_s=R_s, R_r=R_r, L_ell=L_ell)
        # Saturation model
        self.L_s = lambda psi: L_su/(1. + (beta*np.abs(psi))**S)

    def currents(self, psi_ss, psi_rs):
        """Override the base class method."""
        # Saturated value of the stator inductance.
        L_s = self.L_s(psi_ss)
        # Currents
        i_rs = (psi_rs - psi_ss)/self.L_ell
        i_ss = psi_ss/L_s - i_rs
        return i_ss, i_rs


# %%
class InductionMotorInvGamma(InductionMotor):
    """
    Inverse-Γ model of an induction motor.

    This extends the InductionMotor class (based on the Γ model) by providing
    the interface for the inverse-Γ model parameters. Linear magnetics are
    assumed. If magnetic saturation is to be modeled, the Γ model is preferred.

    Parameters
    ----------
    p : int
        Number of pole pairs.
    R_s : float
        Stator resistance.
    R_R : float
        Rotor resistance.
    L_sgm : float
        Leakage inductance.
    L_M : float
        Magnetizing inductance.

    """

    def __init__(self, p=2, R_s=3.7, R_R=2.1, L_sgm=.021, L_M=.224):
        # pylint: disable=too-many-arguments, disable=super-init-not-called
        # Convert the inverse-Γ parameters to the Γ parameters
        gamma = L_M/(L_M + L_sgm)  # Magnetic coupling factor
        self.p = p
        self.R_s = R_s
        self.L_s = L_M + L_sgm
        self.L_ell = L_sgm/gamma
        self.R_r = R_R/gamma**2
        # Initial values
        self.psi_s0 = 0j
        self.psi_r0 = 0j  # self.psi_rs0 = self.psi_Rs0/gamma
