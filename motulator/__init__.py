"""Import simulation environment."""
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=wrong-import-position
from motulator.simulation import Simulation

"""
--------------------------------------------------------------------------
----------------       CONTINUOUS-TIME MODELS      -----------------------
--------------------------------------------------------------------------
"""

# Import motor drive models
from motulator.model.mech import Mechanics
from motulator.model.converter import (
    Inverter,
    FrequencyConverter,
)
from motulator.model.im import (
    InductionMotor,
    InductionMotorSaturated,
    InductionMotorInvGamma,
)
from motulator.model.sm import (
    SynchronousMotor,
    SynchronousMotorSaturated,
)
from motulator.model.im_drive import (
    InductionMotorDrive,
    InductionMotorDriveDiode,
)
from motulator.model.sm_drive import SynchronousMotorDrive

"""
--------------------------------------------------------------------------
"""

# import grid converter models
from motulator.model.grid_filter import (
    LFilter,
    LCLFilter,
)
from motulator.model.grid_VS import (
    Grid,
)
from motulator.model.dc_grid import (
    DcGrid,
)
from motulator.model.grid_connection import (
    GridCompleteModel,
    ACDCGridCompleteModel,
    ACDCGridLCLModel,
)

"""
--------------------------------------------------------------------------
------------------        CONTROL SYSTEMS        -------------------------
--------------------------------------------------------------------------
"""
# Import controllers
from motulator.control.im_vhz import (
    InductionMotorVHzCtrl,
    InductionMotorVHzCtrlPars,
)
from motulator.control.im_vector import (
    InductionMotorVectorCtrl,
    InductionMotorVectorCtrlPars,
)
from motulator.control.sm_vector import (
    SynchronousMotorVectorCtrl,
    SynchronousMotorVectorCtrlPars,
)
from motulator.control.sm_flux_vector import (
    SynchronousMotorFluxVectorCtrl,
    SynchronousMotorFluxVectorCtrlPars,
)
from motulator.control.sm_signal_inj import (
    SynchronousMotorSignalInjectionCtrl,
    SynchronousMotorSignalInjectionCtrlPars,
)

"""
--------------------------------------------------------------------------
"""
#import grid-connected converter controllers
from motulator.control.grid_following import (
    GridFollowingCtrl,
    GridFollowingCtrlPars,
)
from motulator.control.power_synchronization import (
    PSCtrl,
    PSCtrlPars,
)

"""
--------------------------------------------------------------------------
----------------        GENERAL FEATURES        --------------------------
--------------------------------------------------------------------------
"""
# Import other useful stuff
from motulator.helpers import (
    BaseValues,
    BaseValuesElectrical,
    abc2complex,
    complex2abc,
    Sequence,
    Step,
)
"""
--------------------------------------------------------------------------
"""

# Import some default plotting functions
from motulator.plots import (
    plot,
    plot_extra,
    plot_grid,
)

# from motulator.plotting import (
#     plot,
# )

# Delete imported modules
del sys, os
