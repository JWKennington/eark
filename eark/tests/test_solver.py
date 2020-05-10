"""Unittests for the inhour module
"""

import numpy as np

from eark import solver
from eark.tests import _parameters
from eark.utilities import testing


class TestSolver:
    def test_solve(self):
        soln = solver.solve(power_initial=_parameters.POWER_INITIAL,
                            precursor_density_initial=_parameters.PRECURSOR_DENSITY_INITIAL,
                            beta_vector=_parameters.BETA_VECTOR,
                            precursor_constants=_parameters.PRECURSOR_CONSTANTS,
                            total_beta=_parameters.BETA,
                            period=_parameters.PERIOD,
                            heat_coeff=_parameters.HEAT_COEFF,
                            mass_mod=-_parameters.MASS_MOD,
                            heat_cap_mod=_parameters.HEAT_CAP_MOD,
                            mass_flow=_parameters.MASS_FLOW,
                            mass_fuel=_parameters.MASS_FUEL,
                            heat_cap_fuel=_parameters.HEAT_CAP_FUEL,
                            temp_in=_parameters.TEMP_IN,
                            temp_mod_initial=_parameters.TEMP_MOD_INITIAL,
                            temp_fuel_initial=_parameters.TEMP_FUEL_INITIAL,
                            drum_control_rule=_parameters.DRUM_SPEED,
                            drum_angle_initial=_parameters.DRUM_ANGLE_INITIAL,
                            t_max=3,
                            num_iters=3)

        desired = np.array([[2.5e+07, 1.70212e+10, 3.5298e+10, 1.00496e+10, 9.83563e+09, 7.08691e+08, 3.8832e+07,
                             4.42045e+02, 4.48295e+02, -4.73217e-03, -3.90459e-04, 6.465e+01, 5.0077e-03],
                            [2.50869e+07, 1.70221e+10, 3.53028e+10, 1.00540e+10, 9.84668e+09, 7.10599e+08, 3.89639e+07,
                             4.42017e+02, 4.48289e+02, -4.73204e-03, -3.90145e-04, 6.465e+01, 5.0077e-03],
                            [2.51124e+07, 1.70233e+10, 3.53091e+10, 1.00595e+10, 9.85752e+09, 7.11494e+08, 3.90035e+07,
                             4.41976e+02, 4.48254e+02, -4.73129e-03, -3.89684e-04, 6.465e+01, 5.0077e-03]])

        testing.assert_array_approx_equal(soln.array, desired, significant=5)
