"""Unittests for the inhour module
"""

import numpy as np

from eark import solver
from eark.tests import _parameters


class TestSolver:
    def test_solve(self):
        soln = solver.solve(power_initial=_parameters.POWER_INITIAL,
                            precursor_density_initial=_parameters.PRECURSOR_DENSITY_INITIAL,
                            beta_vector=_parameters.BETA_VECTOR,
                            precursor_constants=_parameters.PRECURSOR_CONSTANTS,
                            total_beta=_parameters.BETA,
                            period=_parameters.PERIOD,
                            heat_coeff=_parameters.HEAT_COEFF,
                            mass_mod=_parameters.MASS_MOD,
                            heat_cap_mod=_parameters.HEAT_CAP_MOD,
                            mass_flow=_parameters.MASS_FLOW,
                            mass_fuel=_parameters.MASS_FUEL,
                            heat_cap_fuel=_parameters.HEAT_CAP_FUEL,
                            temp_in=_parameters.TEMP_IN,
                            temp_mod=_parameters.TEMP_MOD_INITIAL,
                            temp_fuel=_parameters.TEMP_FUEL_INITIAL,
                            rho_fuel_temp=_parameters.RHO_FUEL_TEMP_INITIAL,
                            rho_mod_temp_initial=_parameters.RHO_MOD_TEMP_INITIAL,
                            drum_control_rule=_parameters.DRUM_SPEED,
                            drum_angle_initial=_parameters.DRUM_ANGLE_INITIAL,
                            drum_angle=_parameters.THETA_C0,
                            omega_drum=_parameters.CDSPD,
                            t_max=3,
                            num_iters=3)

        desired = np.array([[4.75000000e+02, 3.23402336e+05, 6.70661482e+05, 1.90942067e+05, 1.86876919e+05, 1.34651306e+04,
                             7.37807351e+02, 7.00000000e+02, 2.00000000e+03, 5.41730000e-03, 1.00000000e+02],
                            [5.55032278e+01, 3.18215495e+05, 6.43696286e+05, 1.65900387e+05, 1.25437100e+05, 3.28257323e+03,
                             8.78439807e+01, 7.09244142e+02, 7.09747983e+02, 5.38759167e-03, 9.85000000e+01],
                            [4.56445015e+01, 3.12942313e+05, 6.17038406e+05, 1.43785208e+05, 8.52383969e+04, 1.62916739e+03,
                             7.18298715e+01, 6.83814217e+02, 6.84286749e+02, 5.35747792e-03, 9.70000000e+01]])

        for a, d in zip(soln.array.ravel().tolist(), desired.ravel().tolist()):
            np.testing.assert_approx_equal(actual=a, desired=d, significant=3)
