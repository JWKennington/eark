"""Unittests for the inhour module
"""

import numpy as np

from eark import solver
from eark.tests import _parameters


class TestSolver:
    def test_solve(self):
        soln = solver.solve(power_initial=_parameters.N_INITIAL,
                            precursor_density_initial=_parameters.PRECURSOR_DENSITY_INITIAL,
                            beta_vector=_parameters.BETA_VECTOR,
                            precursor_constants=_parameters.PRECURSOR_CONSTANTS,
                            total_beta=_parameters.BETA,
                            period=_parameters.PERIOD,
                            heat_coeff=_parameters.h,
                            mass_mod=_parameters.M_M,
                            heat_cap_mod=_parameters.C_M,
                            mass_flow=_parameters.W_M,
                            mass_fuel=_parameters.M_F,
                            heat_cap_fuel=_parameters.C_F,
                            temp_in=_parameters.T_in,
                            temp_mod=_parameters.T_mod0,
                            temp_fuel=_parameters.T_fuel0,
                            rho_fuel_temp=_parameters.RHO_FUEL_TEMP0,
                            drum_angle=_parameters.THETA_C0,
                            omega_drum=_parameters.CDSPD,
                            t_max=3,
                            num_iters=3)

        desired = np.array([[1.00000000e+02, 6.80847024e+04, 1.41191891e+05, 4.01983299e+04, 3.93425092e+04, 2.83476433e+03, 1.55327863e+02, 7.00000000e+02, 2.00000000e+03, 5.41730000e-03, 1.00000000e+02],
                            [1.16848902e+01, 6.69927358e+04, 1.35515007e+05, 3.49263973e+04, 2.64078106e+04, 6.91068051e+02, 1.84934697e+01, 7.09244124e+02, 7.09747955e+02, 5.38759167e-03, 9.85000000e+01],
                            [9.60936881e+00, 6.58825922e+04, 1.29902822e+05, 3.02705702e+04, 1.79449257e+04, 3.42982606e+02, 1.51220783e+01, 6.83814187e+02, 6.84286710e+02, 5.35747792e-03, 9.70000000e+01]])

        for a, d in zip(soln.array.ravel().tolist(), desired.ravel().tolist()):
            np.testing.assert_approx_equal(actual=a, desired=d, significant=3)
