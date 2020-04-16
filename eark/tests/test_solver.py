"""Unittests for the inhour module
"""

import numpy as np
import pytest

from eark import solver


class TestInhour:
    @pytest.fixture(scope='class', autouse=True)
    def beta(self):
        return 0.0075

    @pytest.fixture(scope='class', autouse=True)
    def beta_vector(self):
        return np.array([0.033, 0.219, 0.196, 0.395, 0.115, 0.042])

    @pytest.fixture(scope='class', autouse=True)
    def n_initial(self):
        return 4000

    @pytest.fixture(scope='class', autouse=True)
    def period(self):
        return 6.0e-5

    @pytest.fixture(scope='class', autouse=True)
    def precursor_constants(self):
        return np.array([0.0124, 0.0305, 0.1110, 0.3011, 1.1400, 3.0100])

    @pytest.fixture(scope='class', autouse=True)
    def precursor_density(self):
        return np.array([5000, 6000, 5600, 4700, 7800, 6578])

    @pytest.fixture(scope='class', autouse=True)
    def rho(self, beta):
        return 0.5 * beta

    def test_total_neutron_deriv(self, rho, beta, period, n_initial, precursor_constants, precursor_density):
        res = solver.total_neutron_deriv(rho=rho,
                                         beta=beta,
                                         period=period,
                                         n=n_initial,
                                         precursor_constants=precursor_constants,
                                         precursor_density=precursor_density)
        np.testing.assert_almost_equal(actual=res, desired=-219026.44999999998, decimal=5)

    def test_delay_neutron_deriv(self, period, n_initial, beta_vector, precursor_constants, precursor_density):
        res = solver.delay_neutron_deriv(beta_vector=beta_vector,
                                         period=period,
                                         n=n_initial,
                                         precursor_constants=precursor_constants,
                                         precursor_density=precursor_density)
        desired = np.array([2199938., 14599817., 13066045.06667, 26331918.16333, 7657774.66667, 2780200.22])
        np.testing.assert_almost_equal(actual=res, desired=desired, decimal=5)

    def test_solve(self, n_initial, precursor_density, precursor_constants, beta_vector, beta, rho, period):
        C_F = 200  # specific Heat Capacity of Fuel [J/kg/K]
        C_M = 4000  # specific Heat Capacity of Moderator             [J/kg/K]
        h = 4e6  # heat transfer coefficient of fuel and moderator [J/K/sec]
        M_F = 40000  # mass of Fuel                                    [kg]
        M_M = 7000  # mass of Moderator                               [kg]
        N_INITIAL = 1500e6  # initial Reactor Power [W]
        W_M = 8000
        T_in = 550  # inlet coolant temperature [K]
        T_mod0 = T_in + (N_INITIAL / (2 * W_M * C_M))  # initial moderator temperature [K]
        T_fuel0 = T_in + (1 / (2 * W_M * C_M) + (1 / h)) * N_INITIAL  # initial fuel temperature  [K]
        soln = solver.solve(n_initial=n_initial,
                            precursor_density_initial=precursor_density,
                            beta_vector=beta_vector,
                            precursor_constants=precursor_constants,
                            rho=rho,
                            total_beta=beta,
                            period=period,
                            t_max=1,
                            num_iters=3,
                            h=h,
                            M_M=M_M,
                            C_M=C_M,
                            W_M=W_M,
                            M_F=M_F,
                            C_F=C_F,
                            T_in=T_in,
                            T_mod0=T_mod0,
                            T_fuel0=T_fuel0)

        desired = np.array([[4.00000000e+03, 5.00000000e+03, 6.00000000e+03, 5.60000000e+03, 4.70000000e+03, 7.80000000e+03, 6.57800000e+03, 5.73437500e+02, 9.48437500e+02],
                            [1.88124889e+15, 1.85564422e+16, 1.23107336e+17, 1.10019470e+17, 2.20970880e+17, 6.33845771e+16, 2.24123989e+16, 1.08381137e+04, 4.18176945e+06],
                            [2.39687883e+27, 2.36425620e+28, 1.56849723e+29, 1.40174615e+29, 2.81536605e+29, 8.07576031e+28, 2.85553947e+28, 1.30808220e+16, 5.32684469e+18]])
        for a, d in zip(soln.array.ravel().tolist(), desired.ravel().tolist()):
            np.testing.assert_approx_equal(actual=a, desired=d, significant=3)
