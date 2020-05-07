"""Unittests for the inhour module
"""
import pathlib
import tempfile

import numpy as np
import pytest

from eark import solver
from eark.utilities import plot


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

    def test_plot_solution(self, n_initial, precursor_density, precursor_constants, beta_vector, beta, rho, period):
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
        with tempfile.TemporaryDirectory() as tmpdir:
            # find the temporary path
            plot_file = pathlib.Path(tmpdir) / 'plot_file.png'
            assert not plot_file.exists()

            # Plot solution
            plot.plot_solution(soln, show_densities=False, output_file=plot_file.as_posix())

            # Check image was written
            assert plot_file.exists()
