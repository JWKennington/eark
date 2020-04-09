"""Unittests for the inhour module
"""
import pathlib
import tempfile

import numpy as np
import pytest

from eark import inhour, plot


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
        soln = inhour.solve(n_initial=n_initial,
                            precursor_density_initial=precursor_density,
                            beta_vector=beta_vector,
                            precursor_constants=precursor_constants,
                            rho=rho,
                            total_beta=beta,
                            period=period,
                            t_max=10,
                            num_iters=30000)
        with tempfile.TemporaryDirectory() as tmpdir:
            # find the temporary path
            plot_file = pathlib.Path(tmpdir) / 'plot_file.png'
            assert not plot_file.exists()

            # Plot solution
            plot.plot_solution(soln, show_densities=False, output_file=plot_file.as_posix())

            # Check image was written
            assert plot_file.exists()
