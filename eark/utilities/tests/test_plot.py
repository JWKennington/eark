"""Unittests for the inhour module
"""
import pathlib
import tempfile

from eark import solver
from eark.tests import _parameters


class TestPlot:

    def test_plot_solution(self):
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
        with tempfile.TemporaryDirectory() as tmpdir:
            # find the temporary path
            plot_file = pathlib.Path(tmpdir) / 'plot_file.png'
            assert not plot_file.exists()

            # Plot solution
            soln.plot_power(output_file=plot_file.as_posix())

            # Check image was written
            assert plot_file.exists()
