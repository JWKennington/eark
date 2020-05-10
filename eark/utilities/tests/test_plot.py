"""Unittests for the inhour module
"""
import pathlib
import tempfile

from eark import solver
from eark.tests import _parameters


class TestPlot:

    def test_plot_solution(self):
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
        with tempfile.TemporaryDirectory() as tmpdir:
            # find the temporary path
            plot_file = pathlib.Path(tmpdir) / 'plot_file.png'
            assert not plot_file.exists()

            # Plot solution
            soln.plot_power(output_file=plot_file.as_posix())

            # Check image was written
            assert plot_file.exists()
