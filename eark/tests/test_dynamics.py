"""Unittests for the inhour module
"""

import numpy as np

from eark import dynamics
from eark.tests import _parameters
from eark.utilities import testing


class TestDynamicsPopulation:
    def test_total_neutron_deriv(self):
        res = dynamics.total_neutron_deriv(beta=_parameters.BETA,
                                           period=_parameters.PERIOD,
                                           power=_parameters.POWER_INITIAL,
                                           precursor_constants=_parameters.PRECURSOR_CONSTANTS,
                                           precursor_density=_parameters.PRECURSOR_DENSITY_INITIAL,
                                           rho_fuel_temp= dynamics.temp_fuel_reactivity(beta=_parameters.BETA, temp_fuel=_parameters.TEMP_FUEL_INITIAL),
                                           rho_mod_temp= dynamics.temp_mod_reactivity(beta=_parameters.BETA, temp_mod=_parameters.TEMP_MOD_INITIAL),
                                           rho_con_drum= dynamics.con_drum_reactivity(beta=_parameters.BETA, drum_angle=_parameters.DRUM_ANGLE_INITIAL))
        np.testing.assert_almost_equal(actual=res, desired=15502168.996537209, decimal=5)

    def test_delay_neutron_deriv(self):
        res = dynamics.delay_neutron_deriv(beta_vector=_parameters.BETA_VECTOR,
                                           period=_parameters.PERIOD,
                                           power=_parameters.PERIOD,
                                           precursor_constants=_parameters.PRECURSOR_CONSTANTS,
                                           precursor_density=_parameters.PRECURSOR_DENSITY_INITIAL)
        desired = np.array([-2.12605e+08, -1.12114e+09, -1.10209e+09, -3.13152e+09, -9.57250e+08, -3.39258e+08])
        for r, d in zip(res, desired):
            np.testing.assert_approx_equal(actual=r, desired=d, significant=5)


class TestDynamicsThermal:
    def test_mod_temp_deriv(self):
        res = dynamics.mod_temp_deriv(heat_coeff=_parameters.HEAT_COEFF,
                                      mass_mod=_parameters.MASS_MOD,
                                      heat_cap_mod=_parameters.HEAT_CAP_MOD,
                                      mass_flow=_parameters.MASS_FLOW,
                                      temp_fuel=_parameters.TEMP_FUEL_INITIAL,
                                      temp_mod=_parameters.TEMP_MOD_INITIAL,
                                      temp_in=_parameters.TEMP_IN)
        np.testing.assert_almost_equal(actual=res, desired=1.7763568394002505e-15, decimal=5)

    def test_fuel_temp_deriv(self):
        res = dynamics.fuel_temp_deriv(power=_parameters.POWER_INITIAL,
                                       mass_fuel=_parameters.MASS_FUEL,
                                       heat_cap_fuel=_parameters.HEAT_CAP_FUEL,
                                       heat_coeff=_parameters.HEAT_COEFF,
                                       temp_fuel=_parameters.TEMP_FUEL_INITIAL,
                                       temp_mod=_parameters.TEMP_MOD_INITIAL)
        np.testing.assert_almost_equal(actual=res, desired=2.842170943040401e-14, decimal=5)


class TestDynamicsReactivity:
    def test_temp_fuel_reactivity_deriv(self):
        res = dynamics.temp_fuel_reactivity_deriv(power=_parameters.POWER_INITIAL,
                                                  beta=_parameters.BETA,
                                                  mass_fuel= _parameters.MASS_FUEL,
                                                  heat_cap_fuel= _parameters.HEAT_CAP_FUEL,
                                                  heat_coeff= _parameters.HEAT_COEFF,
                                                  temp_fuel= _parameters.TEMP_FUEL_INITIAL,
                                                  temp_mod= _parameters.TEMP_MOD_INITIAL)
        np.testing.assert_almost_equal(actual=res, desired=-1.3007200000000001e-5, decimal=5)

    def test_temp_mod_reactivity(self):
        res = dynamics.temp_mod_reactivity_deriv(beta=_parameters.BETA,
                                                 heat_coeff= _parameters.HEAT_COEFF,
                                                 mass_mod= _parameters.MASS_MOD,
                                                 heat_cap_mod=_parameters.HEAT_CAP_MOD,
                                                 mass_flow= _parameters.MASS_FLOW,
                                                 temp_fuel=_parameters.TEMP_FUEL_INITIAL,
                                                 temp_mod=_parameters.TEMP_MOD_INITIAL,
                                                 temp_in=_parameters.TEMP_IN)
        np.testing.assert_almost_equal(actual=res, desired=-1.9701184522153366e-20, decimal=5)

    def test_drum_reactivity_deriv(self):
        res = dynamics.con_drum_reactivity_deriv(beta=_parameters.BETA,
                                                 drum_speed=1.0,
                                                 drum_angle=_parameters.DRUM_ANGLE_INITIAL)
        np.testing.assert_almost_equal(actual=res, desired=-0.0008849438052825002, decimal=5)
