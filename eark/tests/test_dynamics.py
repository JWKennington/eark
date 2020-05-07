"""Unittests for the inhour module
"""

import numpy as np

from eark import dynamics
from eark.tests import _parameters


class TestDynamicsPopulation:
    def test_total_neutron_deriv(self):
        res = dynamics.total_neutron_deriv(beta=_parameters.BETA,
                                           period=_parameters.PERIOD,
                                           n=_parameters.N_INITIAL,
                                           precursor_constants=_parameters.PRECURSOR_CONSTANTS,
                                           precursor_density=_parameters.PRECURSOR_DENSITY_INITIAL,
                                           rho_fuel_temp=_parameters.RHO_FUEL_TEMP0,
                                           temp_mod=_parameters.T_mod0,
                                           drum_angle=_parameters.THETA_C0)
        np.testing.assert_almost_equal(actual=res, desired=-644418.452096195, decimal=5)

    def test_delay_neutron_deriv(self):
        res = dynamics.delay_neutron_deriv(beta_vector=_parameters.BETA_VECTOR,
                                           period=_parameters.PERIOD,
                                           power=_parameters.PERIOD,
                                           precursor_constants=_parameters.PRECURSOR_CONSTANTS,
                                           precursor_density=_parameters.PRECURSOR_DENSITY_INITIAL)
        desired = np.array([-4039.489, -21301.61586, -20939.66062, -59498.80452, -18187.7548, -6445.90521])
        np.testing.assert_almost_equal(actual=res, desired=desired, decimal=5)


class TestDynamicsThermal:
    def test_mod_temp_deriv(self):
        res = dynamics.mod_temp_deriv(heat_coeff=_parameters.h,
                                      mass_mod=_parameters.M_M,
                                      heat_cap_mod=_parameters.C_M,
                                      mass_flow=_parameters.W_M,
                                      temp_fuel=_parameters.T_fuel0,
                                      temp_mod=_parameters.T_mod0,
                                      temp_in=_parameters.T_in)
        np.testing.assert_almost_equal(actual=res, desired=1282.4, decimal=5)

    def test_fuel_temp_deriv(self):
        res = dynamics.fuel_temp_deriv(power=_parameters.N_INITIAL,
                                       mass_fuel=_parameters.M_F,
                                       heat_cap_fuel=_parameters.C_F,
                                       heat_coeff=_parameters.h,
                                       temp_fuel=_parameters.T_fuel0,
                                       temp_mod=_parameters.T_mod0)
        np.testing.assert_almost_equal(actual=res, desired=-45217.38717391304, decimal=5)


class TestDynamicsDrum:
    def test_omega_drum_deriv(self):
        res = dynamics.drum_angle_deriv(1.0)
        np.testing.assert_almost_equal(actual=res, desired=1.0, decimal=5)


class TestDynamicsReactivity:
    def test_temp_fuel_reactivity_deriv(self):
        res = dynamics.temp_fuel_reactivity_deriv(beta=_parameters.BETA,
                                                  temp_fuel=_parameters.T_fuel0)
        np.testing.assert_almost_equal(actual=res, desired=-1.3007200000000001e-05, decimal=5)

    def test_temp_mod_reactivity(self):
        res = dynamics.temp_mod_reactivity(beta=_parameters.BETA,
                                           temp_mod=_parameters.T_mod0)
        np.testing.assert_almost_equal(actual=res, desired=-0.012634876, decimal=5)

    def test_drum_reactivity(self):
        res = dynamics.drum_reactivity(beta=_parameters.BETA,
                                       drum_angle=_parameters.THETA_C0)
        np.testing.assert_almost_equal(actual=res, desired=-0.028645944000000007, decimal=5)
