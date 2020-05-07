"""This module defines the "state" of the reactor as used by the numerical integrator in solver.py
"""
import enum

import numpy as np


class StateComponent(enum.IntEnum):
    """StateComponent enumeration defines the array-index components of the state vector

    """
    NeutronPopulation = 0
    PrecursorDensity1 = 1
    PrecursorDensity2 = 2
    PrecursorDensity3 = 3
    PrecursorDensity4 = 4
    PrecursorDensity5 = 5
    PrecursorDensity6 = 6
    TMod = 7
    TFuel = 8
    RhoFuelTemp = 9
    RhoModTemp = 10
    DrumAngle = 11


class State:
    __slots__ = ('neutron_population', 'precursor_densities', 't_mod', 't_fuel', 'rho_fuel_temp', 'rho_mod_temp', 'drum_angle')

    def __init__(self, neutron_population: float, precursor_densities: np.ndarray, t_mod: float, t_fuel: float,
                 rho_fuel_temp: float, rho_mod_temp:float, theta_c: float):
        """[TBD]

        Args:
            neutron_population:
            precursor_densities:
            t_mod:
            t_fuel:
            rho_fuel_temp:
            rho_mod_temp:
            theta_c:
        """
        self.neutron_population = neutron_population
        self.precursor_densities = precursor_densities
        self.t_mod = t_mod
        self.t_fuel = t_fuel
        self.rho_fuel_temp = rho_fuel_temp
        self.rho_mod_temp = rho_mod_temp
        self.drum_angle = theta_c

    def to_array(self):
        return np.concatenate((np.array([self.neutron_population]),
                               self.precursor_densities,
                               np.array([self.t_mod,
                                         self.t_fuel,
                                         self.rho_fuel_temp,
                                         self.rho_mod_temp,
                                         self.drum_angle])), axis=0)

    @staticmethod
    def from_array(state_array: np.ndarray):
        return State(neutron_population=state_array[StateComponent.NeutronPopulation],
                     precursor_densities=state_array[StateComponent.PrecursorDensity1:StateComponent.TMod],
                     t_mod=state_array[StateComponent.TMod],
                     t_fuel=state_array[StateComponent.TFuel],
                     rho_fuel_temp=state_array[StateComponent.RhoFuelTemp],
                     rho_mod_temp=state_array[StateComponent.RhoModTemp],
                     theta_c=state_array[StateComponent.DrumAngle])
