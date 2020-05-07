import numpy as np

from eark.state import StateComponent
from eark.utilities import plot


class Solution:
    __slots__ = ('_array', '_t')

    def __init__(self, array: np.ndarray, t: np.ndarray):
        self._array = array
        self._t = t

    @property
    def array(self):
        return self._array

    @property
    def t(self):
        return self._t

    @property
    def neutron_population(self):
        return self._array[:, StateComponent.NeutronPopulation]

    @property
    def precursor_densities(self):
        return self._array[:, StateComponent.PrecursorDensity1:StateComponent.TMod]

    def precursor_density(self, i: int):
        """Get a timeseries of precursor densitity of the ith kind

        Args:
            i:
                index of the vector component. These are 1-indexed, to match the mathematical notation.

        Returns:
            ndarray
        """
        return self.precursor_densities[:, i - 1]

    @property
    def temp_mod(self):
        return self._array[:, StateComponent.TMod]

    @property
    def temp_fuel(self):
        return self._array[:, StateComponent.TFuel]

    @property
    def rho_fuel_temp(self):
        return self._array[:, StateComponent.RhoFuelTemp]

    @property
    def drum_angle(self):
        return self._array[:, StateComponent.DrumAngle]

    def plot_power(self):
        plot.plot_soln_quantity(t=self.t, y=self.neutron_population, label='$P(t)$', y_label='Power', title='Power v. Time')

    def plot_densities(self):
        plot.plot_soln_quantity(t=self.t, y=[self.precursor_density(i) for i in range(1, 7)], label=lambda i: '$c_{:d}$'.format(i),
                                color=plot.DENSITY_COLORS,
                                title="Concentration of Neutron Precursors vs. Time",
                                y_label="Concentration of Neutron Precursors, $c_i [\#/dr^3]$")

    def plot_temp_fuel(self):
        plot.plot_soln_quantity(t=self.t, y=self.temp_fuel, y_label="Fuel  Temperature [K]", title='Fuel Temperature vs. Time')

    def plot_temp_mod(self):
        plot.plot_soln_quantity(t=self.t, y=self.temp_mod, y_label="Moderator Temperature [K]", title='Moderator Temperature vs. Time')

