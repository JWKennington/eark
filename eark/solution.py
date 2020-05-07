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
    def T_mod(self):
        return self._array[:, StateComponent.TMod]

    @property
    def T_fuel(self):
        return self._array[:, StateComponent.TFuel]

    @property
    def rho_fuel_temp(self):
        return self._array[:, StateComponent.RhoFuelTemp]

    @property
    def theta_c(self):
        return self._array[:, StateComponent.DrumAngle]

    def plot_power(self):
        plot.plot_soln_quantity(t=self.t, y=self.neutron_population, label='$P(t)$', y_label='Power', title='Power v. Time')
