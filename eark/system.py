import os
import sys
from eark.utilities.propertytable import PropertyTable, PropertyData

table = PropertyTable("/Users/vignesh.manickam/repos/eark/eark/utilities/ThermoPhysicalProperties.h5")
H2 = table.read("H2")
H2.evaluate("h", pressure=0.2, temperature=15)

def friction_factor(reynolds_bulk: float):
    """
        Args:
            reynolds_bulk:
                float, Reynold's Number for Bulk Fluid                     []

    References:
    [1] Correlation of friction coefficients for laminar and turbulent flow with ratios of surface to bulk temperature
    from 0.35 to 7.35,, NASA-TR-R-267
    """

    return 0.25 * (0.0345 + (363 / reynolds_bulk ** 1.25))


def pressure_pipe_drop(p_in: float, reynolds_bulk: float, l: float, d: float, temp_in: float, temp_out: float,
                       velocity_in: float, velocity_out: float):
    """
        Args:
            reynolds_bulk:
                float, Reynold's Number for Bulk Fluid                 []
            p_in:
                float, Pressure at initial point                       [Mpa]
            l:
                float, length of pipe                                  [m]
            d:
                float, hydraulic diameter                              [m]
            temp_in:
                float, temperature of fluid at initial point           [K]
            temp_out:
                float, temperature of fluid at exit point              [K]
            velocity_in:
                float, velocity of fluid at initial point              [m/s]
            velocity_out:
                float, velocity of fluid at exit point                 [m/s]


            Returns:
                p_out:
                    float, Pressure at exit point                      [MPa]
        """

    density_in = H2.evaluate("r", pressure=p_in, temperature=temp_in)
    density_out = H2.evaluate("r", temperature=temp_out)

    return p_in - (friction_factor(reynolds_bulk) * l / ((d * density_in * velocity_in ** 2) / 2)) - \
           (density_in * velocity_in ** 2 - density_out * velocity_out ** 2)

def pump_pressure(p_in: float, temp_in: float, head: float):
    """
        Args:
            p_in:
                float, Pressure at initial point                       [Mpa]
            temp_in:
                float, temperature of fluid at initial point           [K]
            head:
                float, head of the pump                                [m]

        Returns:
            p_out:
                float, Pressure at exit point                          [MPa]
    """

    density_in = H2.evaluate("r", pressure=p_in, temperature=temp_in)

    return density_in * head + p_in

def pump_temp(p_in: float, temp_in: float, pump_power: float, head: float, pump_mass_flow: float ):
    """
        Args:
            p_in:
                float, Pressure at initial point                       [Mpa]
            temp_in:
                float, temperature of fluid at initial point           [kg/m^3]
            head:
                    float, head of the pump                            [m]

        Returns:
            temp_out:
                float, temperature at exit point                       [K]
    """

    H2_pump_heat_cap = H2.evaluate("cp", pressure=p_in, temperature=temp_in)
    pump_n = head / (pump_power * pump_mass_flow)

    return temp_in + (pump_power * pump_n) / (pump_mass_flow * H2_pump_heat_cap)
