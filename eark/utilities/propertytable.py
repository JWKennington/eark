"""propertytable

Interface into pre-generated mono- or multi-dimensional property tables.

Provides a very simple look-up table like interface into standard
thermophysical tables. Assumes a specific structure of the data stored in a
hierarchical data format (HDF) file. Is designated to be used for the purpose
of NTP thermal analyses.

Created on Mon Mar  9 19:27:04 2020 @author: Dan Kotlyar
Last updated on Sat April 10 10:30:00 2020 @author: Dan Kotlyar
Note: heavily relies on the xsteam function written by Andrew Johnson

"""

import pathlib
import numbers
import bisect
from collections import namedtuple

import matplotlib.pyplot as plt
import h5py
import numpy

Units = namedtuple("Units", ["name", "units"])


def mixbinary(val1, val2, w1, method):
    """Evaluate a weighted value for a binary mixture

    The method evaluates the weighted property given two values that represent
    the same property for two materials given a certain weight fraction
    at a specific temperature and/or pressure.


    Parameters
    ----------
    val1 : float
        value for material of type 1, e.g. thermal conductivity for tungsten
    val2 : float
        value for material of type 2, e.g. thermal conductivity for molybdenum
    w1 : float
        weight fraction of material-1
    method : {"Parallel", "Series", "Bruggeman-Fricke"}
        weighting method, e.g. "Parallel"

    Returns
    -------
    float
        value of the weighted property for the mixture

    Raises
    ------
    ValueError
        If any of the ``val1``, ``val2``, or ``w1`` are not properly defined,
        e.g. negative numbers.
        If the given ``method`` does not exist.
    TypeError
        Any of the values, e.g. ``val1`` is not a float type.

    Note
    ----
    The Bruggeman-Fricke weighting method assumes that ``val1`` represent
    the particles, ``val2`` represent the matrix, and ``w1`` is the volume
    fraction of the particles in the matrix.


    Examples
    --------
    >>> mixbinary(val1= 1.7, val2= 2.5, w1= 0.01, method="Parallel")
    2.488290398126464
    >>> mixbinary(val1= 1.7, val2= 2.5, w1= 0.01, method="Serial")
    2.492

    """

    # list of all the weighting methods
    methodsList = ["Serial", "Parallel", "Bruggeman-Fricke"]

    if method not in methodsList:
        raise ValueError("method= {} does not exist. Available methods are {}"
                         .format(method, methodsList))

    if not isinstance(val1, numbers.Real):
        raise TypeError("val1 must be of type float and not {}".format(val1))
    if not isinstance(val2, numbers.Real):
        raise TypeError("val1 must be of type float and not {}".format(val2))
    if not isinstance(w1, numbers.Real):
        raise TypeError("w1 must be of type float and not {}".format(w1))

    if val1 <= 0:
        raise ValueError("val1={} must be positive".format(val1))
    if val2 <= 0:
        raise ValueError("val2={} must be positive".format(val2))
    if not 0 < w1 < 1:
        raise ValueError("Value of w1={} must be between 0 and 1".format(w1))

    if method == "Serial":
        return w1*val1 + (1 - w1)*val2
    if method == "Parallel":
        return 1 / (w1 / val1 + (1 - w1) / val2)
    if method == "Bruggeman-Fricke":
        return val1 + (1 - w1)*(val2 - val1)*(val1 / val2)**(1 / 3)


class PropertyTable:
    """Interface into the provided data file

    Methods
    -------
    read : read the HDF5 data set for a specific material
    getref : the reference for the material database
    materials: get a list of all the materials in the database

    Parameters
    ----------
    h5path : Union[str, pathlib.Path, h5py.File, h5py.Group]
        Potentially supports passing an opened File object
        or Group directly, but best to just provide the file name

    Raises
    ------
    OSError
        If the ``h5path`` is not valid.

    Examples
    --------
    >>> table = PropertyTable("ThermoPhysicalProperties.h5")

    """

    def __init__(self, h5path):
        if isinstance(h5path, (str, pathlib.Path)):
            self._h5 = h5py.File(h5path, "r")
        elif not isinstance(h5path, (h5py.File, h5py.Group)):
            raise TypeError(h5path)
        else:
            self._h5 = h5path

    def materials(self):
        """Obtain all the materials in the datafile

        Returns
        -------
        list
            All the materials in the database

        Raises
        ------
        KeyError
            If the data file has no materials.

        Examples
        --------
        >>> table = PropertyTable("ThermoPhysicalProperties.h5)
        >>> table.materials()
        ['H2', 'Molybdenum', 'Tungsten', 'UC', 'UN', 'UO2', 'Zircaloy', 'ZrC']

        """

        allmat = self._h5.keys()
        if not allmat:
            raise KeyError("No materials in the current database")
        return list(allmat)

    def read(self, mat):
        """Obtain a complete data set for a specific material

        Parameters
        ----------
        mat : string
            name of the material in the databse, e.g. "H2"

        Returns
        -------
        class
            PropertyData instance

        Raises
        ------
        KeyError
            If the material ``mat`` does not exist.
        TypeError
            If the material ``mat`` is not a string

        Examples
        --------
        >>> table = PropertyTable("ThermoPhysicalProperties.h5)
        >>> H2 = table.read("H2")
        >>> UO2 = table.read("UO2")

        """

        # Obtain the set for the specific material
        matset = self._getmataset(mat)
        return PropertyData(matset)

    def _getmataset(self, mat):
        """obtain the set for a specific material"""
        # check that mat is a string
        if not isinstance(mat, str):
            raise TypeError("mat must be a string and not {}".format(str(mat)))
        matset = self._h5.get(mat)
        if matset is None:
            raise KeyError("The material {} does not exist".format(mat))
        return matset

    def getref(self, mat):
        """Obtain the reference for a specific material

        Parameters
        ----------
        mat : string
            name of the material in the databse, e.g. "H2"

        Returns
        -------
        str
            Description of the reference

        Raises
        ------
        KeyError
            If the reference does not exist.
            If the material does not exist.
        TypeError
            If ``mat`` is not str type.

        Note
        ----
        * The reference is provided only for the material and not property.

        Examples
        --------
        >>> table = PropertyTable("ThermoPhysicalProperties.h5)
        >>> table.getref("H2")
        Hydrogen from 20 to 10000 K, NASA Lewis Research Center. NASA-TP-3378.

        """

        matset = self._getmataset(mat)
        reference = matset.attrs.get("reference")
        if reference is None:
            raise KeyError("No reference for material {}".format(mat))
        return reference.decode()


class PropertyData:
    """Interface to work with a specific material data set

    Methods
    -------
    evaluate : obtain a specific property value
    plot : plot the property value and the data points from the table
    whatis : description of the parameter and its units
    properties: obtain a list of all the available properties

    Parameters
    ----------
    matset : hdf5 data field

    Raises
    ------
    TypeError
        If ``matset`` is not a group in HDF5.
    KeyError
        If the members ``P`` and ``T`` do not exist in ``matset``.

    Note
    ----
    * At least one of the ``pressures`` or ``temperatures``
    dependencies must exist in the data sets.
    * Burnup dependence is not included in the current version.

    Examples
    --------
    >>> table = PropertyTable("ThermoPhysicalProperties.h5)
    >>> H2 = table.read("H2")

    """

    def __init__(self, matset):

        if not isinstance(matset, h5py.Group):
            raise TypeError("dataset must be a group in HDF5, not {}"
                            .format(type(matset)))
        self._matset = matset

        # Obtain dependencies
        self._P = self._getDependency("P")
        self._T = self._getDependency("T")
        if self._P is None and self._T is None:
            raise KeyError("No pressure and temperature dependencies for "
                           "material in the database")

    def evaluate(self, pty, pressure=None, temperature=None):
        """Evaluate a specific property for given temperature and/or pressure

        Pressure and/or temperatures can be provided as arguments,
        or by name. If just the temperature is used, either directly
        pass a ``None`` pressure, e.g.  ``evaluate("tc", None, 600)`` or
        use named arguments with ``evaluate("tc", temperature=600)``.
        Similarly for just pressure, but the option also exists to not pass
        anything as well, e.g. ``evalute("tc", 20)``

        Parameters
        ----------
        pty : string
            name of the property, e.g. "tc" thermal conductivity
        pressure : float, optional
            pressure in MPa
        temperature : float, optional
            temperature in Kelvin

        Returns
        -------
        float
            value of the property

        Raises
        ------
        KeyError
            Is the property ``pty`` does not exist in the ``matset``.
        TypeError
            If ``pty`` is not str type.
            If ``pressure`` and/or ``temperature`` are not properly defined.
        ValueError
            If ``pressure`` and/or ``temperature`` are not properly defined,
            e.g. values are out of bounds.

        Note
        ----
        * 2-D interpolation is allowed for temperature and pressure.
        * 1-D interpolation is allowed only for temperature.

        Examples
        --------
        >>> table = PropertyTable("ThermoPhysicalProperties.h5)
        >>> H2 = table.read("H2")
        >>> H2.evaluate("tc", pressure=8, temperature=600)
        0.3058
        >>> UO2 = table.read("UO2")
        >>> UO2.evaluate("tc", temperature=600)
        5.6532

        """

        if not isinstance(pty, str):
            raise ValueError("pty must be a string and not {}".format(pty))
        # Obtain the data for the specific property
        data = self._getdata(pty)

        if not isinstance(pressure, numbers.Real) and not isinstance(
                temperature, numbers.Real):
            raise ValueError("Need pressure and/or temperature")

        # 2-D interpolation on pressure and temperature
        if self._T is not None and self._P is not None:
            if pressure is None or temperature is None:
                raise ValueError("Both pressure and temperature are required")
            # 2D interpolation on pressure and temperature
            return self._interp2D(
                    pressure,
                    self._P,
                    "presssure",
                    temperature,
                    self._T,
                    "temperature",
                    data)

        # 1-D interpolation on temperature
        if self._T is not None and self._P is None:
            if temperature is None or pressure is not None:
                raise ValueError("Only temperature is required")
            return self._interp1D(
                    temperature, self._T, "temperature", data)

    def _interp2D(self, x, xvalues, xdesc, y, yvalues, ydesc, Z):

        if x < min(xvalues) or x > max(xvalues):
            raise ValueError(
                "{} must be between {} and {}, not {}".format(
                    xdesc, min(xvalues), max(xvalues), x
                )
            )
        if y < min(yvalues) or y > max(yvalues):
            raise ValueError(
                "{} must be between {} and {}, not {}".format(
                    ydesc, min(yvalues), max(yvalues), y
                )
            )

        # Find the extreme cases (P,T)min and (P,T)max
        idx00 = numpy.intersect1d(numpy.where(xvalues <= x), numpy.where(
                yvalues <= y), return_indices=False)[-1]
        idx11 = numpy.intersect1d(numpy.where(xvalues >= x), numpy.where(
                yvalues >= y), return_indices=False)[0]

        # (P,T) exist and there is no need to interpolate
        if idx00 == idx11:
            return Z[idx00]
        # same P[MPa], but different T[K]
        if xvalues[idx00] == xvalues[idx11]:
            ypts = yvalues[idx00], yvalues[idx11]
            zpts = Z[idx00], Z[idx11]
            return self._local1DInterp(y, ypts, zpts)
        # same T[K], but different P[MPa]
        elif yvalues[idx00] == yvalues[idx11]:
            xpts = xvalues[idx00], xvalues[idx11]
            zpts = Z[idx00], Z[idx11]
            return self._local1DInterp(x, xpts, zpts)

        zvalues = [
            [Z[idx00], Z[idx00+1]],
            [Z[idx11-1], Z[idx11]],
        ]

        xpts = xvalues[idx00], xvalues[idx11]
        ypts = yvalues[idx00], yvalues[idx11]

        return self._bilinear2D(
            x,
            y,
            xpts,
            ypts,
            zvalues,
        )

    @staticmethod
    def _bilinear2D(x, y, xv, yv, zm):
        denom = (xv[1] - xv[0]) * (yv[1] - yv[0])
        xlead = [xv[1] - x, x - xv[0]]
        ytail = [yv[1] - y, y - yv[0]]
        prod = numpy.matmul(zm, ytail)
        return numpy.matmul(xlead, prod) / denom

    def _interp1D(self, x, xvalues, xdesc, yvalues):
        if x < min(xvalues) or x > max(xvalues):
            raise ValueError(
                "{} must be between {} and {}, not {}".format(
                    xdesc, min(xvalues), max(xvalues), x
                )
            )
        # Find index that is closest to requested value
        index = bisect.bisect_left(xvalues, x)
        if xvalues[index] == x:
            return yvalues[index]
        return self._local1DInterp(x, xvalues[index:index+2],
                                   yvalues[index:index+2])

    @staticmethod
    def _local1DInterp(c, x, y):
        assert len(x) == len(y)
        slope = (y[1] - y[0]) / (x[1] - x[0])
        return y[0] + slope * (c - x[0])

    def whatis(self, pty):
        """Return information on a specific property

        Parameters
        ----------
        pty : string
            name of the thermal property, e.g. "tc" thermal conductivity

        Returns
        -------
        Units
            Object with ``units`` and ``name`` attributes

        Raises
        ------
        TypeError
            If the variable ``pty`` type is not string.
        KeyError
            If the ``pty`` does not exist.If the property lacks either
            the `units` or `description` attributes.

        Examples
        --------
        >>> UO2 = table.read("UO2")
        >>> UO2.whatis("tc")
        Units(name='Thermal Conductivity', units='W/m/K')
        >>> h = UO2.whatis("tc")
        >>> h.name
        'Thermal Conductivity'
        >>> h.units
        'W/m/K'

        """

        if not isinstance(pty, (str)):
            raise TypeError("pty must be a string and not {}".format(pty))
        # Obtain the attributes for the specific property
        dataset = self._matset.get(pty)
        if dataset is None:
            raise KeyError("No property {}".format(pty))
        units = dataset.attrs.get("units")
        descr = dataset.attrs.get("description")
        if units is None:
            raise KeyError("No units for property {}".format(pty))
        if descr is None:
            raise KeyError("No description for property {}".format(pty))

        return Units(descr.decode(), units.decode())

    def propertynames(self):
        """Print all the available properties for a specific material

        Returns
        -------
        list
            All the properties for the material in the database

        Examples
        --------
        >>> table = PropertyTable("ThermoPhysicalProperties.h5)
        >>> matsetH2 = table.read("H2")
        >>> H2 = PropertyData(matsetH2)
        >>> H2.propertynames()
        [cp', 'cv', 'g', 'h', 'my', 'pr', 'r', 's', 'tc', 'v']

        """

        propList = list(self._matset.keys())
        if "P" in propList:
            propList.remove("P")
        if "T" in propList:
            propList.remove("T")
        return propList

    def _getdata(self, pty):
        """Obtain the data for a certain property, e.g. density"""
        dataset = self._matset.get(pty)
        if dataset is None:
            raise KeyError("No property {}".format(pty))
        if dataset.shape[0] < 2 and dataset.shape[1] < 2:
            raise KeyError("Propery {} must have more data points than {}"
                           .format(pty, len(dataset)))
        data = dataset[:]
        return data.flatten()

    def _getDependency(self, dep):
        """obtain the dependency, e.g. P or T"""
        x = self._matset.get(dep)
        if x is not None:
            if len(x.shape) > 1 and x.shape[0] > 1:
                return x[:, 0]
            else:
                return x[0, :]
        else:
            return None

    def plot(self, pty, pressure=None, temperature=None,
             desc=None, units=None):
        """Comparative plot for the value of a specific property.

        The use of this method is similar to the ``evaluate`` method.

        Parameters
        ----------
        pty : string
            name of the thermal property, e.g. "tc" thermal conductivity
        pressure : float, optional
            pressure in MPa
        temperature : float, optional
            temperature in Kelvin
        desc : str, optional
            description of the property
            default value is read through the ``whatis`` method
        units : str, optional
            description of the property's units
            default value is read through the ``whatis`` method

        Raises
        ------
        TypeError
            If ``pty`` is not a string. If ``desc`` and/or ``units`` are given
            but these are not strings.
        KeyError
            If ``pressure`` and/or ``temperature`` are not properly defined.
        ValueError
             The value for ``pressure`` or ``temperature`` are out of bounds.

        Examples
        --------
        >>> table = PropertyTable("ThermoPhysicalProperties.h5")
        >>> H2 = table.read("H2")
        >>> H2.plot("tc", pressure=8, temperature=600)
        >>> H2.plot("tc", pressure=8, temperature=600, 'conductivity', 'W/m/K')

        """

        if not isinstance(pty, str):
            raise TypeError("pty must be a string and not {}".format(pty))
        # Obtain the data for the specific property
        data = self._getdata(pty)

        if pressure is None and temperature is None:
            raise KeyError("Need pressure and/or temperature")

        if desc is not None:
            if not isinstance(desc, str):
                raise TypeError("desc must be a string, not {}"
                                .format(type(desc)))
        else:
            h = self.whatis(pty)
            desc = h.name

        if units is not None:
            if not isinstance(units, str):
                raise TypeError("units must be a string, not {}"
                                .format(type(units)))
        else:
            h = self.whatis(pty)
            units = h.units

        plt.figure(figsize=(7, 4))
        if temperature is None:
            val = self.evaluate(pty, pressure=pressure)
            plt.plot(self._T, data, label='Database')
            plt.xlabel("Pressure, MPa")
            plt.ylabel('{} [{}]'
                       .format(desc, units))
            plt.plot(pressure, val, marker='o', fillstyle='none',
                     markeredgewidth=2.0, label='Interpolated')
            plt.axvline(pressure, c='grey', alpha=0.5, linestyle='--',
                        label='P={}MPa'.format(round(pressure, 2)))
            plt.axhline(val, c='grey', alpha=0.5, linestyle='--',
                        label='val={} [{}]'
                        .format(round(val, 3), units))
            plt.grid()
            plt.legend()

        elif pressure is None:
            val = self.evaluate(pty, temperature=temperature)
            plt.plot(self._T, data, label='Database')
            plt.xlabel("Temperature, K")
            plt.ylabel('{} [{}]'
                       .format(desc, units))
            plt.plot(temperature, val, marker='o', fillstyle='none',
                     markeredgewidth=2.0, label='Interpolated')
            plt.axvline(temperature, c='grey', alpha=0.5, linestyle='--',
                        label='T={}K'.format(round(temperature, 1)))
            plt.axhline(val, c='grey', alpha=0.5, linestyle='--',
                        label='val={} [{}]'
                        .format(round(val, 3), units))
            plt.grid()
            plt.legend()

        else:  # both pressure and temperature are given
            val = self.evaluate(pty, pressure=pressure,
                                temperature=temperature)

            # Find bounding pressures and temperature
            idx00 = numpy.intersect1d(numpy.where(self._P <= pressure),
                                      numpy.where(self._T <= temperature),
                                      return_indices=False)[-1]
            idx11 = numpy.intersect1d(numpy.where(self._P >= pressure),
                                      numpy.where(self._T >= temperature),
                                      return_indices=False)[0]

            xpts = numpy.array([self._T[idx00], self._T[idx00+1]])
            zpts0 = numpy.array([data[idx00], data[idx00+1]])
            zpts1 = numpy.array([data[idx11-1], data[idx11]])

            plt.plot(xpts, zpts0, label='P={} MPa'
                     .format(round(self._P[idx00], 3)))
            plt.plot(xpts, zpts1, label='P={} MPa'
                     .format(round(self._P[idx11], 3)))
            plt.plot(temperature, val, marker='o', fillstyle='none',
                     markeredgewidth=2.0, label='Interpolated')
            plt.axvline(temperature, c='grey', alpha=0.5, linestyle='--',
                        label='T={} K'.format(round(temperature, 1)))
            plt.axhline(val, c='grey', alpha=0.5, linestyle='--',
                        label='val={} [{}]'
                        .format(round(val, 3), units))
            plt.xlabel("Temperature, K")
            plt.ylabel('{} [{}]'
                       .format(desc, units))
            plt.grid()
            plt.legend()

            fontsize = 14
            plt.rc('font', size=fontsize)      # controls default text sizes
            plt.rc('axes', labelsize=fontsize)  # fontsize of the labels
            plt.rc('xtick', labelsize=fontsize)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=fontsize)  # fontsize of the tick labels
            plt.rc('legend', fontsize=fontsize)  # legend fontsize
