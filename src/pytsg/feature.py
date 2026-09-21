from typing import Callable, Union

import numpy as np
import numpy.polynomial.polynomial as poly
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull


def gaussian(x: np.ndarray, amplitude: float, mu: float, std: float) -> NDArray:
    """
    make a gaussian
    parameters: x (np.ndarray) the range on which the gaussian will be evaluated
                amplitude (float) intensity of gaussian
                mu (float) mean
                std (float) standard deviation
    returns: (NDArray) gaussian evaluated on x
    """
    g = amplitude * np.exp(-0.5 * ((x - mu) ** 2) / std**2)
    return g


def band_extractor(
    spectra: NDArray,
    start: int = 0,
    end: int = -1,
    statistic: list[Callable] = None,
) -> NDArray:
    """
    Function to extract band statistics from a spectra use of a list of callables.
    i.e numpy's min, max, argmin etc.

    The functions will calculate the statistics for each row of spectra as passing in a N x C array of spectra
    where N is the number of spectra and C is the wavelengths.

    A N x len(statistic) array is returned.

    Args:
        spectra (NDArray): 2d array of spectra rows x wavelength
        start (int, optional): starting channel of the band. Defaults to 0.
        end (int, optional): last channel of the band. Defaults to -1.
        statistic (list[Callable], optional): list of callables that will be called on the spectra. Defaults
            to [np.argmin, np.min, np.sum].

    Raises:
        ValueError: if spectra is not a 2d array

    Returns:
        NDArray: 2d array of band statistics rows represent the spectra, columns represent the parameters,
            results are returned in channel space.

    """
    if spectra.ndim != 2:
        raise ValueError("spectra must be a 2d array")
    if end == -1:
        # handle the -1 case for the end of the spectra
        end = spectra.shape[1]
    if statistic is None:
        statistic = [np.argmin, np.min, np.sum]

    nout: int = len(statistic)
    r: int = spectra.shape[0]
    output: NDArray = np.zeros((r, nout))
    tmp_band: NDArray
    for i, fun in enumerate(statistic):
        tmp_band = np.apply_along_axis(fun, 1, spectra[:, start:end])
        output[:, i] = tmp_band

    return output


def sqm(
    wavelength: NDArray,
    spectra: NDArray,
    start_wavelength: Union[None, float] = None,
    end_wavelength: Union[None, float] = None,
) -> tuple[NDArray, NDArray]:
    """
    Implementation of the simple quadratic method for extracting the feature depth and position.
    https://doi.org/10.1016/j.rse.2011.11.025

    Args:
        wavelength (NDArray): wavelength of the spectra
        spectra (NDArray): 2d array of spectra rows x wavelength
        start_wavelength (Union[None, float], optional): starting wavelength of the band if None than the
            first channel is used. Defaults to None.
        end_wavelength (Union[None, float], optional): ending wavelength of the band if the end wavelength is
            None then the last channel is used. Defaults to None.

    Raises:
        ValueError: if spectra is not a 2d array

    Returns:
        tuple[NDArray, NDArray]: a 2d array of the paramters representing  amplitude, centre, and width at the
            zero crossing i.e. the polynomial roots each row is a represent a spectrum. A 2d NDArray
            representing the coefficients of the 2nd degree polynomial in descending order
    """
    # handle the None case for the start and end of the wavelength selection
    start: int
    end: int
    if spectra.ndim != 2:
        raise ValueError("spectra must be a 2d array")
    if start_wavelength is None:
        start = 0
    else:
        start = np.min(np.where(wavelength >= start_wavelength)[0])

    if end_wavelength is None:
        end = spectra.shape[1]
    else:
        end = np.max(np.where(wavelength <= end_wavelength)[0])

    rows = spectra.shape[0]
    # we need to transpose the spectra so that we can run polyfit on the array

    coefs = poly.polyfit(wavelength[start:end], spectra[:, start:end].T, 2)
    axis_of_symmetry: NDArray = -(coefs[1, :] / (2 * coefs[2, :]))
    # calculate amplitude, width and centre
    output: NDArray = np.zeros((rows, 3))
    vertex: NDArray = poly.polyval(output[:, 0], coefs)

    output[:, 0] = axis_of_symmetry.ravel()
    output[:, 1] = vertex[:, 0]
    # calculate the width of the polynomial at 0
    for i in range(rows):
        output[i, 2] = np.diff(poly.polyroots(coefs[:, i].ravel()))[0]
    # concatenate the parameters into output array
    output_coefs = coefs.T
    return output, output_coefs


def fit_gaussian(
    wavelength: NDArray,
    spectra: NDArray,
    x0: NDArray = None,
) -> NDArray[np.float64]:
    """Function to optimise a gaussian fit to the spectra using least squares

    Args:
        wavelength (NDArray): wavelength of the spectra
        spectra (NDArray): 2d array of spectra rows x wavelength
        x0 (NDArray, optional): array representing the initial amplitude, centre and width of the gaussian.
            Defaults to array([1, 10, 1]).

    Raises:
        ValueError: if spectra is not a 2d array

    Returns:
        NDArray[np.float64]: a 2d array of the parameters of the gaussian fit amplitude, centre, width. Rows
            represent the spectra. Results are returned in wavelength units.
    """

    if spectra.ndim != 2:
        raise ValueError("spectra must be a 2d array")
    nspectra: int = spectra.shape[0]

    parameters: NDArray = np.zeros((nspectra, 3))

    if x0 is None:
        x0 = np.asarray([1, 10, 1])

    for i in range(nspectra):
        lsf = least_squares(lambda x, i=i: spectra[i] - gaussian(wavelength, x[0], x[1], x[2]), x0=x0)
        parameters[i, :] = lsf.x

    return parameters


def chull(xy: NDArray) -> NDArray:
    """
    calculates the convex hull of a spectrum
    parameters:
        xy (np.ndarray): wavelength and reflectance of the spectra
    returns: (NDArray) the convex hull of the spectra
    """

    vertices: NDArray = ConvexHull(xy).vertices
    # cross product to check if the points are above or below the line if they are below the line
    # the we will remove them
    good_points: NDArray = np.cross(xy[vertices] - xy[0], xy[vertices] - xy[-1]) >= 0
    good_verts: NDArray = np.sort(vertices[good_points])
    hull: NDArray = np.interp(xy[:, 0], xy[good_verts, 0], xy[good_verts, 1])
    return hull
