import logging
import numpy as np
from typing import Callable
from .log_utils import get_logger
from .base_classes.deconvolution_normaliser import BaseDeconWSINormaliser, BaseDeconTileNormaliser, BaseDeconStainAugmenter
from .file_readers import read_wsi, read_tile
from .utils import _RGB_to_OD, _normalize_matrix_rows, IntensitySamplingMethod

import numpy as np

def _get_n_stain_components(image_arr:np.ndarray, num_stains:int=2, angular_percentile:int=99) -> np.ndarray:
    """ Extracts the stain component matrix from an image.

    This represents the contribution of each stain to the red green and blue channels in optical density space.

    Args:
        image_arr (np.ndarray): The image array.
        num_stains (int, optional): The number of stains to extract. Defaults to 2.
        angular_percentile (int, optional): The percentile used to compute the min and max angles.
            Defaults to 99.

    Returns:
        np.ndarray: The stain matrix.
    """

    OD = _RGB_to_OD(image_arr).reshape((-1, 3))

    # Eigenvectors of cov in OD space (orthogonal as cov symmetric)
    _, SV = np.linalg.eigh(np.cov(OD, rowvar=False))

    # The two principle eigenvectors
    SV = SV[:, -num_stains:]

    # Make sure vectors are pointing the right way
    SV[:, SV[0, :] < 0] *= -1

    # Project on this basis.
    That = np.dot(OD, SV)

    # Angular coordinates with repect to the principle, orthogonal eigenvectors
    phi = np.arctan2(That[:, -1::-1], That[:, :])

    # Min and max angles for each stain
    minPhi = np.percentile(phi, 100 - angular_percentile, axis=0)
    maxPhi = np.percentile(phi, angular_percentile, axis=0)

    # The principle colors for each stain
    stain_matrix = np.zeros((num_stains, 3))
    for i in range(num_stains):
        v1 = np.dot(SV, np.array([np.cos(minPhi[i]), np.sin(minPhi[i])]))
        v2 = np.dot(SV, np.array([np.cos(maxPhi[i]), np.sin(maxPhi[i])]))

        if v1[0] > v2[0]:
            stain_matrix[i, :] = v1
        else:
            stain_matrix[i, :] = v2

    return _normalize_matrix_rows(stain_matrix)

def _get_stain_components(image_arr:np.ndarray, angular_percentile:int=99) -> np.ndarray:
	""" Extracts the stain component matrix from an image.

		This represents the contribution of each stain to the red green and blue channels in optical density space.

		Args:
			image_arr (np.ndarray): The image array.
			regularizer (float, optional): The regularisation parameter. Defaults to 0.1.

		Returns:
			np.ndarray: The stain matrix.
	"""

	OD = _RGB_to_OD(image_arr).reshape((-1, 3))

	# Eigenvectors of cov in OD space (orthogonal as cov symmetric)
	_, SV = np.linalg.eigh(np.cov(OD, rowvar=False))

	# The two principle eigenvectors
	SV = SV[:, [2, 1]]

	# Make sure vectors are pointing the right way
	if SV[0, 0] < 0: SV[:, 0] *= -1
	if SV[0, 1] < 0: SV[:, 1] *= -1

	# Project on this basis.
	That = np.dot(OD, SV)

	# Angular coordinates with repect to the prinicple, orthogonal eigenvectors
	phi = np.arctan2(That[:, 1], That[:, 0])

	# Min and max angles
	minPhi = np.percentile(phi, 100 - angular_percentile)
	maxPhi = np.percentile(phi, angular_percentile)

	# the two principle colors
	v1 = np.dot(SV, np.array([np.cos(minPhi), np.sin(minPhi)]))
	v2 = np.dot(SV, np.array([np.cos(maxPhi), np.sin(maxPhi)]))

	# Order of H and E.
	# H first row.
	if v1[0] > v2[0]:
		stain_matrix = np.array([v1, v2])
	else:
		stain_matrix = np.array([v2, v1])

	return _normalize_matrix_rows(stain_matrix)

class MacenkoWSINormaliser(BaseDeconWSINormaliser):
	""" Class to transform the stain profile and intensity in a whole slide image to match a target using the Macenko stain vector estimation and normalisation algorithm (https://www.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf). Warning: This class considers the entire slide to have uniform staining, this is useful for visualisation but may not be appropriate for analysis when stain intensity varies across the slide. For analysis, use the MacenkoTileNormaliser or MacenkoWSITesselatedNormaliser class.

		Args:
			num_threads (int, optional): The number of threads to use. Defaults to 16.
			file_loader (Callable, optional): The function to use to load the image. Defaults to read_wsi. A function that accepts a path and returns a numpy array can be supplied to allow for custom loading.
			logfile (Path, optional): The path to save the log file to. Defaults to None.
			terminal_log_level (int, optional): The log level to print to the terminal. Defaults to logging.DEBUG.
			file_log_level (int, optional): The log level to print to the log file. Defaults to logging.DEBUG.
	"""
	def __init__(self, num_threads:int=16, intensity_sampling_method=IntensitySamplingMethod.PERCENTILE, intensity_percentile:float=95.0, no_pixel_samples:int=2000000, file_loader:Callable=read_wsi, logfile=None, terminal_log_level=logging.DEBUG, file_log_level=logging.DEBUG) -> None:
		logger = get_logger("Macenko WSI Normaliser", logfile, terminal_log_level, file_log_level)
		super().__init__(file_loader, logger, _get_stain_components, num_threads, intensity_sampling_method, intensity_percentile, no_pixel_samples)

class MacenkoTileNormaliser(BaseDeconTileNormaliser):
	""" Class to transform the stain profle and intensity in a tile to match a target using the Macenko stain vector estimation and normalisation algorithm (https://www.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf).

		Args:
			intensity_sampling_method (IntensitySamplingMethod, optional): The method to use to sample the intensity of the image. Defaults to IntensitySamplingMethod.PERCENTILE.
			stain_percentile (float, optional): The percentile to use to sample the intensity of the image. Defaults to 95.0.
			file_loader (Callable, optional): The function to use to load the image. Defaults to read_tile. A function that accepts a path and returns a numpy array can be supplied to allow for custom loading.
			logfile (Path, optional): The path to save the log file to. Defaults to None.
			terminal_log_level (int, optional): The log level to print to the terminal. Defaults to logging.DEBUG.
			file_log_level (int, optional): The log level to print to the log file. Defaults to logging.DEBUG.
	"""
	def __init__(self, intensity_sampling_method=IntensitySamplingMethod.PERCENTILE, intensity_percentile:float=95.0, file_loader:Callable=read_tile, logfile=None, terminal_log_level=logging.DEBUG, file_log_level=logging.DEBUG) -> None:
		logger = get_logger("Macenko Tile Normaliser", logfile, terminal_log_level, file_log_level)
		super().__init__(file_loader, logger, _get_stain_components, intensity_sampling_method, intensity_percentile)

class MacenkoStainAugmenter(BaseDeconStainAugmenter):
	"""	Class to augment the stain profile and intensity of a tile through Macenko stain vector estimation (https://www.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf) and subsequent intensity or compoment matrix perturbation.
	
		Args:
			sigma1 (float): The standard deviation of the Gaussian blur applied to the background.
			sigma2 (float): The standard deviation of the Gaussian blur applied to the foreground.
			file_loader (Callable, optional): The function to use to load the tile. Defaults to read_tile. A function that accepts a path and returns a numpy array can be supplied to allow for custom loading.
			logfile (Path, optional): The path to save the log file to. Defaults to None.
			terminal_log_level (int, optional): The log level to print to the terminal. Defaults to logging.DEBUG.
			file_log_level (int, optional): The log level to print to the log file. Defaults to logging.DEBUG.
	"""
	def __init__(self, sigma1=0.5, sigma2=0.05, file_loader:Callable=read_tile, logfile=None, terminal_log_level=logging.DEBUG, file_log_level=logging.DEBUG):
		logger = get_logger("Macenko Tile Normaliser", logfile, terminal_log_level, file_log_level)
		super().__init__(file_loader, logger, _get_stain_components, sigma1, sigma2)