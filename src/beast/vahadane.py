import spams
import logging
import numpy as np
from typing import Callable
from .log_utils import get_logger
from .base_classes.deconvolution_normaliser import BaseDeconWSINormaliser, BaseDeconTileNormaliser, BaseDeconStainAugmenter
from .file_readers import read_wsi, read_tile
from .utils import _RGB_to_OD, _normalize_matrix_rows, IntensitySamplingMethod

def _get_stain_components(image_arr:np.ndarray, regularizer=0.1) -> np.ndarray:
	""" Extracts the stain component matrix from an image.

		This represents the contribution of each stain to the red green and blue channels in optical density space.

		Args:
			image_arr (np.ndarray): The image array.
			regularizer (float, optional): The regularisation parameter. Defaults to 0.1.

		Returns:
			np.ndarray: The stain matrix.
	"""
	OD = _RGB_to_OD(image_arr).reshape((-1, 3))
	stain_matrix = spams.trainDL(X=OD.T, K=2, lambda1=regularizer, mode=2, modeD=0, posAlpha=True, posD=True, verbose=False).T

	if stain_matrix[0, 0] < stain_matrix[1, 0]:
		stain_matrix = stain_matrix[[1, 0], :]

	stain_matrix = _normalize_matrix_rows(stain_matrix)

	return stain_matrix

class VahadaneWSINormaliser(BaseDeconWSINormaliser):
	"""	Class to transform the stain intensity in a whole slide image to match a target using the Vahadane NMF stain deconvolution algorithm (https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7460968). Warning: This class considers the entire slide to have uniform staining, this is useful for visualisation but may not be appropriate for analysis when stain intensity varies across the slide. For analysis, use the VahadaneTileNormaliser or VahadaneWSITesselatedNormaliser class.

		Args:
			num_threads (int, optional): The number of threads to use. Defaults to 16.
			intensity_sampling_method (IntensitySamplingMethod, optional): The method to use to sample the intensity. Defaults to IntensitySamplingMethod.PERCENTILE.
			stain_percentile (float, optional): The maximum percentile of the stain intensity to use. Defaults to 90.0.
			no_pixel_samples (int, optional): The number of pixels to sample. Defaults to 2000000.
			file_loader (Callable, optional): The function to use to load the image. Defaults to read_wsi. A function that accepts a path and returns a numpy array can be supplied to allow for custom loading.
			logfile (Path, optional): The path to save the log file to. Defaults to None.
			terminal_log_level (int, optional): The log level to print to the terminal. Defaults to logging.DEBUG.
			file_log_level (int, optional): The log level to print to the log file. Defaults to logging.DEBUG.
	"""
	def __init__(self, num_threads:int=16, intensity_sampling_method=IntensitySamplingMethod.PERCENTILE, intensity_percentile:float=95.0, no_pixel_samples:int=2000000, file_loader:Callable=read_wsi, logfile=None, terminal_log_level=logging.DEBUG, file_log_level=logging.DEBUG) -> None:
		logger = get_logger("Vahadane WSI Normaliser", logfile, terminal_log_level, file_log_level)
		super().__init__(file_loader, logger, _get_stain_components, num_threads, intensity_sampling_method, intensity_percentile, no_pixel_samples)

class VahadaneTileNormaliser(BaseDeconTileNormaliser):
	""" Class to transform the stain colour profile and intensity in a tile to match a target using the Vahadane NMF stain deconvolution algorithm (https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7460968).

		Args:
			intensity_sampling_method (IntensitySamplingMethod, optional): The method to use to sample the intensity. Defaults to IntensitySamplingMethod.PERCENTILE.
			stain_percentile (float, optional): The maximum percentile of the stain intensity to use. Defaults to 90.0.
			file_loader (Callable, optional): The function to use to load the image. Defaults to read_tile. A function that accepts a path and returns a numpy array can be supplied to allow for custom loading.
			logfile (Path, optional): The path to save the log file to. Defaults to None.
			terminal_log_level (int, optional): The log level to print to the terminal. Defaults to logging.DEBUG.
			file_log_level (int, optional): The log level to print to the log file. Defaults to logging.DEBUG.
	"""
	def __init__(self, intensity_sampling_method=IntensitySamplingMethod.PERCENTILE, intensity_percentile:float=95.0, file_loader:Callable=read_tile, logfile=None, terminal_log_level=logging.DEBUG, file_log_level=logging.DEBUG) -> None:
		logger = get_logger("Vahadane Tile Normaliser", logfile, terminal_log_level, file_log_level)
		super().__init__(file_loader, logger, _get_stain_components, intensity_sampling_method, intensity_percentile)

class VahadaneStainAugmenter(BaseDeconStainAugmenter):
	""" Class to augment the tissue colour profile of a tile through the Vahadane NMF stain deconvolution algorithm (https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7460968) and subsequent intensity or compoment matrix perturbation.
	
		Args:
			sigma1 (float): The standard deviation of the Gaussian blur applied to the background.
			sigma2 (float): The standard deviation of the Gaussian blur applied to the foreground.
			file_loader (Callable): The function to use to load the tile.
			logger (logging.Logger, optional): The logger to use. Defaults to None.
	"""
	def __init__(self, sigma1=0.4, sigma2=0.05, file_loader:Callable=read_tile, logger:logging.Logger=None):
		super().__init__(file_loader, logger, _get_stain_components, sigma1, sigma2)