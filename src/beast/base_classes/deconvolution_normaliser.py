import json
import spams
import logging
import numpy as np
from pathlib import Path
from typing import Callable, Tuple
from multiprocessing.pool import ThreadPool
from functools import partial
from .normaliser import BaseNormaliser
from ..utils import rgb_luminosity_standardiser, _RGB_to_OD, _OD_to_RGB, generate_tile_coordinates, get_otsu_tissue_mask, get_tissue_mask, TileCoordinate, IntensitySamplingMethod

SLIDE_ARR = None
OUTPUT_ARR = None
USE_SOURCE = False

def _calculate_scaling_coefficients(source_dict:dict, target_dict:dict, logger:logging.Logger) -> np.ndarray:
	scaling_coefficients = target_dict["intensity_samples"] / source_dict["intensity_samples"]

	if np.nan in scaling_coefficients:
		logger.warning("Scaling coefficients contain NaNs, replacing with 1.")
		scaling_coefficients[np.isnan(scaling_coefficients)] = 1
	
	if np.inf in scaling_coefficients:
		logger.warning("Scaling coefficients contain Inf, replacing with 1.")
		scaling_coefficients[np.isinf(scaling_coefficients)] = 1

	if np.min(scaling_coefficients) <= 0:
		logger.warning("Scaling coefficients contain negative or zero values, replacing with 1.")
		scaling_coefficients[scaling_coefficients <= 0] = 1
	return scaling_coefficients

def _get_stain_intensities(image_arr:np.ndarray, source_matrix:np.ndarray, regularizer=0.01) -> np.ndarray:
	""" Extracts the stain intensities from an image.

		Args:
			image_arr (np.ndarray): The image array.
			source_matrix (np.ndarray): The source stain matrix.
			regularizer (float, optional): The regularisation parameter. Defaults to 0.01.

		Returns:
			np.ndarray: The stain intensities.
	"""
	new_shape = list(image_arr.shape)
	new_shape[-1] = 2

	OD = _RGB_to_OD(image_arr).reshape((-1, 3))
	intensities = spams.lasso(X=OD.T, D=source_matrix.T, mode=2, lambda1=regularizer, pos=True).toarray().T
	return intensities.reshape(tuple(new_shape))

def _normalise_stain_intensities(intensity_matrix:np.ndarray, scaling_coefficients:np.ndarray) -> np.ndarray:
	"""	Normalises the stain intensities using the scaling coefficients.

		Args:
			intensity_matrix (np.ndarray): The stain intensities.
			scaling_coefficients (np.ndarray): The scaling coefficients.

		Returns:
			np.ndarray: The normalised stain intensities.
	"""
	intensity_matrix *= scaling_coefficients
	return intensity_matrix

def _transform_stain(intensity_arr:np.ndarray, source_matrix:np.ndarray, target_matrix:np.ndarray, scaling_coefficients:np.ndarray) -> np.ndarray:
	""" Transforms the stain of an image using the source and target stain matrices.

		Args:
			intensity_arr (np.ndarray): The image array.
			source_matrix (np.ndarray): The source stain matrix.
			target_matrix (np.ndarray): The target stain matrix.
			scaling_coefficients (np.ndarray): The scaling coefficients.
		
		Returns:
			np.ndarray: The transformed image array.
	"""
	source_intensities = _normalise_stain_intensities(intensity_arr, scaling_coefficients)
	combined_stains = source_intensities @ target_matrix
	return _OD_to_RGB(combined_stains)

def _translate_to_stain1_rgb(intensity_arr:np.ndarray, source_matrix:np.ndarray, target_matrix:np.ndarray, scaling_coefficients:np.ndarray) -> np.ndarray:
	""" Filters an image to display only the first stain of the target stain matrix.

		Args:
			intensity_arr (np.ndarray): The image array.
			source_matrix (np.ndarray): The source stain matrix.
			target_matrix (np.ndarray): The target stain matrix.
			scaling_coefficients (np.ndarray): The scaling coefficients.

		Returns:
			np.ndarray: The filtered image array.
	"""
	if USE_SOURCE:
		target_matrix =  source_matrix[0,:]
	else:
		source_intensities = _normalise_stain_intensities(intensity_arr, scaling_coefficients)
		target_matrix =  target_matrix[0,:]

	S1 = source_intensities[:,:, 0]
	S1_EX = np.expand_dims(S1, axis=-1)

	target_matrix = np.expand_dims(target_matrix, axis=0)

	S1_DOT =  S1_EX @ target_matrix
	S1_RGB = _OD_to_RGB(S1_DOT)
	return S1_RGB

def _translate_to_stain2_rgb(intensity_arr:np.ndarray, source_matrix:np.ndarray, target_matrix:np.ndarray, scaling_coefficients:np.ndarray) -> np.ndarray:
	""" Filters an imaee to display only the second stain of the target stain matrix.

		Args:
			intensity_arr (np.ndarray): The image array.
			source_matrix (np.ndarray): The source stain matrix.
			target_matrix (np.ndarray): The target stain matrix.
			scaling_coefficients (np.ndarray): The scaling coefficients.

		Returns:
			np.ndarray: The filtered image array.
	"""
	if USE_SOURCE:
		target_matrix =  source_matrix[1,:]
	else:
		source_intensities = _normalise_stain_intensities(intensity_arr, scaling_coefficients)
		target_matrix =  target_matrix[1,:]
	
	S2 = source_intensities[:,:, 1]
	S2_EX = np.expand_dims(S2, axis=-1)

	target_matrix = np.expand_dims(target_matrix, axis=0)

	S2_DOT =  S2_EX @ target_matrix
	S2_RGB = _OD_to_RGB(S2_DOT)
	return S2_RGB

def _translate_to_stain_concentrations(intensity_arr:np.ndarray, source_matrix:np.ndarray, target_matrix:np.ndarray, scaling_coefficients:np.ndarray) -> np.ndarray:
	"""	Translates the image to represent the stain concentrations using the source stain matrix.

		Args:
			intensity_arr (np.ndarray): The image array.
			source_matrix (np.ndarray): The source stain matrix.
			target_matrix (np.ndarray): The target stain matrix.
			scaling_coefficients (np.ndarray): The scaling coefficients.
		
		Returns:
			np.ndarray: The translated image array.
	"""
	if USE_SOURCE:
		source_intensities = intensity_arr
	else:
		source_intensities = _normalise_stain_intensities(intensity_arr, scaling_coefficients)
	HE_RS = _OD_to_RGB(source_intensities)
	return HE_RS

def _multithread_transform(tile_coords:TileCoordinate, transformation_function:Callable, source_matrix:np.ndarray, target_matrix:np.ndarray, coefficient_matrix:np.ndarray) -> None:
	""" Applies a transformation function to a tile of the slide.

		Args:
			tile_coords (TileCoordinate): The tile coordinates.
			transformation_function (Callable): The transformation function.
			source_matrix (np.ndarray): The source stain matrix.
			target_matrix (np.ndarray): The target stain matrix.
			coefficient_matrix (np.ndarray): The scaling coefficients.

		Returns:
			None
	"""
	global OUTPUT_ARR
	source_patch = SLIDE_ARR[tile_coords.ystart:tile_coords.yend,tile_coords.xstart:tile_coords.xend,:]
	intensity_arr = _get_stain_intensities(source_patch, source_matrix)
	translated_patch = transformation_function(intensity_arr, source_matrix, target_matrix, coefficient_matrix)
	OUTPUT_ARR[tile_coords.ystart:tile_coords.yend,tile_coords.xstart:tile_coords.xend,:] = translated_patch

class BaseDeconWSINormaliser(BaseNormaliser):
	"""
		Class to transform the stain intensity in a whole slide image to match a target using the stain deconvolution algorithm. Warning: This class considers the entire slide to have uniform staining, this is useful for visualisation but may not be appropriate for analysis when stain intensity varies across the slide. For analysis, use a TileNormaliser or WSITesselatedNormaliser class.

		Args:
			file_loader (Callable, optional): The function to use to load the image. Defaults to read_wsi. A function that accepts a path and returns a numpy array can be supplied to allow for custom loading.
			logger (logging.Logger, optional): The logger to use. Defaults to None.
			_stain_component_fn (Callable, optional): The function to use to calculate the stain components.
			num_threads (int, optional): The number of threads to use. Defaults to 16.
			stain_percentile (float, optional): The percentile to use to calculate the maximum stain intensity.
			no_pixel_samples (int, optional): The number of pixels to sample to calculate the maximum stain intensity.
	"""
	def __init__(self, file_loader:Callable, logger:logging.Logger, _stain_component_fn:Callable, num_threads:int, intensity_sampling_method:IntensitySamplingMethod, intensity_percentile:float, no_pixel_samples:int) -> None:
		super().__init__(file_loader, logger)
		self.tile_size = (5000,5000)
		self.num_threads = num_threads
		self.stain_component_fn = _stain_component_fn
		self.intensity_sampling_method = intensity_sampling_method
		self.intensitiy_percentile = intensity_percentile
		self.no_pixel_samples = no_pixel_samples

	def save_fit(self, json_pth: Path) -> None:
		"""	Saves the target stain metadata to a json file.

			Args:
				json_pth (Path): The path to save the json file to.

			Returns:
				None
		"""
		output_json = {}

		for key in self.target_dict:
			output_json[key] = self.target_dict[key].tolist()

		with open(json_pth, "w") as json_file:
			json.dump(output_json, json_file)
		self.logger.debug("Saved target stain metadata to: {}".format(json_pth))

	def load_fit(self, json_pth: Path) -> None:
		""" Loads the target stain metadata from a json file.

			Args:
				json_pth (Path): The path to load the json file from.

			Returns:
				dict: The target stain metadata.
		"""
		with open(json_pth, "r") as json_file:
			input_json = json.load(json_file)

		self.target_dict = {}
		for key in input_json:
			self.target_dict[key] = np.array(input_json[key])
		self.logger.debug("Loaded target stain metadata from: {}".format(json_pth))
		self.logger.debug("Target stain metadata: {}".format(self.target_dict))
		return self.target_dict

	def _mt_transform(self, slide_arr:np.ndarray, translation_function:Callable, source_dict:dict, scaling_coefficients:np.ndarray) -> np.ndarray:
		""" Applies a transformation function to a slide using multiple threads.

			Args:
				slide_arr (np.ndarray): The slide array.
				translation_function (Callable): The transformation function.
				source_dict (dict): The source stain metadata.
				scaling_coefficients (np.ndarray): The scaling coefficients.
			
			Returns:
				np.ndarray: The transformed slide array.
		"""
		global SLIDE_ARR
		global OUTPUT_ARR

		SLIDE_ARR = None
		OUTPUT_ARR = None

		SLIDE_ARR = slide_arr
		OUTPUT_ARR = slide_arr

		dimensions = slide_arr.shape

		tile_coordinates = generate_tile_coordinates((dimensions[1], dimensions[0]), self.tile_size)
	
		if translation_function == _translate_to_stain_concentrations:
			dimensions = (dimensions[0], dimensions[1], 2)
			OUTPUT_ARR = np.zeros(dimensions, dtype=np.uint8)

		args = [(tile_coord, translation_function, source_dict.get("stain_matrix"), self.target_dict.get("stain_matrix"), scaling_coefficients) for tile_coord in tile_coordinates]

		self.logger.debug("Starting multithreaded transform...")

		with ThreadPool(self.num_threads) as pool:
			pool.starmap(_multithread_transform, args, chunksize=1)

		self.logger.debug("Multithreaded transform complete.")

		return OUTPUT_ARR

	def _fit(self, slide_arr: np.ndarray) -> Tuple[dict, np.ndarray]:
		""" Fits the stain matrix to the target stain matrix.

			Args:
				slide_arr (np.ndarray): The slide array.

			Returns:
				dict: The source stain metadata.
		"""
		lums_arr = rgb_luminosity_standardiser(slide_arr)

		sample_arr = self._sample_wsi(lums_arr, self.no_pixel_samples)
		tissue_mask = get_otsu_tissue_mask(sample_arr)
		tissue_samples = sample_arr[tissue_mask==0]
		
		stain_matrix = self.stain_component_fn(tissue_samples)
		stain_intensities = _get_stain_intensities(tissue_samples, stain_matrix)

		if self.intensity_sampling_method == IntensitySamplingMethod.MEAN:
			statistic = partial(np.mean, axis=0)
		elif self.intensity_sampling_method == IntensitySamplingMethod.MEDIAN:
			statistic = partial(np.median, axis=0)
		elif self.intensity_sampling_method == IntensitySamplingMethod.MIN:
			statistic = partial(np.min, axis=0)
		elif self.intensity_sampling_method == IntensitySamplingMethod.MAX:
			statistic = partial(np.max, axis=0)
		elif self.intensity_sampling_method == IntensitySamplingMethod.PERCENTILE:
			statistic = partial(np.percentile, q=self.intensitiy_percentile, axis=0)

		sampled_intensities = statistic(stain_intensities).reshape((1, 2))
		fit_dictionary = {"stain_matrix": stain_matrix, "intensity_samples": sampled_intensities}

		return fit_dictionary, lums_arr

	def _normalise(self, slide_arr: np.ndarray) -> np.ndarray:
		""" Normalises the slide to the target stain matrix.

			Args:
				slide_arr (np.ndarray): The slide array.

			Returns:
				np.ndarray: The normalised slide array.
		"""
		source_dict, lums_arr = self._fit(slide_arr)
		self.logger.debug("Source metadata: {}".format(source_dict))

		scaling_coefficients = _calculate_scaling_coefficients(source_dict, self.target_dict, self.logger)
		
		normalised_arr = self._mt_transform(lums_arr, _transform_stain, source_dict, scaling_coefficients)

		return normalised_arr

	def convert_to_stain1(self, slide_arr:np.ndarray, use_source:bool=False) -> np.ndarray:
		""" Converts the slide to only display stain 1 in RGB.

			Args:
				slide_arr (np.ndarray): The slide array.
				use_source (bool): Whether to use the source stain matrix or the target stain matrix to render stain 1.


			Returns:
				np.ndarray: The converted slide array.
		"""
		global USE_SOURCE
		USE_SOURCE = use_source

		slide_arr = self._load_path_or_array(slide_arr)

		source_dict, lums_arr = self._fit(slide_arr)
		self.logger.debug("Source metadata: {}".format(source_dict))

		scaling_coefficients = _calculate_scaling_coefficients(source_dict, self.target_dict, self.logger)
		
		converted_arr = self._mt_transform(lums_arr, _translate_to_stain1_rgb, source_dict, scaling_coefficients)

		return converted_arr

	def convert_to_stain2(self, slide_arr:np.ndarray, use_source:bool=False) -> np.ndarray:
		""" Converts the slide to only display stain two in RGB.

			Args:
				slide_arr (np.ndarray): The slide array.
				use_source (bool): Whether to use the source stain matrix or the target stain matrix to render stain 2.

			Returns:
				np.ndarray: The converted slide array.
		"""
		global USE_SOURCE
		USE_SOURCE = use_source

		slide_arr = self._load_path_or_array(slide_arr)

		source_dict, lums_arr = self._fit(slide_arr)
		self.logger.debug("Source metadata: {}".format(source_dict))

		scaling_coefficients = _calculate_scaling_coefficients(source_dict, self.target_dict, self.logger)
		
		converted_arr = self._mt_transform(lums_arr, _translate_to_stain2_rgb, source_dict, scaling_coefficients)

		return converted_arr

	def convert_to_stain_concentrations(self, slide_arr:np.ndarray) -> np.ndarray:
		""" Converts the slide to a two channel array of stain concentrations.

			Args:
				slide_arr (np.ndarray): The slide array.

			Returns:
				np.ndarray: The converted slide array.
		"""

		slide_arr = self._load_path_or_array(slide_arr)

		source_dict, lums_arr = self._fit(slide_arr)
		self.logger.debug("Source metadata: {}".format(source_dict))

		scaling_coefficients = _calculate_scaling_coefficients(source_dict, self.target_dict, self.logger)
		
		converted_arr = self._mt_transform(lums_arr, _translate_to_stain_concentrations, source_dict, scaling_coefficients)

		return converted_arr

class BaseDeconTileNormaliser(BaseNormaliser):
	""" Class to normalise a single tile using stain deconvolution and normalisation.

		Args:
			file_loader (Callable): The function to load the slide array.
			logger (logging.Logger): The logger to use.
			_stain_component_fn (Callable): The function to use to calculate the stain components.
			max_stain_percentile (float): The percentile to use to calculate the maximum stain intensity.
	"""
	def __init__(self, file_loader:Callable, logger:logging.Logger, _stain_component_fn:Callable, intensity_sampling_method:IntensitySamplingMethod, intensity_percentile:float):
		super().__init__(file_loader, logger)
		self.stain_component_fn = _stain_component_fn
		self.intensity_sampling_method = intensity_sampling_method
		self.intensity_percentile = intensity_percentile

	def save_fit(self, json_pth: Path) -> None:
		"""	Saves the target stain metadata to a json file.

			Args:
				json_pth (Path): The path to save the json file to.

			Returns:
				None
		"""
		output_json = {}

		for key in self.target_dict:
			output_json[key] = self.target_dict[key].tolist()

		with open(json_pth, "w") as json_file:
			json.dump(output_json, json_file)
		self.logger.debug("Saved target stain metadata to: {}".format(json_pth))

	def load_fit(self, json_pth: Path) -> None:
		"""	Loads the target stain metadata from a json file.

			Args:
				json_pth (Path): The path to load the json file from.

			Returns:
				dict: The target stain metadata.
		"""
		with open(json_pth, "r") as json_file:
			input_json = json.load(json_file)

		self.target_dict = {}
		for key in input_json:
			self.target_dict[key] = np.array(input_json[key])
		self.logger.debug("Target metadata: {}".format(self.target_dict))
		return self.target_dict

	def _st_transform(self, tile_arr:np.ndarray, translation_function:Callable, source_dict:dict, scaling_coefficients:np.ndarray) -> np.ndarray:
		"""	Performs the stain translation in a single thread.

			Args:
				tile_arr (np.ndarray): The tile array represented as stain concentrations.
				translation_function (Callable): The function to use to perform the stain translation.
				scaling_coefficients (np.ndarray): The scaling coefficients to use.
	
			Returns:
				np.ndarray: The translated tile array.
		"""
		self.logger.debug("Starting transform...")

		stain_intensities = _get_stain_intensities(tile_arr, source_dict.get("stain_matrix"))
		translated_arr = translation_function(stain_intensities, source_dict.get("stain_matrix"), self.target_dict.get("stain_matrix"), scaling_coefficients)

		return translated_arr

	def _fit(self, tile_arr: np.ndarray) -> Tuple[dict, np.ndarray]:
		"""	Fits the stain matrix and maximum stain intensity to the tile array.
		
			Args:
				tile_arr (np.ndarray): The tile array.
			
			Returns:
				Tuple[dict, np.ndarray]: The stain matrix and maximum stain intensity, and the tile array.
		"""
		lums_arr = rgb_luminosity_standardiser(tile_arr)

		tissue_mask = get_otsu_tissue_mask(lums_arr)

		if np.invert(tissue_mask).sum() == 0:
			self.logger.warning("No tissue detected in tile with OTSU method, using thresholding.")
			tissue_mask = get_tissue_mask(lums_arr)
			if np.invert(tissue_mask).sum() == 0:
				raise ValueError("No tissue detected in tile.")
			
		tissue_pixels = lums_arr[tissue_mask==0]
		stain_matrix = self.stain_component_fn(tissue_pixels)

		stain_intensities = _get_stain_intensities(tissue_pixels, stain_matrix)

		if self.intensity_sampling_method == IntensitySamplingMethod.MEAN:
			statistic = partial(np.mean, axis=0)
		elif self.intensity_sampling_method == IntensitySamplingMethod.MEDIAN:
			statistic = partial(np.median, axis=0)
		elif self.intensity_sampling_method == IntensitySamplingMethod.MIN:
			statistic = partial(np.min, axis=0)
		elif self.intensity_sampling_method == IntensitySamplingMethod.MAX:
			statistic = partial(np.max, axis=0)
		elif self.intensity_sampling_method == IntensitySamplingMethod.PERCENTILE:
			statistic = partial(np.percentile, q=self.intensity_percentile, axis=0)
		
		sampled_intensities = statistic(stain_intensities).reshape(-1, 2)

		sampled_intensities = sampled_intensities.reshape((1, 2))

		fit_dictionary = {"stain_matrix": stain_matrix, "intensity_samples": sampled_intensities}

		return fit_dictionary, lums_arr

	def _normalise(self, tile_arr: np.ndarray) -> np.ndarray:
		"""	Normalises the tile array.

			Args:
				tile_arr (np.ndarray): The tile array.
			
			Returns:
				np.ndarray: The normalised tile array.
		"""
		source_dict, lums_arr = self._fit(tile_arr)
		self.logger.debug("Source metadata: {}".format(source_dict))

		scaling_coefficients = _calculate_scaling_coefficients(source_dict, self.target_dict, self.logger)
	
		normalised_arr = self._st_transform(lums_arr, _transform_stain, source_dict, scaling_coefficients)

		return normalised_arr

	def convert_to_stain1(self, tile_arr:np.ndarray, use_source:bool=False) -> np.ndarray:
		"""	Converts the tile to only display stain one in RGB.

			Args:
				tile_arr (np.ndarray): The tile array.
				use_source (bool): Whether to use the source stain matrix or the target stain matrix to render stain 1.


			Returns:
				np.ndarray: The converted tile array.
		"""
		global USE_SOURCE
		USE_SOURCE = use_source

		tile_arr = self._load_path_or_array(tile_arr)

		source_dict, lums_arr = self._fit(tile_arr)
		self.logger.debug("Source metadata: {}".format(source_dict))

		scaling_coefficients = _calculate_scaling_coefficients(source_dict, self.target_dict, self.logger)

		converted_arr = self._st_transform(lums_arr, _translate_to_stain1_rgb, source_dict, scaling_coefficients)

		return converted_arr

	def convert_to_stain2(self, tile_arr:np.ndarray, use_source:bool=False) -> np.ndarray:
		"""	Converts the tile to only display stain two in RGB.

			Args:
				tile_arr (np.ndarray): The tile array.
				use_source (bool): Whether to use the source stain matrix or the target stain matrix to render stain 2.

			Returns:
				np.ndarray: The converted tile array.
		"""
		global USE_SOURCE
		USE_SOURCE = use_source

		tile_arr = self._load_path_or_array(tile_arr)

		source_dict, lums_arr = self._fit(tile_arr)
		self.logger.debug("Source metadata: {}".format(source_dict))

		scaling_coefficients = _calculate_scaling_coefficients(source_dict, self.target_dict, self.logger)

		converted_arr = self._st_transform(lums_arr, _translate_to_stain2_rgb, source_dict, scaling_coefficients)

		return converted_arr

	def convert_to_stain_concentrations(self, tile_arr:np.ndarray, normalise_to_target_intensities=True) -> np.ndarray:
		"""	Converts the tile to a two channel array of stain concentrations.

			Args:
				tile_arr (np.ndarray): The tile array.

			Returns:
				np.ndarray: The converted tile array.
		"""
		global USE_SOURCE
		USE_SOURCE = not normalise_to_target_intensities

		tile_arr = self._load_path_or_array(tile_arr)

		source_dict, lums_arr = self._fit(tile_arr)
		self.logger.debug("Source metadata: {}".format(source_dict))

		scaling_coefficients = _calculate_scaling_coefficients(source_dict, self.target_dict, self.logger)

		converted_arr = self._st_transform(lums_arr, _translate_to_stain_concentrations, scaling_coefficients)

		return converted_arr

class BaseDeconStainAugmenter():
	"""	Augments a tile through stain deconvolution and intensity perturbation.
	
		Args:
			file_loader (Callable): The function to use to load the tile.
			logger (logging.Logger): The logger to use.
			_stain_component_fn (Callable): The function to use to get the stain components.
			sigma1 (float): The maximum value to randomly scale the stain intensity.
			sigma2 (float): The maxium value to randomly offset the stain intensity.
	"""
	def __init__(self, file_loader:Callable, logger:logging.Logger, _stain_component_fn:Callable, sigma1:float=0.3, sigma2:float=0.1):
		self.file_loader = file_loader
		self.logger = logger
		self._stain_component_fn = _stain_component_fn
		self.sigma1 = sigma1
		self.sigma2 = sigma2
		self.tile_dict = None

	def save_fit(self, json_pth: Path) -> None:
		"""	Saves the target stain metadata to a json file.

			Args:
				json_pth (Path): The path to save the json file to.

			Returns:
				None
		"""
		output_json = {}

		for key in self.tile_dict:
			output_json[key] = self.tile_dict[key].tolist()

		with open(json_pth, "w") as json_file:
			json.dump(output_json, json_file)

	def load_fit(self, json_pth: Path) -> None:
		"""	Loads the target stain metadata from a json file.

			Args:
				json_pth (Path): The path to load the json file from.

			Returns:
				dict: The target stain metadata.
		"""
		with open(json_pth, "r") as json_file:
			input_json = json.load(json_file)

		self.tile_dict = {}
		for key in input_json:
			self.tile_dict[key] = np.array(input_json[key])
		return self.tile_dict

	def _load_path_or_array(self, path_or_array) -> np.ndarray:
		"""	Loads the tile array from a path or an array.
		
			Args:
				path_or_array (Or[str, Path, np.ndarray]): The path to the tile or the tile array.
		
			Returns:
				np.ndarray: The tile array.
		"""
		if type(path_or_array) == str or type(path_or_array) == Path:
			slide_arr = self.file_loader(path_or_array)
		elif type(path_or_array) == np.ndarray:
			slide_arr = path_or_array
		else:
			raise ValueError("Invalid input type.")
		
		return slide_arr

	def fit(self, tile_arr: np.ndarray) -> Tuple[dict, np.ndarray]:
		"""	Retrieves the stain components for the given tile.

			Args:
				tile_arr (np.ndarray): The tile array.

			Returns:
				dict: The stain matrix.
		"""
		tile_arr = self._load_path_or_array(tile_arr)

		lums_arr = rgb_luminosity_standardiser(tile_arr)
		tissue_mask = get_tissue_mask(lums_arr)
		tissue_pixels = lums_arr[tissue_mask==0]
		stain_matrix = self._stain_component_fn(tissue_pixels)
		
		self.stain_intensities = _get_stain_intensities(lums_arr, stain_matrix)

		self.tile_dict = {"stain_matrix": stain_matrix}

		return self.tile_dict

	def augment_intensity(self, tile_arr: np.ndarray, stain_to_augment:int=-1) -> np.ndarray:
		"""	Augments the tile stain intensity.

			Args:
				tile_arr (np.ndarray): The tile array.
				stain_to_augment (int): (-1,0,1) The stain to augment. If -1, all stains are augmented.
			
			Returns:
				np.ndarray: The converted tile array.
		"""
		if self.tile_dict is None:
			_ = self.fit(tile_arr)
		else:
			tile_arr = self._load_path_or_array(tile_arr)
			lums_arr = rgb_luminosity_standardiser(tile_arr)
			self.stain_intensities = _get_stain_intensities(lums_arr, self.tile_dict["stain_matrix"])

		intensities = self.stain_intensities

		if stain_to_augment == -1:
			no_stains = intensities.shape[-1]
			for idx in range(no_stains):
				alpha = np.random.uniform(1 - self.sigma1, 1 + self.sigma1)
				beta = np.random.uniform(-self.sigma2, self.sigma2)
		
				intensities[:, :, idx] *= alpha
				intensities[:, :, idx] += beta
		else:
			alpha = np.random.uniform(1 - self.sigma1, 1 + self.sigma1)
			beta = np.random.uniform(-self.sigma2, self.sigma2)
	
			intensities[:, :, stain_to_augment] *= alpha
			intensities[:, :, stain_to_augment] += beta

		OD = intensities @ self.tile_dict["stain_matrix"]
		rgb = _OD_to_RGB(OD)

		return rgb

	def augment_colour(self, tile_arr: np.ndarray, stain_to_augment:int=-1) -> np.ndarray:
		"""	Augments the tile stain colour profile.

			Args:
				tile_arr (np.ndarray): The tile array.
			
			Returns:
				np.ndarray: The converted tile array.
		"""
		if self.tile_dict is None:
			_ = self.fit(tile_arr)
		else:
			tile_arr = self._load_path_or_array(tile_arr)
			lums_arr = rgb_luminosity_standardiser(tile_arr)
			self.stain_intensities = _get_stain_intensities(lums_arr, self.tile_dict["stain_matrix"])

		stain_dict = self.tile_dict["stain_matrix"]
		scale = np.random.uniform(1 - self.sigma1, 1 + self.sigma1, (2,3))
		shift = np.random.uniform(-self.sigma2, self.sigma2, (2,3))
		stain_dict = stain_dict * scale + shift
		stain_dict = np.clip(stain_dict, 0, 1)

		if stain_to_augment == -1:

			OD = self.stain_intensities @ stain_dict

		else:
			
			new_stain_dict = self.tile_dict["stain_matrix"]
			new_stain_dict[stain_to_augment] = stain_dict[stain_to_augment]

			OD = self.stain_intensities @ stain_dict
		
		rgb = _OD_to_RGB(OD)

		return rgb