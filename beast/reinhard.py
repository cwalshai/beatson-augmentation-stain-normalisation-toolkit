import cv2
import logging
import numpy as np
from typing import Callable
from multiprocessing.pool import ThreadPool
from .log_utils import get_logger
from .base_classes.normaliser import BaseNormaliser
from .file_readers import read_wsi, read_tile
from .utils import lab_luminosity_standardiser, _get_channel_stats, generate_tile_coordinates, get_tissue_mask, get_otsu_tissue_mask

class ArrayTypeExcpeiton(Exception):
	""" Raised when a numpy array is not of type UInt8. """
	def __init__(self, message="Numpy array is not of type UInt8."):
		super().__init__(self.message)

SHARED_LAB = None

def _apply_channel(channel_arr:np.ndarray, channel_no:int, src_meta:dict, tgt_meta:dict) -> np.ndarray:
	"""	Applies the normalisation to a single channel. 

		Args:
			channel_arr (np.ndarray): The channel array to be normalised.
			channel_no (int): The channel number.
			src_meta (dict): The source metadata.
			tgt_meta (dict): The target metadata.

		Returns:
			np.ndarray: The normalised channel array.
	"""
	channel_arr = channel_arr.astype(np.float32)
	channel_arr = (((channel_arr-src_meta.get("means")[channel_no])/src_meta.get("stdevs")[channel_no])*tgt_meta.get("stdevs")[channel_no])+tgt_meta.get("means")[channel_no]
	channel_arr = np.clip(channel_arr, 0, 255).astype(np.uint8)
	return channel_arr

def _normalise_mp(args:list) -> None:
	"""	Multiprocessing function to map to a list of arguments normalise a tile.

		Args:
			args (list): A list of arguments.

		Returns:
			None
	"""
	global SHARED_LAB
	tile_coords, src_meta, tgt_meta = args

	lab_patch = SHARED_LAB[tile_coords.ystart:tile_coords.yend,tile_coords.xstart:tile_coords.xend,:]

	L = _apply_channel(lab_patch[:,:,0], 0, src_meta, tgt_meta)
	A = _apply_channel(lab_patch[:,:,1], 1, src_meta, tgt_meta)
	B = _apply_channel(lab_patch[:,:,2], 2, src_meta, tgt_meta)
	lab_stacked = np.dstack((L,A,B))

	SHARED_LAB[tile_coords.ystart:tile_coords.yend,tile_coords.xstart:tile_coords.xend,:] = lab_stacked

def _normalise_tile(tile:np.ndarray, src_meta, tgt_meta) -> None:
	"""	Normalises a tile.

		Args:
			tile (np.ndarray): The tile to be normalised.
			src_meta (dict): The source metadata.
			tgt_meta (dict): The target metadata.

		Returns:
			np.ndarray: The normalised tile.
	"""
	L = _apply_channel(tile[:,:,0], 0, src_meta, tgt_meta)
	A = _apply_channel(tile[:,:,1], 1, src_meta, tgt_meta)
	B = _apply_channel(tile[:,:,2], 2, src_meta, tgt_meta)
	lab_stacked = np.dstack((L,A,B))

	return lab_stacked

class ReinhardWSINormaliser(BaseNormaliser):
	"""	Class to transform the colour profile of a whole slide image to match a target using the Reinhard Colour Transfer Algorithm (https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf). Warning: This class considers the colour profile of the entire slide when normalising, this is useful for visualisation of the slide but may not be appropriate for analysis when stain intensity can vary across the slide. For analysis, use the ReinhardTileNormaliser or ReinhardWSITesselatedNormaliser class.

		Normalistion is carried out by standardising the luminosity of the source image, then calculating the mean and standard deviation of the channels in the LAB colour space. These channels are then scaled across the whole slide to match a target colour profile.

		Args:
			file_loader (Callable, optional): The function to use to load the file. Defaults to read_wsi. A function that accepts a path and returns a numpy array can be supplied for custom loading.
			logfile (str, optional): The path to the log file. Defaults to None.
			terminal_log_level (int, optional): The log level for the terminal. Defaults to logging.DEBUG.
			file_log_level (int, optional): The log level for the log file. Defaults to logging.DEBUG.
			num_threads (int, optional): The number of threads to use. Defaults to 16.

		Raises:
			ArrayTypeExcpeiton: Raised when a numpy array is not of type UInt8.
	"""
	def __init__(self, file_loader:Callable=read_wsi, logfile=None, terminal_log_level=logging.DEBUG, file_log_level=logging.DEBUG, num_threads:int=16) -> None:
		logger = get_logger("Reinhard WSI Normaliser", logfile, terminal_log_level, file_log_level)
		super().__init__(file_loader, logger)
		self.num_threads = num_threads

	def _normalise_channels(self, LAB:np.array, src_meta:dict, tgt_meta:dict) -> np.array:
		"""	Normalises the channels of a LAB image.

			Args:
				LAB (np.array): The LAB image to be normalised.
				src_meta (dict): The source metadata.
				tgt_meta (dict): The target metadata.

			Returns:
				np.array: The normalised LAB image.

			Raises:
				ArrayTypeExcpeiton: Raised when the input array is not of type UInt8.
		"""
		global SHARED_LAB
		SHARED_LAB = None

		if LAB.dtype != np.uint8:
			raise ArrayTypeExcpeiton()
		
		dimensions = (LAB.shape[1], LAB.shape[0])
		tile_coordinate_list = generate_tile_coordinates(dimensions)
		args = [(tile_coords, src_meta, tgt_meta) for tile_coords in tile_coordinate_list]

		SHARED_LAB = LAB

		with ThreadPool(self.num_threads) as p:
			p.map(_normalise_mp, args, chunksize=1)

		return SHARED_LAB

	def _fit(self, slide_arr:np.ndarray):
		""" Fits the normaliser to the source image.

			Args:
				slide_arr (np.ndarray): The source image.
			
			Returns:
				dict: The source metadata.
				slide_arr (np.ndarray): The source image in LAB colour space with standardised luminosity.
		"""
		lab_slide_arr = cv2.cvtColor(slide_arr, cv2.COLOR_RGB2LAB)
		lab_slide_arr = lab_luminosity_standardiser(lab_slide_arr)

		sampled_lab_pixels = self._sample_wsi(lab_slide_arr)

		mask = get_otsu_tissue_mask(sampled_lab_pixels, convert_to_lab=False)

		means = _get_channel_stats(sampled_lab_pixels, mask)
		stdevs = _get_channel_stats(sampled_lab_pixels, mask, statistic=np.std)

		return {"means":means, "stdevs":stdevs}, lab_slide_arr

	def _normalise(self, slide_arr:np.ndarray) -> np.ndarray:
		"""	Normalises the input image.

			Args:
				slide_arr (np.ndarray): The image to be normalised.
			
			Returns:
				np.ndarray: The normalised image.
		"""
		self.source_dict, lab_slide_arr = self._fit(slide_arr)
		self.logger.info(f"Source metadata: {self.source_dict}")

		normalised_lab = self._normalise_channels(lab_slide_arr, self.source_dict, self.target_dict)

		normalised_rgb = cv2.cvtColor(normalised_lab, cv2.COLOR_LAB2RGB)
		return normalised_rgb

class ReinhardTileNormaliser(BaseNormaliser):
	"""	Class to transform the colour profile of an image tile to match a target using the Reinhard Colour Transfer Algorithm (https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf).

		Normalisation is performed by computing the mean and standard deviation of the channels in LAB colour space for the source and target images. The mean and standard deviation of the source image is then scaled to match the target image.

		Args:
			file_loader (Callable, optional): The function to load the tile. Defaults to read_tile. A function that accepts a path and returns a numpy array can be supplied by the user for custom loading.
			logfile (str, optional): The path to the log file. Defaults to None.
			terminal_log_level (int, optional): The level of logging to the terminal. Defaults to logging.DEBUG.
			file_log_level (int, optional): The level of logging to the file. Defaults to logging.DEBUG.
	"""
	def __init__(self, file_loader:Callable=read_tile, logfile=None, terminal_log_level=logging.DEBUG, file_log_level=logging.DEBUG) -> None:
		logger = get_logger("Reinhard WSI Normaliser", logfile, terminal_log_level, file_log_level)
		super().__init__(file_loader, logger)

	def _fit(self, tile_arr:np.ndarray):
		"""	Fits the normaliser to the source image.

			Args:
				tile_arr (np.ndarray): The source image.

			Returns:
				dict: The source metadata.
				slide_arr (np.ndarray): The source image in LAB colour space with standardised luminosity.
		"""
		lab_tile_arr = cv2.cvtColor(tile_arr, cv2.COLOR_RGB2LAB)
		lab_tile_arr = lab_luminosity_standardiser(lab_tile_arr)

		tissue_mask = get_otsu_tissue_mask(lab_tile_arr, convert_to_lab=False)

		means = _get_channel_stats(lab_tile_arr, tissue_mask)
		stdevs = _get_channel_stats(lab_tile_arr, tissue_mask, statistic=np.std)

		return {"means":means, "stdevs":stdevs}, lab_tile_arr

	def _normalise(self, tile_arr:np.ndarray) -> np.ndarray:
		"""	Normalises the input image.

			Args:
				tile_arr (np.ndarray): The image to be normalised.

			Returns:
				np.ndarray: The normalised image.
		"""
		self.source_dict, lab_slide_arr = self._fit(tile_arr)
		normalised_lab = _normalise_tile(lab_slide_arr, self.source_dict, self.target_dict)
		normalised_rgb = cv2.cvtColor(normalised_lab, cv2.COLOR_LAB2RGB)
		return normalised_rgb