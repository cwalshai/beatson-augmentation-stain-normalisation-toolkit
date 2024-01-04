import cv2
import numpy as np
from typing import List, NamedTuple
from enum import Enum

OD_INTENSITY_MAX = -np.log((1/255))

class TileCoordinate(NamedTuple):
	""" NamedTuple for tile coordinates. """
	xstart: int
	ystart: int
	xend: int
	yend: int

class TileDimensions(NamedTuple):
	"""	NamedTuple for tile dimensions. """
	x: int
	y: int

class AggregationMethod(Enum):
	""" Enum to define the method to use when aggregating the stain metadata from a list of sources. """
	MEAN = 0
	MEDIAN = 1
	MIN = 2
	MAX = 3
	PERCENTILE = 4

class IntensitySamplingMethod(Enum):
	"""	Enum to specify the method to use to sample the stain intensity. """
	MEAN = 1
	MEDIAN = 2
	MIN = 3
	MAX = 4
	PERCENTILE = 5

def generate_tile_coordinates(dimensions:tuple, tile_size:tuple=(5000,5000), xstart:int=0, ystart:int=0) -> List[TileCoordinate]:
	"""	Generate a list of tile coordinates for a given slide dimensions and patch size.

		Args:
			dimensions <Tuple>: Tuple of the slide dimensions (x,y).
			patch_size <Tuple>: Tuple of the tile size (x,y).
			xstart <Integer>: Starting x coordinate.
			ystart <Integer>: Starting y coordinate.

		Returns:
			List of tile coordinates <TileCoordinate>.
	"""
	tile_coordinates = []

	xmax = dimensions[0]
	ymax = dimensions[1]
	tilex = tile_size[0]
	tiley = tile_size[1]

	ydiff = ymax % tiley
	xdiff = xmax % tilex

	ymain_end = (ymax - ydiff)
	xmain_end = (xmax - xdiff)

	for y in range(ystart, ymain_end, tiley):
		for x in range(xstart, xmain_end, tilex):
			coordinate = TileCoordinate(x, y, x+tilex, y+tiley)
			tile_coordinates.append(coordinate)

	if ydiff != 0:
		for x in range(xstart, xmain_end, tilex):
			coordinate = TileCoordinate(x, ymain_end, x+tilex, ymax)
			tile_coordinates.append(coordinate)

	if xdiff != 0:
		for y in range(ystart, ymain_end, tiley):
			coordinate = TileCoordinate(xmain_end, y, xmax, y+tiley)
			tile_coordinates.append(coordinate)

	if ydiff != 0 and xdiff != 0:
		coordinate = TileCoordinate(xmain_end, ymain_end, xmax, ymax)
		tile_coordinates.append(coordinate)
				
	return tile_coordinates

def lab_luminosity_standardiser(lab_arr:np.ndarray, percentile=98) -> np.ndarray:
	"""	Standardise the luminosity of an LAB image.

		Args:
			lab_arr <np.ndarray>: LAB image.

		Returns:
			Standardised LAB image <np.ndarray>.
	"""
	L = lab_arr[:, :, 0].astype(float)
	p = np.percentile(L, percentile)
	lab_arr[:, :, 0] = np.clip(255 * (L / p), 0, 255).astype(np.uint8)
	return lab_arr

def rgb_luminosity_standardiser(rgb:np.ndarray, percentile=98) -> np.ndarray:
	"""	Standardise the luminosity of an RGB image.

		Args:
			rgb <np.ndarray>: RGB image.

		Returns:
			Standardised RGB image <np.ndarray>.
	"""
	LAB_ARR = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
	LAB_ARR = lab_luminosity_standardiser(LAB_ARR, percentile)
	rgb = cv2.cvtColor(LAB_ARR, cv2.COLOR_LAB2RGB)
	return rgb

def _get_channel_stats(lab_arr:np.ndarray, tissue_mask:np.ndarray, pixel_class:int=0, statistic=np.mean) -> tuple:
	"""
		Get the mean L, A and B values for a given pixel class.

		Args:
			lab_arr <np.ndarray>: LAB image.
			tissue_mask <np.ndarray>: Tissue mask.
			pixel_class <Integer>: Pixel class.
			statistic <Function>: Statistic to use.

		Returns:
			Mean L, A and B values <Tuple>.
	"""
	L = lab_arr[:,:,0]
	A = lab_arr[:,:,1]
	B = lab_arr[:,:,2]

	L = L[tissue_mask==pixel_class]
	A = A[tissue_mask==pixel_class]
	B = B[tissue_mask==pixel_class]

	L_mean = statistic(L)
	A_mean = statistic(A)
	B_mean = statistic(B)

	return L_mean, A_mean, B_mean

def get_tissue_mask(tile_arr:np.ndarray, luminosity_threshold:float=0.8, convert_to_lab=True) -> np.ndarray:
	""" Generate a tissue mask for a given tile.

		Args:
			tile_arr <np.ndarray>: Tile array.
			luminosity_threshold <Float>: Luminosity threshold.

		Returns:
			Tissue mask <np.ndarray>.

		Notes:
			- Luminosity threshold is a value between 0 and 1.
			- Tissue mask is a binary mask with 0 for tissue and 1 for background.
	"""
	if convert_to_lab:
		lab_tile = cv2.cvtColor(tile_arr, cv2.COLOR_RGB2LAB)
	else:
		lab_tile = tile_arr
	L = lab_tile[:, :, 0] / 255.0
	mask = L > luminosity_threshold

	return mask.astype(np.uint8)

def get_background_mask(tile_arr:np.ndarray, luminosity_threshold:float=0.8, convert_to_lab=True) -> np.ndarray:
	""" Generate a background mask for a given tile.

		Args:
			tile_arr <np.ndarray>: Tile array.
			luminosity_threshold <Float>: Luminosity threshold.

		Returns:
			Tissue mask <np.ndarray>.

		Notes:
			- Luminosity threshold is a value between 0 and 1.
			- Background mask is a binary mask with 0 for tissue and 1 for background.
	"""
	if convert_to_lab:
		lab_tile = cv2.cvtColor(tile_arr, cv2.COLOR_RGB2LAB)
	else:
		lab_tile = tile_arr

	lab_tile = lab_luminosity_standardiser(lab_tile)

	L = lab_tile[:, :, 0] / 255.0
	mask = L > luminosity_threshold

	return mask.astype(np.uint8)

def get_otsu_tissue_mask(tile_arr:np.ndarray, convert_to_lab=True) -> np.ndarray:
	"""	Generate a tissue mask for a given tile using Otsu's method.

		Args:
			tile_arr <np.ndarray>: Tile array.

		Returns:
			Tissue mask <np.ndarray>.

		Notes:
			- Tissue mask is a binary mask with 0 for tissue and 1 for background.
	"""
	if convert_to_lab:
		lab_tile = cv2.cvtColor(tile_arr, cv2.COLOR_RGB2LAB)
	else:
		lab_tile = tile_arr
	
	L = lab_tile[:, :, 0]
	ret,th3 = cv2.threshold(L,0,1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	
	return th3.astype(np.uint8)

def _RGB_to_OD(img_arr:np.ndarray) -> np.ndarray:
	""" Convert from RGB to optical density (OD_RGB).

		OD_RGB = -1 * log(RGB/255)

		Args:
			img_arr <np.ndarray>: RGB image.

		Returns:
			Optical density image <np.ndarray>. Possible range is 0 to 2.40654
	"""
	assert img_arr.dtype == np.uint8, "Image must be of type np.uint8."
	zero_mask = (img_arr == 0)
	img_arr[zero_mask] = 1
	return -1 * np.log(img_arr / 255)

def _clip_OD(od_arr:np.ndarray) -> np.ndarray:
	"""	Clips the optical density array to the range [0, -np.log((1/255))].

		Args:
			od_arr (np.ndarray): The optical density array.

		Returns:
			np.ndarray: The clipped optical density array.
	"""
	od_arr = np.maximum(od_arr, 0)
	od_arr = np.minimum(od_arr, OD_INTENSITY_MAX)
	return od_arr

def _OD_to_RGB(od_arr:np.ndarray) -> np.ndarray:
	""" Convert from optical density (OD_RGB) to RGB.

		RGB = 255 * exp(-1 * OD_RGB)

		Args:
			od_arr <np.ndarray>: Optical density image.

		Returns:
			RGB image <np.ndarray>.
	"""
	od_arr = _clip_OD(od_arr)
	rgb_norm = np.exp(-1 * od_arr)
	return (255 * rgb_norm).astype(np.uint8)

def _normalize_matrix_rows(array:np.ndarray) -> np.ndarray:
	""" This function normalizes the rows of a 2D array, A, using the L2 norm of each row. The function returns the same array with its rows normalized to have a L2 norm of 1. The calculation is done by dividing each row of A by its L2 norm (computed using the np.linalg.norm function) along the first axis (axis=1) and then adding a new axis to the result using [:, None] to preserve the 2D structure of the input array.

		Args:
			array <np.ndarray>: Matrix to be normalized.

		Returns:
			Normalized matrix <np.ndarray>.
	"""
	return array / np.linalg.norm(array, axis=1)[:, None]