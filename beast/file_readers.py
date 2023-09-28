import numpy as np
import tifffile as tf
from pathlib import Path
from .utils import generate_tile_coordinates, TileDimensions

class SlideReadException(Exception):
	def __init__(self, message="Could not read slide."):
		super().__init__(message)

class TileReadException(Exception):
	def __init__(self, message="Could not read tile."):
		super().__init__(message)

def _read_wsi_by_tessellation(slide_path:Path) -> np.ndarray:
	"""
		Read a whole slide image by tiling it and then stitching the tiles together.

		Args:
			slide_path <pathlib.Path>: Path to the whole slide image.

		Returns:
			Numpy array of the whole slide image.
	"""
	import openslide as ops

	#get slide dimensions from openslide
	slide = ops.OpenSlide(slide_path)
	slide_dims = slide.level_dimensions[0]

	#get coordinates of tiles
	tile_dims = TileDimensions(5000, 5000)
	tile_coords = generate_tile_coordinates(slide_dims, (tile_dims.x, tile_dims.y))
	
	#create new array to hold slide
	slide_array = np.zeros((slide_dims[1], slide_dims[0], 3), dtype=np.uint8)

	#read each tile and add to slide array
	for tile_coord in tile_coords:
		x_size = tile_coord.xend - tile_coord.xstart
		y_size = tile_coord.yend - tile_coord.ystart
		tile = slide.read_region((tile_coord.xstart,tile_coord.ystart), 0, (x_size, y_size))
		tile = tile.convert("RGB")
		tile = np.array(tile)
		slide_array[tile_coord.ystart:tile_coord.yend, tile_coord.xstart:tile_coord.xend, :] = tile

	return slide_array

def read_wsi(slide_pth:Path) -> np.ndarray:
	"""
		Read a whole slide image and return a numpy array.

		Args:
			slide_pth: Path to the whole slide image.

		Returns:
			Numpy array of the whole slide image.
	"""

	try:
		with tf.TiffFile(slide_pth) as tif:
			slide = tif.pages[0].asarray()
			return slide
	except:
		try:
			slide = _read_wsi_by_tessellation(slide_pth)
			return slide
		except Exception as e:
			raise SlideReadException(f"Could not read wsi at {slide_pth}. {e}")

def read_tile(tile_pth:Path) -> np.ndarray:
	"""
		Read a tile image and return a numpy array.

		Args:
			tile_pth: Path to the tile image.

		Returns:
			Numpy array of the tile image.
	"""

	try:
		with tf.TiffFile(tile_pth) as tif:
			tile = tif.pages[0].asarray()
			return tile
	except:
		try:
			from PIL import Image
			tile = Image.open(tile_pth)
			tile = np.array(tile)
			return tile
		except Exception as e:
			raise TileReadException(f"Could not read tile at {tile_pth}. {e}")