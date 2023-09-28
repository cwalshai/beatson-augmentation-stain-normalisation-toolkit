import json
import numpy as np
from pathlib import Path
from typing import List
from enum import Enum
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from functools import partial
from .reinhard import ReinhardTileNormaliser, ReinhardWSINormaliser
from .macenko import MacenkoTileNormaliser, MacenkoWSINormaliser
from .vahadane import VahadaneTileNormaliser, VahadaneWSINormaliser
from .file_writers import write_tile
from .utils import generate_tile_coordinates
from .utils import get_tissue_mask, rgb_luminosity_standardiser, AggregationMethod, TileCoordinate, get_background_mask

NORM_INST = None
SHARED_SLIDE_ARR = None

def _normalisation_wrapper1(target:Path, output:Path, discard_background:bool, background_threshold:float) -> None:
	tile_arr = NORM_INST._load_path_or_array(target)

	tile_mask = get_background_mask(tile_arr)
	background_per = np.mean(tile_mask)
	
	NORM_INST.logger.debug("Discard Background: " + str(discard_background))

	if discard_background:
		if background_per >= background_threshold:
			NORM_INST.logger.debug(f"Discarding tile {target.name}_{background_per}")
			return

	if background_per >=  background_threshold:
		tile_arr = rgb_luminosity_standardiser(tile_arr)
	else:
		tile_arr = NORM_INST.normalise(tile_arr)

	write_tile(tile_arr, output)

def _normalisation_wrapper2(coord:TileCoordinate, output_dir:Path, discard_background:bool, background_threshold:float, tile_ext:str) -> None:
	global SHARED_SLIDE_ARR
	tile_arr = SHARED_SLIDE_ARR[coord.ystart:coord.yend, coord.xstart:coord.xend, :]

	tile_mask = get_background_mask(tile_arr)
	background_per = np.mean(tile_mask)
	
	if discard_background:
		if background_per >= background_threshold:
			NORM_INST.logger.debug(f"Discarding tile {coord.ystart}_{coord.xstart}_{coord.yend}_{coord.xend}")
			return

	if background_per >=  background_threshold:
		tile_arr = rgb_luminosity_standardiser(tile_arr)
	else:
		tile_arr = NORM_INST.normalise(tile_arr)
	
	write_tile(tile_arr, output_dir / f"{coord.ystart}_{coord.xstart}_{coord.yend}_{coord.xend}.{tile_ext}")
	SHARED_SLIDE_ARR[coord.ystart:coord.yend, coord.xstart:coord.xend, :] = tile_arr

class TesselatedWSINormaliser():
	"""
		Class to normalise a WSI through tesselation and normalisation of the individual tiles.

		The tiles will then be written out to form a dataset of normalised tiles optionally the whole WSI can be reconstructed and returned.

		Background tiles can be discarded by setting the discard_background to "threshold".
		
		The background_threshold parameter determines the percentage of background pixels present in the tile for the tile to be discarded.

		Tiles will be written out in the format ystart_xstart_yend_xend.png

		Args:
			normalistion_class: The normalisation class to use. Must be one of ReinhardTileNormaliser, MacenkoTileNormaliser, VahadaneTileNormaliser
			discard_background: The method to use to discard background. Must be one of None, "threshold"
			background_threshold (float): The percentage of pixels in a tile that must be background for the tile to be discarded. Only used when discard_background="otsu"
			discard_non_conforming (bool): Whether to discard tiles that do not conform to the target tile size. (Default: True)
			num_threads (int): The number of threads to use when tesselating and normalising the WSI.
			tile_ext (str): The extension to use for the tiles. (Default: ".png")
			**kwargs: Additional keyword arguments to pass to the normalisation class. See the documentation for the normalisation class for more details.
	"""
	def __init__(self, normalistion_class: ReinhardTileNormaliser | MacenkoTileNormaliser | VahadaneTileNormaliser = MacenkoTileNormaliser, discard_background:str=None, background_threshold:float=0.9, num_threads:int=8, discard_non_conforming:bool=True, tile_ext="png", **kwargs) -> None:
		assert normalistion_class in [ReinhardTileNormaliser, MacenkoTileNormaliser, VahadaneTileNormaliser], "normalistion_class must be one of ReinhardTileNormaliser, MacenkoTileNormaliser, VahadaneTileNormaliser"
		assert discard_background in [None, "threshold"], "discard_background must be one of None, 'threshold'"
		self.normalistion_instance = normalistion_class(**kwargs, terminal_log_level="ERROR")
		self.normalistion_class = normalistion_class
		self.discard_background = discard_background
		self.background_threshold = background_threshold
		self.num_threads = num_threads
		self.discard_non_conforming = discard_non_conforming
		self.tile_ext = tile_ext.replace(".", "")
		self.stain_metadata = None

	def _filter_tiles_by_size(self, tile_list:List[TileCoordinate], tile_size:tuple) -> List[TileCoordinate]:
		"""
			Filters a list of tiles by size.

			Args:
				tile_list (List[Path]): The list of tiles to filter.
				tile_size (tuple): The size of the tiles to filter by.

			Returns:
				List[Path]: The filtered list of tiles.
		"""
		target_x = tile_size[0]
		target_y = tile_size[1]
		filtered_list = []
		for tileCoord in tile_list:
			
			x_size = tileCoord.xend - tileCoord.xstart
			y_size = tileCoord.yend - tileCoord.ystart

			if x_size == target_x and y_size == target_y:
				filtered_list.append(tileCoord)
		return filtered_list

	def normalise(self, slide_path:Path, output_dir:Path, tile_size:tuple=(224,224), return_slide:bool=False) -> None:
		"""
			Tesselates a WSI and normalises each tile.

			Args:
				slide_path (Path): The path to the WSI to normalise.
				output_dir (Path): The directory to write the normalised tiles to.
				tile_size (tuple): The size of the tiles to extract from the WSI.
				return_slide (bool): Whether to return the normalised slide. (Default: False)
		"""
		global NORM_INST, SHARED_SLIDE_ARR
		output_dir = Path(output_dir)
		output_dir.mkdir(parents=True, exist_ok=True)

		assert self.stain_metadata is not None, "The target stain metadata must be loaded before tesselating and normalising a WSI."
		self.normalistion_instance.target_dict = self.stain_metadata
		
		slide_arr = self.normalistion_instance.file_loader(slide_path)
		slide_dims = slide_arr.shape

		self.normalistion_instance.logger.info(f"loaded slide with dimensions {slide_dims}.")

		coordinate_list = generate_tile_coordinates((slide_dims[1], slide_dims[0]), tile_size)

		original_len = len(coordinate_list)
		if self.discard_non_conforming:
			self.normalistion_instance.logger.info(f"filtering non-conforming tiles.")
			coordinate_list = self._filter_tiles_by_size(coordinate_list, tile_size)
			filtered_len = len(coordinate_list)
			self.normalistion_instance.logger.debug(f"removed {original_len - filtered_len} non-conforming tiles.")

		args = [(coord, output_dir, self.discard_background, self.background_threshold, self.tile_ext) for coord in coordinate_list]

		NORM_INST = self.normalistion_instance
		SHARED_SLIDE_ARR = slide_arr

		self.normalistion_instance.logger.info("Starting Parallel Normalisation.")
		with ThreadPool(self.num_threads) as p:
			p.starmap(_normalisation_wrapper2, args)
		self.normalistion_instance.logger.info("Normalisation complete.")

		if return_slide:
			return slide_arr
		else:
			SHARED_SLIDE_ARR = None
			return None
	
	def load_fit(self, json_path: Path) -> None:
		"""
			Loads the target stain metadata from a json file.

			Args:
				json_pth (Path): The path to load the json file from.

			Returns:
				dict: The target stain metadata.
		"""
		with open(json_path, "r") as json_file:
			input_json = json.load(json_file)

		self.stain_metadata = {}
		for key in input_json:
			self.stain_metadata[key] = np.array(input_json[key])

		return self.stain_metadata

class StainMetadataEstimator():
	"""
		Class to estimate the stain metadata from a single tile or WSI or a list of them.

		If a list of sources is provided, the stain matrix aggregation method can be set to one of "mean", "median", "max", "min" or "percentile".

		Args:
			normalistion_class: The normalisation class to use. Must be one of ReinhardTileNormaliser, MacenkoTileNormaliser, VahadaneTileNormaliser, ReinhardWSINormaliser, MacenkoWSINormaliser, VahadaneWSINormaliser
			aggregation_method (AggregationMethod): The method to use to aggregate the stain metadata from a list of sources. (Default: AggregationMethod.MEDIAN)
			percentile (float): The percentile to use when aggregating the stain metadata. Only used when aggregation_method=AggregationMethod.PERCENTILE
			**kwargs: Additional keyword arguments to pass to the normalisation class. See the documentation for the normalisation class for more details.
	"""
	def __init__(self, normalistion_class: ReinhardTileNormaliser | MacenkoTileNormaliser | VahadaneTileNormaliser | ReinhardWSINormaliser | MacenkoWSINormaliser | VahadaneWSINormaliser = MacenkoTileNormaliser, aggregation_method:AggregationMethod=AggregationMethod.MEDIAN, aggregation_percentile:float=0.75, discard_background:bool=True, background_threshold:float=0.9, **kwargs) -> None:
		assert normalistion_class in [ReinhardTileNormaliser, MacenkoTileNormaliser, VahadaneTileNormaliser, ReinhardWSINormaliser, MacenkoWSINormaliser, VahadaneWSINormaliser], "normalistion_class must be one of ReinhardTileNormaliser, MacenkoTileNormaliser, VahadaneTileNormaliser, ReinhardWSINormaliser, MacenkoWSINormaliser, VahadaneWSINormaliser"
		self.normalisation_class = normalistion_class
		self.normalistion_instance = normalistion_class(**kwargs)
		self.stain_metadata = None
		self.aggregation_method = aggregation_method
		self.aggregation_percentile = aggregation_percentile
		self.discard_background = discard_background
		self.background_threshold = background_threshold

	def _estimate_from_list(self, target: List[Path]) -> dict:
		"""
			Estimate the stain metadata from a list of sources.

			Args:
				target (List[Path]): A list of paths to the sources to estimate the stain metadata from.

			Returns:
				dict: The estimated stain metadata.

			Raises:
				Exception: If the aggregation method is set to AggregationMethod.PERCENTILE and the percentile is not between 0 and 1.
		"""
		metadata_list = []
		total_l = len(target)
		for idx, path in enumerate(target):
			self.normalistion_instance.logger.info(f"{idx+1}/{total_l} Estimating stain metadata from {path}")

			if self.normalisation_class in [ReinhardTileNormaliser, MacenkoTileNormaliser, VahadaneTileNormaliser]:
				tile_arr = self.normalistion_instance._load_path_or_array(path)
				background_mask = get_background_mask(tile_arr)
				background_per = np.mean(background_mask)
					
				if self.discard_background:
					if background_per >= self.background_threshold:
						self.normalistion_instance.logger.debug(f"Discarding {path.name}_{background_per}")
						continue
				
				path = tile_arr

			fit_dict = self._estimate_from_path_or_arr(path)
			self.normalistion_instance.logger.debug(f"Stain metadata: {fit_dict}")
			metadata_list.append(fit_dict)

		if self.aggregation_method == AggregationMethod.MEAN:
			statistic = partial(np.mean, axis=0)
		elif self.aggregation_method == AggregationMethod.MEDIAN:
			statistic = partial(np.median, axis=0)
		elif self.aggregation_method == AggregationMethod.MAX:
			statistic = partial(np.max, axis=0)
		elif self.aggregation_method == AggregationMethod.MIN:
			statistic = partial(np.min, axis=0)
		elif self.aggregation_method == AggregationMethod.PERCENTILE:
			statistic = partial(np.percentile, q=self.aggregation_percentile, axis=0)


		if type(self.normalistion_instance) is ReinhardWSINormaliser or type(self.normalistion_instance) is ReinhardTileNormaliser:
			stain_means = np.array([metadata["means"] for metadata in metadata_list])
			stain_stds = np.array([metadata["stdevs"] for metadata in metadata_list])

			stain_mean = statistic(stain_means)
			stain_std = statistic(stain_stds)
			return {"means": stain_mean, "stdevs": stain_std}
		else:
			stain_matrices = np.array([metadata["stain_matrix"] for metadata in metadata_list])
			stain_maxes = np.array([metadata["intensity_samples"] for metadata in metadata_list])

			stain_matrix = statistic(stain_matrices)
			stain_max = statistic(stain_maxes)
			return {"stain_matrix": stain_matrix, "intensity_samples": stain_max}

	
	def _estimate_from_path_or_arr(self, target: Path) -> np.ndarray:
		"""
			Estimates the stain metadata from a single tile or WSI.

			Args:
				target (Path): The path to the tile or WSI.
			
			Returns:
				dict: The estimated stain metadata.
		"""
		fit_dict = self.normalistion_instance.fit(target)
		return fit_dict	

	def estimate(self, target: List[Path] | Path) -> dict:
		"""
			Estimates the stain metadata from a single tile or WSI or a list of them.

			Args:
				target (List[Path] | Path): The path to the tile or WSI or a list of them.

			Returns:
				dict: The estimated stain metadata.
		"""
		
		if isinstance(target, list):
			self.stain_metadata = self._estimate_from_list(target)
			return self.stain_metadata
		else:
			self.stain_metadata = self._estimate_from_path_or_arr(target)
			return self.stain_metadata

	def save_fit(self, json_path: Path) -> None:
		"""
			Saves the target stain metadata to a json file.

			Args:
				json_path (Path): The path to save the json file to.

			Returns:
				None
		"""
		assert self.stain_metadata is not None, "No stain metadata to save. Please run estimate() first."

		output_json = {}

		for key in self.stain_metadata:
			output_json[key] = self.stain_metadata[key].tolist()

		with open(json_path, "w") as json_file:
			json.dump(output_json, json_file)

class ParallelTileNormaliser():
	"""
		Class to normalise a collection of tiles in parallel.

		Args:
			normalistion_class: The normalisation class to use. Must be one of ReinhardTileNormaliser, MacenkoTileNormaliser, VahadaneTileNormaliser
			num_threads (int): The number of threads to use for parallel processing. (Default: 8)
			**kwargs: The keyword arguments to pass to the normalisation class. See the documentation for the normalisation class for more information.
	"""
	def __init__(self, normalistion_class: ReinhardTileNormaliser | MacenkoTileNormaliser | VahadaneTileNormaliser = MacenkoTileNormaliser, discard_background:bool=False, background_threshold:float=0.9, num_threads:int=8, **kwargs) -> None:
		assert normalistion_class in [ReinhardTileNormaliser, MacenkoTileNormaliser, VahadaneTileNormaliser], "normalistion_class must be one of ReinhardTileNormaliser, MacenkoTileNormaliser, VahadaneTileNormaliser"
		self.normalistion_instance = normalistion_class(**kwargs)
		self.stain_metadata = None
		self.discard_background = discard_background
		self.background_threshold = background_threshold
		self.num_threads = num_threads

	def normalise(self, targets:List[Path], output:List[Path] | Path) -> np.ndarray:
		"""
			Normalises a collection of tiles.

			Args:
				targets (List[Path]): The list of paths to the tiles to normalise.
				output (List[Path] | Path): The list of paths to save the normalised tiles to. If a single path to a directory is provided, the tiles will be saved in that directory.
		"""
		global NORM_INST

		assert self.stain_metadata is not None, "No stain metadata has been set. Please set the stain metadata using the load_fit method or the estimate method."
		self.normalistion_instance.target_dict = self.stain_metadata

		if isinstance(output, list):
			output_paths = output
		else:
			output_path = Path(output)
			output_path.mkdir(parents=True, exist_ok=True)

		output_paths = []
		for target in targets:
			target = Path(target)
			output_paths.append(output_path / target.name)

		NORM_INST = self.normalistion_instance
		
		self.normalistion_instance.logger.info("Starting Parallel Normalisation.")

		args = []
		for target, output_path in zip(targets, output_paths):
			args.append((target, output_path, self.discard_background, self.background_threshold))

		with ThreadPool(self.num_threads) as p:
			p.starmap(_normalisation_wrapper1, args)

		self.normalistion_instance.logger.info("Normalisation complete.")

	def load_fit(self, json_path: Path) -> None:
		"""
			Loads the target stain metadata from a json file.

			Args:
				json_pth (Path): The path to load the json file from.

			Returns:
				dict: The target stain metadata.
		"""
		with open(json_path, "r") as json_file:
			input_json = json.load(json_file)

		self.stain_metadata = {}
		for key in input_json:
			self.stain_metadata[key] = np.array(input_json[key])

		return self.stain_metadata