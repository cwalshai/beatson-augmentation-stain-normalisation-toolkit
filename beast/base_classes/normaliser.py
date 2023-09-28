import json
import numpy as np
from pathlib import Path, PosixPath, WindowsPath
from logging import Logger
from typing import Callable

class BaseNormaliser():
	"""
		Base class to define common functions for all normalisation classes.
	"""
	def __init__(self, file_loader:Callable, logger:Logger) -> None:
		super().__init__()
		self.logger = logger
		self.file_loader = file_loader
		self.source_dict = None
		self.target_dict = None
		self.tissue_predictor = None
		self.flip_clusters = False

	def save_fit(self, path:Path) -> None:
		"""
			Function to save the target normalisation metadata to a JSON file.

			Args:
				path <str, Path>: Path to save the JSON file to.
		"""
		with open(path, "w") as jfile:
			json.dump(self.target_dict, jfile)
	
	def load_fit(self, path:Path) -> None:
		"""
			Function to load the target normalisation metadata from a JSON file.
		
			Args:
				path <str, Path>: Path to load the JSON file from.
			
		"""
		with open(path, "r") as jfile:
			self.target_dict = json.load(jfile)

	def fit(self, path_or_array) -> None:
		"""
			Function to fit the normalisation to a target whole slide image.

			Args:
				path_or_array <str, Path, np.ndarray>: Path to the whole slide image to fit to, or the array to fit to.
		"""
		if type(path_or_array) == str or type(path_or_array) == PosixPath or type(path_or_array) == WindowsPath:
			slide_arr = self.file_loader(path_or_array)
		elif type(path_or_array) == np.ndarray:
			slide_arr = path_or_array
		else:
			raise ValueError("Path or array must be a string, pathlib.Path or numpy.ndarray")

		self.target_dict, _ = self._fit(slide_arr)
		self.logger.debug(f"Target metadata: {self.target_dict}")
		return self.target_dict

	def _fit(self, slide_arr:np.ndarray) -> dict:
		"""
			To be implemented by child classes.
		"""
		raise NotImplementedError

	def _load_path_or_array(self, path_or_array) -> np.ndarray:
		if type(path_or_array) == str or type(path_or_array) == PosixPath or type(path_or_array) == WindowsPath:
			slide_arr = self.file_loader(path_or_array)
			self.logger.debug(f"Loaded array from {path_or_array}")
		elif type(path_or_array) == np.ndarray:
			slide_arr = path_or_array
		else:
			raise ValueError("Invalid input type.")
		
		return slide_arr

	def normalise(self, path_or_array) -> np.ndarray:
		"""
			Function to normalise a whole slide image using the specified normalisation type.

			Args:
				path_or_array <str, Path, np.ndarray>: Path to the whole slide image to normalise, or the array to normalise.

			Returns:
				np.ndarray: Normalised array.
		"""
		if self.target_dict is None:
			raise ValueError("Target normalisation metadata not set. Please first fit to a target.")
		
		slide_arr = self._load_path_or_array(path_or_array)

		return self._normalise(slide_arr)

	def _normalise(self, slide_arr:np.ndarray) -> np.ndarray:
		"""
			To be implemented by child classes.
		"""
		raise NotImplementedError

	def _sample_wsi(self, slide_arr:str, samples=2000000) -> list:

		height, width, _ = slide_arr.shape

		axis_samples = int(np.sqrt(samples))

		y_samples = np.linspace(0, height, axis_samples, dtype=int, endpoint=False)
		x_samples = np.linspace(0, width, axis_samples, dtype=int, endpoint=False)

		y,x = np.meshgrid(y_samples, x_samples)

		sampled_pixels = slide_arr[y,x,:]
		sampled_pixels = sampled_pixels.reshape(-1, 3)
		return np.expand_dims(sampled_pixels, axis=0)