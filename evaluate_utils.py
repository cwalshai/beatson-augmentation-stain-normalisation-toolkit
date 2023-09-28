import numpy as np
import cv2
import csv
import random
import pandas as pd
from beast.file_readers import read_tile
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm

import seaborn as sns
sns.set_theme(style="whitegrid")

def read_images(path_list:list) -> list:
		images = []
		for path in path_list:
				images.append(read_tile(path))
		return images

def write_csv(path:Path, data:list):
		with open(path, "w", newline="") as f:
				writer = csv.writer(f)
				writer.writerows(data)

def compute_histograms_cv2(image_list:list, lab=False) -> list:
		ch1 = []
		ch2 = []
		ch3 = []
		
		for image in image_list:
				if lab:
						image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
				
				ch1_hist = cv2.calcHist([image], [0], None, [256], [0, 256], accumulate=False)
				ch2_hist = cv2.calcHist([image], [1], None, [256], [0, 256], accumulate=False)
				ch3_hist = cv2.calcHist([image], [2], None, [256], [0, 256], accumulate=False)

				ch1.append(ch1_hist)
				ch2.append(ch2_hist)
				ch3.append(ch3_hist)

		return ch1, ch2, ch3

def compute_js_for_class(source_dataset_dir:Path, reference_dataset_dir:Path, class_folder:str, image_extension:str=".tif", sample=None, lab=True):
	source_class_dir = source_dataset_dir / class_folder
	reference_class_dir = reference_dataset_dir / class_folder

	source_image_list = list(source_class_dir.glob(f"*{image_extension}"))
	if sample is not None:
			source_image_list = random.sample(source_image_list, sample)
	source_images = read_images(source_image_list)

	reference_image_list = list(reference_class_dir.glob(f"*{image_extension}"))
	if sample is not None:
			reference_image_list = random.sample(reference_image_list, sample)
	reference_images = read_images(reference_image_list)

	src_ch1, src_ch2, src_ch3 = compute_histograms_cv2(source_images, lab=lab)
	ref_ch1, ref_ch2, ref_ch3 = compute_histograms_cv2(reference_images, lab=lab)

	mean_src1 = np.mean(src_ch1, axis=0)
	mean_src2 = np.mean(src_ch2, axis=0)
	mean_src3 = np.mean(src_ch3, axis=0)

	mean_ref1 = np.mean(ref_ch1, axis=0)
	mean_ref2 = np.mean(ref_ch2, axis=0)
	mean_ref3 = np.mean(ref_ch3, axis=0)

	forward_kl1 = cv2.compareHist(mean_src1, mean_ref1, cv2.HISTCMP_KL_DIV)
	backward_kl1 = cv2.compareHist(mean_ref1, mean_src1, cv2.HISTCMP_KL_DIV)
	jenson_shannon_divergence1 = (forward_kl1 + backward_kl1) / 2

	forward_kl2 = cv2.compareHist(mean_src2, mean_ref2, cv2.HISTCMP_KL_DIV)
	backward_kl2 = cv2.compareHist(mean_ref2, mean_src2, cv2.HISTCMP_KL_DIV)
	jenson_shannon_divergence2 = (forward_kl2 + backward_kl2) / 2

	forward_kl3 = cv2.compareHist(mean_src3, mean_ref3, cv2.HISTCMP_KL_DIV)
	backward_kl3 = cv2.compareHist(mean_ref3, mean_src3, cv2.HISTCMP_KL_DIV)
	jenson_shannon_divergence3 = (forward_kl3 + backward_kl3) / 2

	return jenson_shannon_divergence1, jenson_shannon_divergence2, jenson_shannon_divergence3

def compute_wasserstein(source_dataset_dir:Path, reference_dataset_dir:Path, class_folder:str, image_extension:str=".tif", sample=None, lab=True):
	source_class_dir = source_dataset_dir / class_folder
	reference_class_dir = reference_dataset_dir / class_folder

	source_image_list = list(source_class_dir.glob(f"*{image_extension}"))
	if sample is not None:
			source_image_list = random.sample(source_image_list, sample)
	source_images = read_images(source_image_list)

	reference_image_list = list(reference_class_dir.glob(f"*{image_extension}"))
	if sample is not None:
			reference_image_list = random.sample(reference_image_list, sample)
	reference_images = read_images(reference_image_list)

	src_ch1, src_ch2, src_ch3 = compute_histograms_cv2(source_images, lab=lab)
	ref_ch1, ref_ch2, ref_ch3 = compute_histograms_cv2(reference_images, lab=lab)

	mean_src1 = np.mean(src_ch1, axis=0)
	mean_src2 = np.mean(src_ch2, axis=0)
	mean_src3 = np.mean(src_ch3, axis=0)

	mean_ref1 = np.mean(ref_ch1, axis=0)
	mean_ref2 = np.mean(ref_ch2, axis=0)
	mean_ref3 = np.mean(ref_ch3, axis=0)

	coordinates = np.arange(256).reshape(256, 1)

	mean_src1_sig = np.hstack((mean_src1, coordinates)).astype(np.float32)
	mean_src2_sig = np.hstack((mean_src2, coordinates)).astype(np.float32)
	mean_src3_sig = np.hstack((mean_src3, coordinates)).astype(np.float32)

	mean_ref1_sig = np.hstack((mean_ref1, coordinates)).astype(np.float32)
	mean_ref2_sig = np.hstack((mean_ref2, coordinates)).astype(np.float32)
	mean_ref3_sig = np.hstack((mean_ref3, coordinates)).astype(np.float32)

	channel_1, _, _ = cv2.EMD(mean_src1_sig, mean_ref1_sig, cv2.DIST_L2)
	channel_2, _, _ = cv2.EMD(mean_src2_sig, mean_ref2_sig, cv2.DIST_L2)
	channel_3, _, _ = cv2.EMD(mean_src3_sig, mean_ref3_sig, cv2.DIST_L2)

	return channel_1, channel_2, channel_3

def compare_histograms(source_dataset_dir:Path, class_folder:str, image_extension:str=".tif", sample=None, method=cv2.HISTCMP_CORREL, lab=True):
	source_class_dir = source_dataset_dir / class_folder

	source_image_list = list(source_class_dir.glob(f"*{image_extension}"))
	if sample is not None:
			source_image_list = random.sample(source_image_list, sample)
	source_images = read_images(source_image_list)

	src_ch1, src_ch2, src_ch3 = compute_histograms_cv2(source_images, lab=lab)

	src_mean1 = np.mean(src_ch1, axis=0)
	src_mean2 = np.mean(src_ch2, axis=0)
	src_mean3 = np.mean(src_ch3, axis=0)

	ch1_dist_list = []
	ch2_dist_list = []
	ch3_dist_list = []

	for hist in src_ch1:
		distance = cv2.compareHist(hist, src_mean1, method)
		ch1_dist_list.append(distance)

	for hist in src_ch2:
		distance = cv2.compareHist(hist, src_mean2, method)
		ch2_dist_list.append(distance)

	for hist in src_ch3:
		distance = cv2.compareHist(hist, src_mean3, method)
		ch3_dist_list.append(distance)
	
	return np.mean(ch1_dist_list, axis=0), np.mean(ch2_dist_list, axis=0), np.mean(ch3_dist_list, axis=0)

def get_average_distance_mean_std(source_dataset_dir:Path, reference_dataset_dir:Path, class_folder:str, image_extension:str=".tif", sample=None, lab=True):
	source_class_dir = source_dataset_dir / class_folder
	reference_class_dir = reference_dataset_dir / class_folder

	source_image_list = list(source_class_dir.glob(f"*{image_extension}"))
	if sample is not None:
			source_image_list = random.sample(source_image_list, sample)
	source_images = read_images(source_image_list)

	reference_image_list = list(reference_class_dir.glob(f"*{image_extension}"))
	if sample is not None:
			reference_image_list = random.sample(reference_image_list, sample)
	reference_images = read_images(reference_image_list)

	def get_channel_means(image_list):
		ch1_means = []
		ch2_means = []
		ch3_means = []
		for image in image_list:
			ch1_means.append(np.mean(image[:,:,0]))
			ch2_means.append(np.mean(image[:,:,1]))
			ch3_means.append(np.mean(image[:,:,2]))
		return ch1_means, ch2_means, ch3_means

	def get_channel_stds(image_list):
		ch1_stds = []
		ch2_stds = []
		ch3_stds = []
		for image in image_list:
			ch1_stds.append(np.std(image[:,:,0]))
			ch2_stds.append(np.std(image[:,:,1]))
			ch3_stds.append(np.std(image[:,:,2]))
		return ch1_stds, ch2_stds, ch3_stds

	src_ch1_means, src_ch2_means, src_ch3_means = get_channel_means(source_images)
	ref_ch1_means, ref_ch2_means, ref_ch3_means = get_channel_means(reference_images)

	src_ch1_stds, src_ch2_stds, src_ch3_stds = get_channel_stds(source_images)
	ref_ch1_stds, ref_ch2_stds, ref_ch3_stds = get_channel_stds(reference_images)

	mean_ref1_mean = np.mean(ref_ch1_means, axis=0)
	mean_ref2_mean = np.mean(ref_ch2_means, axis=0)
	mean_ref3_mean = np.mean(ref_ch3_means, axis=0)

	mean_ref1_std = np.mean(ref_ch1_stds, axis=0)
	mean_ref2_std = np.mean(ref_ch2_stds, axis=0)
	mean_ref3_std = np.mean(ref_ch3_stds, axis=0)

	ch1_mean_distances = []
	ch2_mean_distances = []
	ch3_mean_distances = []

	ch1_std_distances = []
	ch2_std_distances = []
	ch3_std_distances = []

	for channel_mean in src_ch1_means:
		ch1_mean_distances.append(np.absolute(channel_mean - mean_ref1_mean))

	for channel_mean in src_ch2_means:
		ch2_mean_distances.append(np.absolute(channel_mean - mean_ref2_mean))

	for channel_mean in src_ch3_means:
		ch3_mean_distances.append(np.absolute(channel_mean - mean_ref3_mean))

	for channel_std in src_ch1_stds:
		ch1_std_distances.append(np.absolute(channel_std - mean_ref1_std))
	
	for channel_std in src_ch2_stds:
		ch2_std_distances.append(np.absolute(channel_std - mean_ref2_std))

	for channel_std in src_ch3_stds:
		ch3_std_distances.append(np.absolute(channel_std - mean_ref3_std))

	ch1_mean_distance = np.mean(ch1_mean_distances, axis=0)
	ch2_mean_distance = np.mean(ch2_mean_distances, axis=0)
	ch3_mean_distance = np.mean(ch3_mean_distances, axis=0)

	ch1_std_distance = np.mean(ch1_std_distances, axis=0)
	ch2_std_distance = np.mean(ch2_std_distances, axis=0)
	ch3_std_distance = np.mean(ch3_std_distances, axis=0)

	return ch1_mean_distance, ch2_mean_distance, ch3_mean_distance, ch1_std_distance, ch2_std_distance, ch3_std_distance
	
def get_mse(source_dataset_dir:Path, reference_dataset_dir:Path, class_folder:str, image_extension:str=".tif", sample=None, lab=True, twin_mean=True):
	source_class_dir = source_dataset_dir / class_folder
	reference_class_dir = reference_dataset_dir / class_folder

	source_image_list = list(source_class_dir.glob(f"*{image_extension}"))
	if sample is not None:
			source_image_list = random.sample(source_image_list, sample)
	source_images = read_images(source_image_list)

	reference_image_list = list(reference_class_dir.glob(f"*{image_extension}"))
	if sample is not None:
			reference_image_list = random.sample(reference_image_list, sample)
	reference_images = read_images(reference_image_list)

	src_ch1, src_ch2, src_ch3 = compute_histograms_cv2(source_images, lab=lab)
	ref_ch1, ref_ch2, ref_ch3 = compute_histograms_cv2(reference_images, lab=lab)

	ref_ch1_mean = np.mean(ref_ch1, axis=0)
	ref_ch2_mean = np.mean(ref_ch2, axis=0)
	ref_ch3_mean = np.mean(ref_ch3, axis=0)

	def mean_squared_error(array1:np.ndarray, array2:np.ndarray):
		"""Calculates the mean squared error between two arrays"""
		return np.mean((array1 - array2) ** 2)

	if twin_mean:
		src_ch1_mean = np.mean(src_ch1, axis=0)
		src_ch2_mean = np.mean(src_ch2, axis=0)
		src_ch3_mean = np.mean(src_ch3, axis=0)

		return mean_squared_error(src_ch1_mean, ref_ch1_mean), mean_squared_error(src_ch2_mean, ref_ch2_mean), mean_squared_error(src_ch3_mean, ref_ch3_mean)

	else:
		mse_ch1 = []
		mse_ch2 = []
		mse_ch3 = []

		for channel in src_ch1:
			mse_ch1.append(mean_squared_error(channel, ref_ch1_mean))

		for channel in src_ch2:
			mse_ch2.append(mean_squared_error(channel, ref_ch2_mean))

		for channel in src_ch3:
			mse_ch3.append(mean_squared_error(channel, ref_ch3_mean))

		return np.mean(mse_ch1), np.mean(mse_ch2), np.mean(mse_ch3)

def plot_metrics(csv_file, output_folder, classes_to_exclude=None, comparison="Correlation", experiment_name=""):
	# Exclude classes -------------------------------------------------------
	df = pd.read_csv(csv_file)

	if classes_to_exclude is not None:
		df = df[~df["Class"].isin(classes_to_exclude)]

	# Compute total Mean and Sum across all classes and channels
	ch1_mean = df["Ch1"].mean(axis=0)
	ch2_mean = df["Ch2"].mean(axis=0)
	ch3_mean = df["Ch3"].mean(axis=0)

	ch1_total = df["Ch1"].sum(axis=0)
	ch2_total = df["Ch2"].sum(axis=0)
	ch3_total = df["Ch3"].sum(axis=0)

	total_mean = np.mean([ch1_mean, ch2_mean, ch3_mean])
	total = np.sum([ch1_total, ch2_total, ch3_total])

	# Plot ------------------------------------------------------------------

	fig, ax = plt.subplots(figsize=(20, 10))

	# Draw a nested barplot by species and sex
	g = sns.barplot(
			data=df.melt(id_vars = ["Class"], value_vars=["Ch1", "Ch2", "Ch3"], var_name="Channel", value_name=f"{comparison}"),
			y=f"{comparison}", x="Channel", hue="Class",
	)

	for p in range(len(g.containers)):
		plt.bar_label(g.containers[p], fmt="%.2f"),

	ax.annotate(f"Mean: {total_mean:.2f}", xy=(1.06, 0.55), xytext=(1.06, 0.55), xycoords="axes fraction", textcoords="axes fraction", ha="center", va="center", fontsize=12, color="black")
	ax.annotate(f"CH1 Mean: {ch1_mean:.2f}", xy=(1.06, 0.50), xytext=(1.06, 0.50), xycoords="axes fraction", textcoords="axes fraction", ha="center", va="center", fontsize=10, color="grey")
	ax.annotate(f"CH2 Mean: {ch2_mean:.2f}", xy=(1.06, 0.45), xytext=(1.06, 0.45), xycoords="axes fraction", textcoords="axes fraction", ha="center", va="center", fontsize=10, color="grey")
	ax.annotate(f"CH3 Mean: {ch3_mean:.2f}", xy=(1.06, 0.40), xytext=(1.06, 0.40), xycoords="axes fraction", textcoords="axes fraction", ha="center", va="center", fontsize=10, color="grey")
	ax.annotate(f"Total: {total:.2f}", xy=(1.06, 0.35), xytext=(1.06, 0.35), xycoords="axes fraction", textcoords="axes fraction", ha="center", va="center", fontsize=10, color="grey")

	plt.title(f"{comparison} : {experiment_name}")

	# ax.legend(loc='upper right', bbox_to_anchor=(1.10, 0.75), ncol=1, fancybox=True, shadow=True)

	plt.savefig(output_folder/f"{comparison.lower().replace(' ','_')}_{experiment_name.lower()}.png", dpi=300)
	plt.close()
	return total_mean

def evaluate_nct_js(source_dataset:Path, reference_dataset_dir:Path, output_folder:Path, classes_to_exclude:list=None, sample=None, experiment_name="", lab=True):
	"""
		Evaluate the divergence in probability distribution between the reference dataset and the normalised dataset.

		Smaller values indicate a better match between the two distributions.
	"""
	
	class_list = ["adi", "back", "deb", "lym", "muc", "mus", "norm", "str", "tum"]

	csv_list = [["Class", "Ch1", "Ch2", "Ch3"]]

	for class_name in class_list:
		js_divergence = compute_js_for_class(Path(source_dataset), Path(reference_dataset_dir), class_name, sample=sample, lab=lab)
		print(f"JS divergence for {class_name}: {js_divergence}")
		csv_list.append([class_name, js_divergence[0], js_divergence[1], js_divergence[2]])

	csv_pth = output_folder / f"js_divergence_{experiment_name.lower()}.csv"
	write_csv(csv_pth, csv_list)
	return plot_metrics(csv_pth, output_folder, classes_to_exclude=classes_to_exclude, comparison="Jenson Shannon Divergence", experiment_name=experiment_name)

def evaluate_wasserstein(source_dataset:Path, reference_dataset_dir:Path, output_folder:Path, classes_to_exclude:list=None, sample=None, experiment_name="", lab=True):
	"""
		Evaluate the divergence in probability distribution between the reference dataset and the normalised dataset using Wasserstein distance.

		Smaller values indicate a better match between the two distributions.
	"""
	
	class_list = ["adi", "back", "deb", "lym", "muc", "mus", "norm", "str", "tum"]

	csv_list = [["Class", "Ch1", "Ch2", "Ch3"]]

	for class_name in class_list:
		ws_divergence = compute_wasserstein(Path(source_dataset), Path(reference_dataset_dir), class_name, sample=sample, lab=lab)
		print(f"Wasserstein distance for {class_name}: {ws_divergence}")
		csv_list.append([class_name, ws_divergence[0], ws_divergence[1], ws_divergence[2]])

	csv_pth = output_folder / f"wasserstein_distance_{experiment_name.lower()}.csv"
	write_csv(csv_pth, csv_list)
	return plot_metrics(csv_pth, output_folder, classes_to_exclude=classes_to_exclude, comparison="Wasserstein Distance of Histograms", experiment_name=experiment_name)

def evaluate_histogram_correlation(source_dataset:Path, output_folder:Path, classes_to_exclude:list=None, sample=None, experiment_name="", lab=True):
	"""
		Evaluate the average distance of the histograms to the mean of the dataset.

		Should evaluate how well normalised the dataset is.

		Look at individual metrics for details.
	"""
	
	class_list = ["adi", "back", "deb", "lym", "muc", "mus", "norm", "str", "tum"]

	# Correlation -----------------------------------------------------------

	"""In the case of the cv2.HISTCMP_CORREL method, the cv2.compareHist function returns a scalar value between -1 and 1, where -1 represents completely different histograms, 0 represents completely uncorrelated histograms, and 1 represents completely identical histograms. The Correlation method calculates the Pearson product-moment correlation coefficient between the two histograms, which is a measure of linear association between two variables. The greater the Correlation value, the greater the similarity between the histograms."""

	csv_list = [["Class", "Ch1", "Ch2", "Ch3"]]
	for class_name in class_list:
		ch1cor, ch2cor, ch3cor = compare_histograms(Path(source_dataset), class_name, sample=sample, method=cv2.HISTCMP_CORREL, lab=lab)
		
		print(f"Correlation for {class_name}: {ch1cor:.2f}, {ch2cor:.2f}, {ch3cor:.2f}")
		csv_list.append([class_name, ch1cor, ch2cor, ch3cor])

	comparison = "Correlation"
	csv_path = output_folder / f"{comparison.lower().replace(' ','_')}_{experiment_name.lower().replace(' ','_')}.csv"
	write_csv(csv_path, csv_list)
	correlation_mean = plot_metrics(csv_path, output_folder, classes_to_exclude=classes_to_exclude, comparison=comparison, experiment_name=experiment_name)

	# Chi Squared -----------------------------------------------------------

	# """In the case of the cv2.HISTCMP_CHISQR method, the cv2.compareHist function returns a scalar value that represents the Chi-Squared distance between two histograms. A value of 0 means that the histograms are completely identical, while a value greater than 0 indicates that the histograms are different, with the value being proportional to the difference between the histograms. The greater the Chi-Squared distance, the greater the difference between the histograms. The Chi-Squared method is commonly used for comparing histograms when the distributions represented by the histograms are Gaussian."""

	# csv_list = [["Class", "Ch1", "Ch2", "Ch3"]]
	# for class_name in class_list:
	# 	ch1chi, ch2chi, ch3chi = compare_histograms(Path(source_dataset), class_name, sample=sample, method=cv2.HISTCMP_CHISQR, lab=lab)
		
	# 	print(f"Chi Squared for {class_name}: {ch1chi:.2f}, {ch2chi:.2f}, {ch3chi:.2f}")
	# 	csv_list.append([class_name, ch1chi, ch2chi, ch3chi])
	
	# comparison = "Chi Squared"
	# csv_path = output_folder / f"{comparison.lower().replace(' ','_')}_{experiment_name.lower().replace(' ','_')}.csv"
	# write_csv(csv_path, csv_list)
	# chi_squared_mean = plot_metrics(csv_path, output_folder, classes_to_exclude=classes_to_exclude, comparison=comparison, experiment_name=experiment_name)

	# Intersection ----------------------------------------------------------

	"""In the case of the cv2.HISTCMP_INTERSECT method, the cv2.compareHist function returns a scalar value that represents the area under the curve of the two histograms, calculated as the sum of the minimum of each corresponding bin value in the two histograms. A value of 0 means that the histograms are completely different, while a value of 1 means that the histograms are completely identical. The greater the Intersection value, the greater the similarity between the histograms."""

	csv_list = [["Class", "Ch1", "Ch2", "Ch3"]]
	for class_name in class_list:
		ch1int, ch2int, ch3int = compare_histograms(Path(source_dataset), class_name, sample=sample, method=cv2.HISTCMP_INTERSECT, lab=lab)
		
		print(f"Intersection for {class_name}: {ch1int:.2f}, {ch2int:.2f}, {ch3int:.2f}")
		csv_list.append([class_name, ch1int, ch2int, ch3int])

	comparison = "Intersection"
	csv_path = output_folder / f"{comparison.lower().replace(' ','_')}_{experiment_name.lower().replace(' ','_')}.csv"
	write_csv(csv_path, csv_list)
	intersection_mean = plot_metrics(csv_path, output_folder, classes_to_exclude=classes_to_exclude, comparison=comparison, experiment_name=experiment_name)

	# Bhattacharyya ----------------------------------------------------------

	# """In the case of the cv2.HISTCMP_BHATTACHARYYA method, the cv2.compareHist function returns a scalar value that represents the Bhattacharyya distance between two histograms. A value of 0 means that the histograms are completely identical, while a value greater than 0 indicates that the histograms are different, with the value being proportional to the difference between the histograms. The greater the Bhattacharyya distance, the greater the difference between the histograms."""

	# csv_list = [["Class", "Ch1", "Ch2", "Ch3"]]
	# for class_name in class_list:
	# 	ch1bat, ch2bat, ch3bat = compare_histograms(Path(source_dataset), class_name, sample=sample, method=cv2.HISTCMP_BHATTACHARYYA, lab=lab)
		
	# 	print(f"Bhattacharyya Distance for {class_name}: {ch1bat:.2f}, {ch2bat:.2f}, {ch3bat:.2f}")
	# 	csv_list.append([class_name, ch1bat, ch2bat, ch3bat])

	# comparison = "Bhattacharyya Distance"
	# csv_path = output_folder / f"{comparison.lower().replace(' ','_')}_{experiment_name.lower().replace(' ','_')}.csv"
	# write_csv(csv_path, csv_list)
	# batty_mean = plot_metrics(csv_path, output_folder, classes_to_exclude=classes_to_exclude, comparison=comparison, experiment_name=experiment_name)

	return correlation_mean, intersection_mean

def evaluate_average_distance_from_mean(source_dataset:Path, reference_dataset_dir:Path, output_folder:Path, classes_to_exclude:list=None, sample=None, experiment_name="", lab=True):
	"""
		Evaluate the average distance of the mean and standard deviation of the normalised dataset to the reference dataset.
	"""
	
	class_list = ["adi", "back", "deb", "lym", "muc", "mus", "norm", "str", "tum"]

	mean_csv_list = [["Class", "Ch1", "Ch2", "Ch3"]]
	std_csv_list = [["Class", "Ch1", "Ch2", "Ch3"]]
	for class_name in class_list:
		ch1mean, ch2mean, ch3mean, ch1std, ch2std, ch3std = get_average_distance_mean_std(Path(source_dataset), Path(reference_dataset_dir), class_name, sample=sample, lab=lab)
		print(f"Average Distance from the reference mean {class_name}: {ch1mean:.2f}, {ch2mean:.2f}, {ch3mean:.2f}")
		print(f"Average Distance from the reference std {class_name}: {ch1std:.2f}, {ch2std:.2f}, {ch3std:.2f}")

		mean_csv_list.append([class_name, ch1mean, ch2mean, ch3mean])
		std_csv_list.append([class_name, ch1std, ch2std, ch3std])

	mean_comparison = "Average MAE of Pixel Means to Reference Pixel Means"
	std_comparison = "Average MAE of Pixel STDs to Reference Pixel STDs"

	mean_csv_pth = output_folder / f"{mean_comparison.lower().replace(' ','_')}_{experiment_name.lower().replace(' ','_')}.csv"
	std_csv_pth = output_folder / f"{std_comparison.lower().replace(' ','_')}_{experiment_name.lower().replace(' ','_')}.csv"

	write_csv(mean_csv_pth, mean_csv_list)
	write_csv(std_csv_pth, std_csv_list)

	mean_mean = plot_metrics(mean_csv_pth, output_folder, classes_to_exclude=classes_to_exclude, comparison=mean_comparison, experiment_name=experiment_name)
	std_mean = plot_metrics(std_csv_pth, output_folder, classes_to_exclude=classes_to_exclude, comparison=std_comparison, experiment_name=experiment_name)

	return mean_mean, std_mean

def evaluate_mse(source_dataset:Path, reference_dataset_dir:Path, output_folder:Path, classes_to_exclude:list=None, sample=None, experiment_name="", lab=True, twin_mean=False):
	"""
		Evaluate the average distance of the mean and standard deviation of the normalised dataset to the reference dataset.
	"""
	class_list = ["adi", "back", "deb", "lym", "muc", "mus", "norm", "str", "tum"]

	if twin_mean:
		mean_comparison = "Average MSE of Histograms"
	else:
		mean_comparison = "Average MSE of Histograms to Reference Histograms"

	mean_csv_list = [["Class", "Ch1", "Ch2", "Ch3"]]
	for class_name in class_list:
		ch1mean, ch2mean, ch3mean = get_mse(Path(source_dataset), Path(reference_dataset_dir), class_name, sample=sample, lab=lab, twin_mean=twin_mean)
		print(f"{mean_comparison} {class_name}: {ch1mean:.2f}, {ch2mean:.2f}, {ch3mean:.2f}")

		mean_csv_list.append([class_name, ch1mean, ch2mean, ch3mean])

	mean_csv_pth = output_folder / f"{mean_comparison.lower().replace(' ','_')}_{experiment_name.lower().replace(' ','_')}.csv"
	write_csv(mean_csv_pth, mean_csv_list)

	return plot_metrics(mean_csv_pth, output_folder, classes_to_exclude=classes_to_exclude, comparison=mean_comparison, experiment_name=experiment_name) 

if __name__ == "__main__":

	reference_dataset_dir = Path("/mnt/mass-storage/cwalsh/datasets/raw/NCT-CRC-HE/train")
	estimation_dataset_dir = Path("/home/local/BICR/cwalsh/workspace/normalisation/beast/nct100k_macenko_normed_est_median")
	evaluate_nct_js(reference_dataset_dir, estimation_dataset_dir, "test")