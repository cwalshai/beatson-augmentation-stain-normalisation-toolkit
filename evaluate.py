import shutil
import random

from pathlib import Path
from beast.reinhard import ReinhardWSINormaliser, ReinhardTileNormaliser
from beast.macenko import MacenkoWSINormaliser, MacenkoTileNormaliser, MacenkoStainAugmenter
from beast.vahadane import VahadaneWSINormaliser, VahadaneTileNormaliser, VahadaneStainAugmenter
from beast.utils import IntensitySamplingMethod, AggregationMethod
from beast.tools import StainMetadataEstimator, ParallelTileNormaliser
from evaluate_utils import evaluate_nct_js, evaluate_histogram_correlation, evaluate_average_distance_from_mean, evaluate_wasserstein, evaluate_mse

#Evaluate the performance of the normalisation methods

aggregation_methods = [
    (AggregationMethod.MEDIAN, 0),
    (AggregationMethod.MEAN, 0),
]

intensity_sampling_methods = [
    (IntensitySamplingMethod.MEDIAN, 0),
    (IntensitySamplingMethod.MEAN, 0),
    # (IntensitySamplingMethod.PERCENTILE, 50),
    (IntensitySamplingMethod.PERCENTILE, 75),
    (IntensitySamplingMethod.PERCENTILE, 90),
    # (IntensitySamplingMethod.PERCENTILE, 95),
    # (IntensitySamplingMethod.PERCENTILE, 98),
    (IntensitySamplingMethod.PERCENTILE, 99),
    # (IntensitySamplingMethod.MAX, 0),
]

classes = [MacenkoTileNormaliser] #ReinhardTileNormaliser

reference_dataset_dir = Path("/mnt/mass-storage/cwalsh/datasets/raw/nct-crc-he/train")
source_dataset = Path("/mnt/mass-storage/cwalsh/datasets/raw/nct-crc-he-no-norm/train")

experiments_folder = Path("/mnt/mass-storage/cwalsh/experiments")
experiments_folder.mkdir(exist_ok=True, parents=True)

class_list = ["adi", "back", "deb", "lym", "muc", "mus", "norm", "str", "tum"]
sample=1000

# nct_dataset_eval_dir = experiments_folder 
# shutil.rmtree(nct_dataset_eval_dir, ignore_errors=True)
# nct_dataset_eval_dir.mkdir(exist_ok=True, parents=True)

# Jenson-Shannon compares probability distributions between datasets.
# Average distance from mean compares the mean distance of an individual image channel mean and std from the source dataset to the mean values of the normalised dataset.
# Histogram correlation compares the average distance of an invididual image channel histogram to the mean of the dataset. (Should give an idea of how well normalised a dataset.)

#Get baseline numbers for a good wasserstein, jenson shannon, and average distance from mean and std by identity.
# evaluate_mse(reference_dataset_dir, reference_dataset_dir, nct_dataset_eval_dir, experiment_name="NORM-IDENTITY", sample=sample, twin_mean=False)
# evaluate_mse(reference_dataset_dir, reference_dataset_dir, nct_dataset_eval_dir, experiment_name="NORM-IDENTITY-TM", sample=sample, twin_mean=True)
# evaluate_average_distance_from_mean(reference_dataset_dir, reference_dataset_dir, nct_dataset_eval_dir, experiment_name="NORM-IDENTITY", sample=sample)
# evaluate_wasserstein(reference_dataset_dir, reference_dataset_dir, nct_dataset_eval_dir, experiment_name="NORM-IDENTITY", sample=sample)
# evaluate_nct_js(reference_dataset_dir, reference_dataset_dir, nct_dataset_eval_dir, experiment_name="NORM-IDENTITY", sample=sample)

# #Evaluate the distance individual image channel histograms from the raw and normalised datasets are from mean histograms for various metrics.
# evaluate_histogram_correlation(source_dataset, nct_dataset_eval_dir, experiment_name="RAW", sample=sample)
# evaluate_histogram_correlation(reference_dataset_dir, nct_dataset_eval_dir, experiment_name="NORM", sample=sample)

# #Evaluate difference in mean and std between raw and normalised datasets, and the jenson shannon divergence between the two datasets.
# evaluate_wasserstein(source_dataset, reference_dataset_dir, nct_dataset_eval_dir, experiment_name="RAW->NORM", sample=sample)
# evaluate_nct_js(source_dataset, reference_dataset_dir, nct_dataset_eval_dir, experiment_name="RAW->NORM", sample=sample)
# evaluate_average_distance_from_mean(source_dataset, reference_dataset_dir, nct_dataset_eval_dir, experiment_name="RAW->NORM", sample=sample)
# evaluate_mse(source_dataset, reference_dataset_dir, nct_dataset_eval_dir, experiment_name="RAW->NORM", sample=sample, twin_mean=False)
# evaluate_mse(source_dataset, reference_dataset_dir, nct_dataset_eval_dir, experiment_name="RAW->NORM", sample=sample, twin_mean=True)

#ALTER METRICS TO COMPARE AVERAGE OF DISTANCES BETWEEN SOURCE HISTOGRAMS AND MEAN OF TARGET, AND A SINGLE DISTANCE BETWEEN THE MEAN HISTOGRAMS OF THE SOURCE AND TARGET DATASETS.

csv_file = experiments_folder / "results.csv"

csv_header = ["experiment_str", "mean_mse_from_imean", "mean_mse_from_ref_mean", "average_distance_from_ref_mean", "average_distance_from_ref_std", "wasserstine_ref"]

import csv
with open(csv_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)

    mean_mse_from_imean = evaluate_mse(reference_dataset_dir, reference_dataset_dir, experiments_folder, experiment_name="Reference to Reference (Identity)", sample=sample, twin_mean=False)
    mean_mse_from_ref_mean = evaluate_mse(source_dataset, reference_dataset_dir, experiments_folder, experiment_name="Raw to Reference", sample=sample, twin_mean=False)
    average_distance_from_ref_mean, average_distance_from_ref_std = evaluate_average_distance_from_mean(source_dataset, reference_dataset_dir, experiments_folder, experiment_name="Raw to Reference", sample=sample)
    wasserstine_ref = evaluate_wasserstein(source_dataset, reference_dataset_dir, experiments_folder, experiment_name="Raw to Reference", sample=sample)

    # Write results to csv file
    row = ["raw-norm", mean_mse_from_imean, mean_mse_from_ref_mean, average_distance_from_ref_mean, average_distance_from_ref_std, wasserstine_ref]
    writer.writerow(row)
    f.flush()

    count=0
    for class_ in classes:
        for aggregation_method in aggregation_methods:
            for intensity_sampling_method in intensity_sampling_methods:

                aggregation_str = aggregation_method[0].name
                if aggregation_method[0] == AggregationMethod.PERCENTILE:
                    aggregation_str += f"_{aggregation_method[1]}"
                
                count+=1

                sampling_str = intensity_sampling_method[0].name
                if intensity_sampling_method[0] == IntensitySamplingMethod.PERCENTILE:
                    sampling_str += f"_{intensity_sampling_method[1]}"

                experiment_str = f"{str(class_.__name__)}_A-{aggregation_str}_I-{sampling_str}"
                experiment_folder = experiments_folder / experiment_str
                experiment_folder.mkdir(exist_ok=True, parents=True)

                # ESTIMATE METADATA ==================================================

                if class_ == ReinhardTileNormaliser or class_ == ReinhardWSINormaliser:
                    estimator = StainMetadataEstimator(class_, aggregation_method[0], aggregation_percentile=aggregation_method[1], terminal_log_level="WARNING")
                else:
                    estimator = StainMetadataEstimator(class_, aggregation_method[0], aggregation_percentile=aggregation_method[1], intensity_sampling_method=intensity_sampling_method[0], intensity_percentile=intensity_sampling_method[1], terminal_log_level="WARNING")

                tile_list = []
                for class_name in class_list:
                    class_dir = reference_dataset_dir / class_name
                    class_tile_list = list(class_dir.glob("*.tif"))

                    if sample:
                        tile_list.extend(random.sample(class_tile_list, sample))
                    else:
                        tile_list.extend(class_tile_list)
                
                estimated_metadata = estimator.estimate(tile_list)

                print("----------------------------------")
                print(experiment_str)
                print(estimated_metadata)
                estimator.save_fit(experiment_folder / "estimated_fit.json")

                # Normalise ==========================================================
                if class_ == ReinhardTileNormaliser or class_ == ReinhardWSINormaliser:
                    normaliser = ParallelTileNormaliser(class_, num_threads=32, terminal_log_level="WARNING")
                else:
                    normaliser = ParallelTileNormaliser(class_, num_threads=32, intensity_sampling_method=intensity_sampling_method[0], intensity_percentile=intensity_sampling_method[1], terminal_log_level="WARNING")
                normaliser.load_fit(experiment_folder / "estimated_fit.json")

                normalised_dir = experiment_folder / "dataset_norm"

                for class_name in class_list:
                    class_dir = source_dataset / class_name
                    class_tile_list = list(class_dir.glob("*.tif"))
                    if sample:
                        class_tile_list = random.sample(class_tile_list, sample)
                    normaliser.normalise(class_tile_list, normalised_dir / class_name)

                # Evaluate ===========================================================
                # mean_correlation_with_imean, mean_intersecton_with_imean = evaluate_histogram_correlation(normalised_dir, experiment_folder, experiment_name=experiment_str, sample=sample)
                mean_mse_from_imean = evaluate_mse(normalised_dir, normalised_dir, experiment_folder, experiment_name="Normalised (BEAST) to Normalised (BEAST) (Identity)", sample=sample, twin_mean=False)
                mean_mse_from_ref_mean = evaluate_mse(normalised_dir, reference_dataset_dir, experiment_folder, experiment_name="Normalised (BEAST) to Reference", sample=sample, twin_mean=False)
                average_distance_from_ref_mean, average_distance_from_ref_std = evaluate_average_distance_from_mean(normalised_dir, reference_dataset_dir, experiment_folder, experiment_name="Normalised (BEAST) to Reference", sample=sample)
                wasserstine_ref = evaluate_wasserstein(normalised_dir, reference_dataset_dir, experiment_folder, experiment_name="Normalised (BEAST) to Reference", sample=sample)

                # Write results to csv file
                row = [experiment_str, mean_mse_from_imean, mean_mse_from_ref_mean, average_distance_from_ref_mean, average_distance_from_ref_std, wasserstine_ref]
                writer.writerow(row)
                f.flush()