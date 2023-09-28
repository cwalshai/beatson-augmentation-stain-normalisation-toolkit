from beast.reinhard import ReinhardWSINormaliser, ReinhardTileNormaliser
from beast.macenko import MacenkoWSINormaliser, MacenkoTileNormaliser, MacenkoStainAugmenter
from beast.vahadane import VahadaneWSINormaliser, VahadaneTileNormaliser, VahadaneStainAugmenter

from beast.file_readers import read_wsi, read_tile
from beast.file_writers import write_ome_tiff, get_mpp, write_tile

from pathlib import Path

import cv2

from PIL import Image

# import matplotlib.pyplot as plt

import numpy as np

if __name__ == "__main__":
    
    # slide_path1 = "/mnt/mass-storage/cwalsh/datasets/raw/TCGA-COAD/slides/TCGA-QL-A97D-01A-01-TSA.02168EBB-1D66-4B79-B363-3222ED7EED77.svs"
    # slide_path2 = "/mnt/mass-storage/cwalsh/datasets/raw/TCGA-COAD/slides/TCGA-NH-A50U-01A-03-TS3.C6099622-D556-42EF-AC68-E99120058977.svs"

    # normaliser = VahadaneWSINormaliser(logfile="normalisation.log")
    # normaliser.fit(slide_path1)
    # # normaliser.save_fit("vahadane_fit.json")
    # # normaliser.load_fit("vahadane_fit.json")
    # normed_slide = normaliser.normalise(slide_path2)
    # write_ome_tiff(normed_slide, *get_mpp(slide_path2), "vahadane_normed.ome.tiff")

    # stain1 = normaliser.convert_to_stain1(slide_path2)
    # write_ome_tiff(stain1, *get_mpp(slide_path2), "vstain1.ome.tiff")

    # stain2 = normaliser.convert_to_stain2(slide_path2)
    # write_ome_tiff(stain2, *get_mpp(slide_path2), "vstain2.ome.tiff")

    # staincs = normaliser.convert_to_stain_concentrations(slide_path2)

    # print(staincs.shape)
    # print(staincs[5000,5000,:])

    output_folder = Path("output_nct100k")

    # # Single Tile
    # tile1 = "/home/local/BICR/cwalsh/workspace/normalisation/beast/test-images/0.png"
    # tile2 = "/home/local/BICR/cwalsh/workspace/normalisation/beast/test-images/1.png"
    # tile3 = "/home/local/BICR/cwalsh/workspace/normalisation/beast/test-images/2.png"
    # tile4 = "/home/local/BICR/cwalsh/workspace/normalisation/beast/test-images/3.png"

    # zoomed_tile1 = "/home/local/BICR/cwalsh/workspace/normalisation/beast/test-images/zoomed/0.png"
    # zoomed_tile2 = "/home/local/BICR/cwalsh/workspace/normalisation/beast/test-images/zoomed/1.png"
    # zoomed_tile3 = "/home/local/BICR/cwalsh/workspace/normalisation/beast/test-images/zoomed/2.png"
    # zoomed_tile4 = "/home/local/BICR/cwalsh/workspace/normalisation/beast/test-images/zoomed/3.png"

    # slide_tile_list = [tile1, tile2, tile3, tile4]
    # zoomed_tile_list = [zoomed_tile1, zoomed_tile2, zoomed_tile3, zoomed_tile4]

    # normaliser_list = [ReinhardTileNormaliser, MacenkoTileNormaliser, VahadaneTileNormaliser]

    # for normaliser_class in normaliser_list:

    #     normaliser = normaliser_class(logfile="normalisation.log")
    #     normaliser.fit(tile1)

    #     class_output_dir = output_folder / f"{normaliser_class.__name__}"
    #     class_output_dir_zoomed = class_output_dir / "zoomed"
    #     class_output_dir.mkdir(parents=True, exist_ok=True)
    #     class_output_dir_zoomed.mkdir(parents=True, exist_ok=True)

    #     for i, tile in enumerate(slide_tile_list):
    #         normed_tile = normaliser.normalise(tile)
    #         write_tile(normed_tile, class_output_dir / f"{i}.png")
    
    #     for i, tile in enumerate(zoomed_tile_list):
    #         normed_tile = normaliser.normalise(tile)
    #         write_tile(normed_tile, class_output_dir_zoomed / f"{i}.png")

    #     if normaliser_class == VahadaneTileNormaliser or normaliser_class == MacenkoTileNormaliser:

    #         for i, tile in enumerate(slide_tile_list):
    #             stain1 = normaliser.convert_to_stain1(tile)
    #             write_tile(stain1, class_output_dir / f"{i}_stain1.png")

    #         for i, tile in enumerate(slide_tile_list):
    #             stain2 = normaliser.convert_to_stain2(tile)
    #             write_tile(stain2, class_output_dir / f"{i}_stain2.png")

    #         for i, tile in enumerate(zoomed_tile_list):
    #             stain1 = normaliser.convert_to_stain1(tile)
    #             write_tile(stain1, class_output_dir_zoomed / f"{i}_stain1.png")

    #         for i, tile in enumerate(zoomed_tile_list):
    #             stain2 = normaliser.convert_to_stain2(tile)
    #             write_tile(stain2, class_output_dir_zoomed / f"{i}_stain2.png")

    # exit()

    # staincs = normaliser.convert_to_stain_concentrations(tile2)

    # print(staincs.shape)
    # print(staincs[100,100,:])

    # Stain Augmentation

    # normaliser = MacenkoStainAugmenter(0.5)
    # normaliser.fit(tile3)
    # normaliser.save_fit("macenko_tile_fit.json")
    # normaliser.load_fit("macenko_tile_fit.json")

    # from PIL import Image
    # images = []
    # for i in range(100):
    #     print(i)
    #     img_arr = normaliser.augment_colour(tile3, stain_to_augment=-1)
    #     images.append(Image.fromarray(img_arr))

    # images[0].save('macenko_caugment.gif', save_all=True, append_images=images[1:], optimize=False, duration=500, loop=0)


    #============================================================

    # Tool Testing
    # type_name = "vahadane"
    # stain_meta_file = f"{type_name}_estimated_metadata.json"
    # normalisation_class = MacenkoTileNormaliser

    # from beast.tools import StainMetadataEstimator, AggreationMethod
    # estimator = StainMetadataEstimator(normalisation_class, AggreationMethod.PERCENTILE, terminal_log_level="INFO")
    # test = estimator.estimate([tile1, tile2, tile3])
    # print(test)
    # estimator.save_fit(stain_meta_file)

    # from beast.tools import TesselatedWSINormaliser
    # normaliser = TesselatedWSINormaliser(normalisation_class, "threshold", num_threads=32)
    # normaliser.load_fit(stain_meta_file)
    # tesselated_slide = normaliser.normalise(slide_path2, "./tesselated_test", return_slide=True)
    # write_ome_tiff(tesselated_slide, *get_mpp(slide_path2), f"{type_name}_tesselated_normed.ome.tiff")

    # from beast.tools import ParallelTileNormaliser
    # normaliser = ParallelTileNormaliser(MacenkoTileNormaliser, 32, logfile="normalisation.log")
    # normaliser.load_fit("vahadane_etsimated_metadata.json")
    # normaliser.normalise([tile1, tile2, tile3], "./parallel_test")


    # Evaluation ==============================================================================================

    # - Evaluation of normalisation methods

    # - For each type, investigate: 
    #     - The estimation aggregation method. Percentiles (0.5, 0.75, 0.9, 0.95, 0.98, 0.99)
    #     - The stain intesnity calculation method. (mean, median, max, min, percentile) percentiles (0.5, 0.75, 0.9, 0.95, 0.98, 0.99)

    from pathlib import Path
    import random
    import shutil

    # from beast.tools import StainMetadataEstimator
    # from beast.utils import AggregationMethod
    
    # estimator = StainMetadataEstimator(MacenkoTileNormaliser, AggregationMethod.MEDIAN, terminal_log_level="INFO")

    reference_dataset_dir = Path("/mnt/mass-storage/cwalsh/datasets/raw/nct-crc-he/train")
    class_list = ["adi", "back", "deb", "lym", "muc", "mus", "norm", "str", "tum"]

    # tile_list = []
    # for class_name in class_list:
    #     class_dir = reference_dataset_dir / class_name
    #     class_tile_list = list(class_dir.glob("*.tif"))
    #     tile_list.extend(random.sample(class_tile_list, 100))
    #     # tile_list.extend(class_tile_list)
    
    # nct100k_metadata = estimator.estimate(tile_list)
    # print(nct100k_metadata)
    # estimator.save_fit("nct100k_normed.json")
    # exit()

    from beast.tools import ParallelTileNormaliser

    unnormalised_nct = Path("/mnt/mass-storage/cwalsh/datasets/raw/nct-crc-he-no-norm/train")

    normaliser = ParallelTileNormaliser(MacenkoTileNormaliser, num_threads=32, terminal_log_level="INFO")
    normaliser.load_fit("nct100k_normed.json")

    no_norm_dir = Path("nct100k_no_norm")
    no_norm_dir.mkdir(parents=True, exist_ok=True)

    for class_name in class_list:
        class_dir = unnormalised_nct / class_name
        class_tile_list = list(class_dir.glob("*.tif"))
        class_tile_list = random.sample(class_tile_list, 10)
        
        no_norm_dir_class = no_norm_dir / class_name
        no_norm_dir_class.mkdir(parents=True, exist_ok=True)

        for tile in class_tile_list:
            tile_name = tile.name
            shutil.copy(tile, no_norm_dir_class / tile_name)            

        normaliser.normalise(class_tile_list, f"nct100k_macenko_normed_est_median/{class_name}")

    # TODO
    # Add an enum to determine the stain intensity calculation method
    # Write a KL divergence run method

    
    
