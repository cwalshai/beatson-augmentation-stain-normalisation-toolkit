import cv2
import sys
import numpy as np
import tifffile as tf
from pathlib import Path
from PIL import Image

def get_mpp(slide_pth):
    try:
        image_ptr = tf.TiffFile(slide_pth).pages[0]
        tags = image_ptr.tags
        xres = tags.get("XResolution")
        xresv = xres.value
        xresv = xresv[1]/xresv[0]

        yres = tags.get("YResolution")
        yresv = yres.value
        yresv = yresv[1]/yresv[0]

        xresv = xresv*10000
        yresv = yresv*10000

        return xresv, yresv
    except Exception as e:
        try:
            import openslide as ops
            slide = ops.OpenSlide(slide_pth)
            mppx = float(slide.properties[ops.PROPERTY_NAME_MPP_X])
            mppy = float(slide.properties[ops.PROPERTY_NAME_MPP_Y])

            return mppx, mppy
        except Exception as e:
            raise Exception(f"Could not get mpp for {slide_pth}. {e}")

def write_ome_tiff(slide_arr:np.ndarray, mppx:float, mppy:float, output_path:Path, compression_arg:str="jpeg", levels:int=8, level_zero_uncompressed=True) -> None:
    """
        Write a whole slide image to a OME-TIFF file.

        Args:
            slide_arr <np.ndarray>: Numpy array of the whole slide image.
            mppx <float>: Microns per pixel in the x direction.
            mppy <float>: Microns per pixel in the y direction.
            compression_arg <str>: Compression to use for the OME-TIFF file.
            levels <int>: Number of levels to write to the OME-TIFF file.
            level_zero_uncompressed <bool>: Whether to write level zero uncompressed.

        Returns:
            None

        Raises:
            None

        Notes:

    """
    metadata={"Pixels": {"PhysicalSizeX":mppx, "PhysicalSizeY":mppy}}

    if level_zero_uncompressed:
        compression = None
    else:
        compression = compression_arg

    if sys.version_info.minor < 7:
        options = {"tile": (256, 256), "compress": compression, "metadata":metadata}
    else:
        options = {"tile": (256, 256), "compression": compression, "metadata":metadata}
    
    with tf.TiffWriter(output_path, bigtiff=True) as tif:
        tif.save(slide_arr, subifds=levels, **options)
        options.update({"subfiletype":1, "compression":compression_arg})
        for _ in range(0, levels):
            slide_arr = cv2.resize(slide_arr,(slide_arr.shape[1] // 2, slide_arr.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
            tif.save(slide_arr, **options)

def write_tile(tile_arr:np.ndarray, output_path:Path) -> None:
    """
        Write a tile to a file.

        Args:
            tile_arr <np.ndarray>: Numpy array of the tile.
            output_path <Path>: Path to write the tile to.
    """
    img = Image.fromarray(tile_arr)
    img.save(output_path)
