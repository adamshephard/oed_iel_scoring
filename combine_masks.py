"""
Use this script to generate a combined dysplasia-epith map. (keep as 2 classes dysplasia vs epith).

Usage:
  combine_masks.py [options] [--help] [<args>...]
  combine_masks.py --version
  combine_masks.py (-h | --help)
  
Options:
  -h --help                   Show this string.
  --version                   Show version.

  --input_epith=<string>      Path to input directory containing HoVer-Net+ epithelium maps.
  --input_dysplasia=<string>  Path to input directory containing the Transformer dysplasia maps.
  --output_dir=<string>       Path to output directory to save results.

Use `combine_masks.py --help` to show their options and usage
"""

from docopt import docopt
import os
import glob
import shutil

import torch
import numpy as np
from scipy import ndimage
import cv2
from skimage import morphology

from tiatoolbox.utils.misc import imread, imwrite
from utils.viz_utils import decolourise, colourise


def combine_masks(
    input_epith_dir: str,
    input_dysp_dir: str,
    output_dir: str,
    epith_colour_dict: dict,
    dysp_colour_dict: dict,
    new_colour_dict: dict,
    ) -> None:
    """
    Combine epithelium and dysplasia masks into a new combined mask.
    """
    os.makedirs(output_dir, exist_ok=True)
    for epith_file in sorted(glob.glob(os.path.join(input_epith_dir, "*.png"))):
        # read in images
        basename = os.path.basename(epith_file).split(".")[0]
        dysp_file = os.path.join(input_dysp_dir, basename + ".png")
        epith_img = imread(epith_file)
        dysp_img = imread(dysp_file)
        epith_img = decolourise(epith_img, epith_colour_dict)
        dysp_img = decolourise(dysp_img, dysp_colour_dict)
        
        # combine masks - process at dysplasia resolution
        combined_img = epith_img.copy()
        if dysp_img.shape != combined_img.shape:
            combined_img = cv2.resize(combined_img, (dysp_img.shape[1], dysp_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Commented out uses the union of both methods
        # combined_img[combined_img >= 2] = 3
        # combined_img[dysp_img == 1] = 2
        
        # Use the overlap of both methods
        epith = combined_img.copy()
        epith[epith <= 1] = 0
        epith[epith >= 2] = 1
        dysp_epith = dysp_img*epith

        tissue_img = combined_img.copy()
        tissue_img[tissue_img >= 1] = 1
        tissue_img[dysp_img >= 1] = 1
        
        combined_img = tissue_img.copy()
        combined_img[epith == 1] = 3
        combined_img[dysp_epith == 1] = 2
        
        combined_img_col = colourise(combined_img, new_colour_dict)

        # save combined mask
        imwrite(os.path.join(output_dir, basename + ".png"), combined_img_col)


if __name__ == '__main__':
    args = docopt(__doc__, help=False, options_first=True, 
                    version='ODYN HoVer-Net+ Inference')

    if args['--help']:
        print(__doc__)
        exit()

    if args['--input_epith']:
        input_epith_dir = args['--input_epith']
    else:      
        input_epith_dir = "/data/ANTICIPATE/github/testdata/output_epith4/epith/"
        
    if args['--input_dysplasia']:
        input_dysp_dir = args['--input_dysplasia']
    else:      
        input_dysp_dir = "/data/ANTICIPATE/github/testdata/output_epith4/dysplasia/"        
    
    if args['--output_dir']:
        output_dir = args['--output_dir']
    else:
        output_dir = "/data/ANTICIPATE/github/testdata/output_epith4/combined/"       

    epith_colour_dict = {
        "nolabel": [0, [0  ,   0,   0]],
        "other": [1, [255, 165,   0]],
        "basal": [2, [255, 0,   0]],
        "epith": [3, [0,   255,   0]],
        "keratin": [4, [0,   0,   255]],
    }
    dysp_colour_dict = {
        "nolabel": [0, [0  ,   0,   0]],
        "dysplasia": [1, [255, 0,   0]],
    }
    
    new_colour_dict = {
        "nolabel": [0, [0  ,   0,   0]],
        "other": [1, [255, 165,   0]],
        "dysplasia": [2, [255, 0,   0]],
        "epith": [3, [0,   255,   0]],
    }  
    
    combine_masks(
        input_epith_dir,
        input_dysp_dir,
        output_dir,
        epith_colour_dict,
        dysp_colour_dict,
        new_colour_dict,
    )

