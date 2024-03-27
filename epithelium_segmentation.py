"""
Use TIAToolbox Multi-Task Segmentor to get nuclear/epithelial layer segmentations with HoVer-Net+.

Usage:
  epithelium_segmentation.py [options] [--help] [<args>...]
  epithelium_segmentation.py --version
  epithelium_segmentation.py (-h | --help)
  
Options:
  -h --help                   Show this string.
  --version                   Show version.

  --input_dir=<string>        Path to input directory containing slides or images.
  --output_dir=<string>       Path to output directory to save results.
  --mode=<string>             Tile-level or WSI-level mode. [default: wsi]
  --model_checkpoint=<string> Path to model weights.
  --nr_loader_workers=<n>     Number of workers during data loading. [default: 10]
  --nr_post_proc_workers=<n>  Number of workers during post-processing. [default: 10]
  --batch_size=<n>            Batch size. [default: 8]

Use `epithelium_segmentation.py --help` to show their options and usage
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

from tiatoolbox.models.engine.multi_task_segmentor import MultiTaskSegmentor
from tiatoolbox.models.architecture.hovernetplus import HoVerNetPlus
from tiatoolbox.utils.misc import imwrite

from models.net_utils import convert_pytorch_checkpoint

# proc functions
def smooth_ep_ker_boundary(
    layer_map: np.ndarray,
    mpp: float,
    ) -> np.ndarray:
    """
    Smooth epithelial/keratin boundary.
    """
    fx = (2 / 2) * mpp
    kernel_size = int(5*fx)

    if kernel_size % 2 == 0:
        kernel_size += 1
    
    min_size = int(1000/(fx*fx))
    basal_map = (layer_map == 2).astype('uint8')
    tissue_map = (layer_map == 1).astype('uint8')
    epith_map = layer_map > 2
    epith_map_binary = epith_map.astype('uint8')
    epith_map = epith_map_binary*layer_map
    epith_lyrs = np.unique(epith_map)[1:]  # exclude background
    layer_map = tissue_map
    epith_new = epith_map_binary.astype('uint8')*3
    layer_map = epith_new.copy()
    layer_map[tissue_map == 1] = 1
    
    for lyr in [4, 3]:
        if lyr in epith_lyrs:
            epith_lyr = epith_map == lyr #np.where(epith_map, lyr, 0).astype('uint8')
            epith_lyr = epith_lyr.astype('uint8')
            epith_lyr = cv2.morphologyEx(
                epith_lyr, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size))
            )
            epith_lyr = cv2.morphologyEx(
                epith_lyr, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size))
            )
            epith_lyr = morphology.remove_small_objects(epith_lyr, min_size)
            epith_lyr = morphology.remove_small_holes(epith_lyr, min_size)
            layer_map[epith_lyr == 1] = lyr

    layer_map[basal_map == 1] = 2
    
    return layer_map


def remove_spurious_epith(
    layer_map: np.ndarray,
    ) -> np.ndarray:
    """
    Remove spurious epithelium. Replace with remove_small_objects....
    """
    epith_map = layer_map > 1
    epith_map_binary = epith_map.astype('uint8')
    epith_map = epith_map_binary*layer_map
    epith_map_labelled = ndimage.label(epith_map_binary)[0]
    epith_id_list = np.unique(epith_map_labelled)[1:]  # exclude background
    
    if len(epith_id_list) >= 1:
        for e in epith_id_list:
            epith = epith_map_labelled == e
            epith_binary = epith.astype('int')
            epith = epith_binary*epith_map
            vals, counts = np.unique(epith, return_counts=True)
            vals = vals[1:]
            counts = counts[1:]
            if len(vals) > 1:
                counts_f = []
                vals_f = []
                max_c = np.max(counts)
                for idx, c in enumerate(counts):
                    if c > max_c//10:
                        counts_f.append(c)
                        vals_f.append(vals[idx])
                if len(vals_f) == 1:
                    layer_map[epith_binary==1] = 1
            else:
                layer_map[epith_binary==1] = 1
        
    return layer_map

def process_segmentation(seg_path: str, out_path: str, colour_dict: dict, mode: str) -> None:
    """
    Post-processing for WSI-level segmentations.
    """
    seg = np.load(seg_path)
    seg = smooth_ep_ker_boundary(seg, 0.5) # or is it 2    
    seg = remove_spurious_epith(seg)
    seg_col = np.expand_dims(seg, axis=2)
    seg_col = np.repeat(seg_col, 3, axis=2)
    for key, value in colour_dict.items():
        seg_col[seg == value[0]] = value[1]
    seg_col = seg_col.astype('uint8')
    imwrite(out_path, seg_col)
    return None

def segment_epithelium(
    input_wsi_dir: str,
    output_dir: str,
    model_weights: str | str = None,
    colour_dict: dict = None,
    mode: str = "wsi",
    nr_loader_workers: int | int = 10,
    nr_post_proc_workers: int | int = 10,
    batch_size: int | int = 8, 
) -> None:
    
    wsi_file_list = glob.glob(input_wsi_dir + "*")

    multi_segmentor = MultiTaskSegmentor(
        pretrained_model="hovernetplus-oed",
        num_loader_workers=nr_loader_workers,
        num_postproc_workers=nr_post_proc_workers,
        batch_size=batch_size,
        auto_generate_mask=False,
    )
    
    if model_weights is not None: # Then use new weights
        pretrained_weights = torch.load(model_weights, map_location=torch.device("cuda"))
        state_dict = pretrained_weights['desc']
        state_dict = convert_pytorch_checkpoint(state_dict)
        # Iterate through the keys of the dictionary
        for key in list(state_dict.keys()):
            # Check if ".tp." is in the key
            if ".tp2." in key:
                # Replace ".tp." with ".tp2."
                new_key = key.replace(".tp2.", ".tp.")
                # Update the dictionary with the new key and the corresponding value
                state_dict[new_key] = state_dict.pop(key)
        multi_segmentor.model.load_state_dict(state_dict, strict=True)

    # WSI prediction
    wsi_output = multi_segmentor.predict(
        imgs=wsi_file_list,
        masks=None,
        save_dir=os.path.join(output_dir, "epith/tmp"),
        mode=mode, #"wsi",
        on_gpu=True,
        crash_on_exception=True,
    )

    # Rename TIAToolbox output files to readability
    layer_dir = os.path.join(output_dir, "epith", "layers")
    nuclei_dir = os.path.join(output_dir, "epith", "nuclei")
    os.makedirs(layer_dir, exist_ok=True)
    os.makedirs(nuclei_dir, exist_ok=True)

    for out in wsi_output:
        basename = os.path.basename(out[0]).split(".")[0]
        outname = os.path.basename(out[1]).split(".")[0]
        process_segmentation(
            seg_path=os.path.join(output_dir, "epith/tmp", f"{outname}.1.npy"),
            out_path=os.path.join(layer_dir, basename + ".png"),
            colour_dict=colour_dict,
            mode=mode,
        )
        # process nuclei too!
        shutil.move(
            os.path.join(output_dir, "epith/tmp", f"{outname}.0.dat"),
            os.path.join(nuclei_dir, basename + ".dat"),
            )
    shutil.rmtree(os.path.join(output_dir, "epith/tmp"))

    return
    

if __name__ == '__main__':
    args = docopt(__doc__, help=False, options_first=True, 
                    version='ODYN HoVer-Net+ Inference')

    if args['--help']:
        print(__doc__)
        exit()

    if args['--input_dir']:
        input_wsi_dir = args['--input_dir']
    else:      
        input_wsi_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/wsis/"
    
    if args['--output_dir']:
        output_dir = args['--output_dir']
    else:
        output_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/output4/"
    
    if args['--mode']:
        mode = args['--mode']
        if mode not in ["tile", "wsi"]:
            raise ValueError("Mode must be tile or wsi")
    else:
        mode = "wsi" # or tile
        
    if args['--model_checkpoint']:
        checkpoint_path = args['--model_checkpoint']
    else:
        checkpoint_path = "/data/ANTICIPATE/outcome_prediction/ODYN_inference/weights/hovernetplus.tar"
        

    epith_colour_dict = {
        "nolabel": [0, [0  ,   0,   0]],
        "other": [1, [255, 165,   0]],
        "basal": [2, [255, 0,   0]],
        "epith": [3, [0,   255,   0]],
        "keratin": [4, [0,   0,   255]],
    }
    
    segment_epithelium(
        input_wsi_dir=input_wsi_dir,
        output_dir=output_dir,
        model_weights=checkpoint_path,
        colour_dict=epith_colour_dict,
        mode=mode,
        nr_loader_workers=int(args['--nr_loader_workers']),
        nr_post_proc_workers=int(args['--nr_post_proc_workers']),
        batch_size=16,#int(args['--batch_size']),
        )
