"""
Use TIAToolbox Semantic Segmentor to get dysplasia segmentations with Trans-UNet.

Usage:
  dysplasia_segmentation.py [options] [--help] [<args>...]
  dysplasia_segmentation.py --version
  dysplasia_segmentation.py (-h | --help)
  
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

Use `dysplasia_segmentation.py --help` to show their options and usage
"""

from docopt import docopt
import os
import glob
import shutil

import numpy as np
import cv2
from skimage import morphology

from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor, IOSegmentorConfig
from tiatoolbox.utils.misc import imwrite

from models.net_desc import TransUNet

# proc functions
def wsi_post_proc(wsi_seg: np.ndarray, fx: int=1) -> np.ndarray:
    """
    Post processing for WSI-level segmentations.
    """
    wsi_seg = wsi_seg[..., 1]
    wsi_seg = np.where(wsi_seg >= 0.5,1,0)

    wsi_seg = np.around(wsi_seg).astype("uint8")  # ensure all numbers are integers
    min_size = int(10000 * fx * fx)
    min_hole = int(15000 * fx * fx)
    kernel_size = int(11*fx)
    if kernel_size % 2 == 0:
        kernel_size += 1
    ep_close = cv2.morphologyEx(
        wsi_seg, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size))
    ).astype("uint8")
    ep_open = cv2.morphologyEx(
        ep_close, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size))
    ).astype(bool)
    ep_open2 = morphology.remove_small_objects(
        ep_open.astype('bool'), min_size=min_size
    ).astype("uint8")
    ep_open3 = morphology.remove_small_holes(
        ep_open2.astype('bool'), min_hole).astype('uint8')
    return ep_open3

def process_segmentation(seg_path: str, out_path: str, colour_dict: dict, mode: str) -> None:
    """
    Post-processing for WSI-level segmentations.
    """
    seg = np.load(seg_path)
    if mode == "wsi":
        seg = wsi_post_proc(seg, fx=1)
    else:
        seg = wsi_post_proc(seg, fx=1)#0.5)
    seg_col = np.expand_dims(seg, axis=2)
    seg_col = np.repeat(seg_col, 3, axis=2)
    for key, value in colour_dict.items():
        seg_col[seg == value[0]] = value[1]
    seg_col = seg_col.astype('uint8')
    imwrite(out_path, seg_col)
    return None


def segment_dysplasia(
    input_wsi_dir: str,
    output_dir: str,
    model_weights: str,
    colour_dict: dict,
    mode: str | str = "wsi",
    nr_loader_workers: int = 10,
    nr_post_proc_workers: int = 10,
    batch_size: int = 8,
    ) -> None:
    """
    Segment dysplasia using TransUNet.
    """

    if colour_dict is None:
        colour_dict = {
            "dysplasia": [1, [255, 0, 0]],
            "normal": [0, [0, 255, 0]],
        }

    wsi_file_list = glob.glob(input_wsi_dir + "*")

    # Load model
    transformer = TransUNet(
        encoder="R50-ViT-B_16",
        num_types=2,
        weights=model_weights,
        )

    segmentor = SemanticSegmentor(
        model=transformer,
        num_loader_workers=nr_loader_workers,
        num_postproc_workers=nr_post_proc_workers,
        batch_size=batch_size,
    )
    
    # Define the I/O configurations
    iostate = IOSegmentorConfig(
        input_resolutions=[{"units": "mpp", "resolution": 1.0}],
        output_resolutions=[{"units": "mpp", "resolution": 1.0}],
        patch_input_shape=[512, 512],
        patch_output_shape=[328, 328],
        stride_shape=[328, 328],
        save_resolution={"units": "mpp", "resolution": 2.0},
    )

    # Prediction
    output = segmentor.predict(
        imgs=wsi_file_list,
        masks=None,
        save_dir=os.path.join(output_dir, "dysplasia/tmp"),
        mode=mode,
        on_gpu=True,
        crash_on_exception=True,
        ioconfig=iostate,
    )

    # Rename TIAToolbox output files to readability
    os.makedirs(output_dir, exist_ok=True)

    for out in output:
        basename = os.path.basename(out[0]).split(".")[0]
        outname = os.path.basename(out[1]).split(".")[0]
        # Post process model outputs and save as RGB images
        process_segmentation(
            seg_path=os.path.join(output_dir, "dysplasia/tmp", f"{outname}.raw.0.npy"),
            out_path=os.path.join(output_dir, "dysplasia", basename + ".png"),
            colour_dict=colour_dict,
            mode=mode,
            )
    shutil.rmtree(os.path.join(output_dir, "dysplasia/tmp"))
    
    return


if __name__ == '__main__':
    args = docopt(__doc__, help=False, options_first=True, 
                    version='ODYN Dysplasia Segmentation Inference')

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
        output_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/output3/"
    
    if args['--mode']:
        mode = args['--mode']
        if mode not in ["tile", "wsi"]:
            raise ValueError("Mode must be tile or wsi")
    else:
        mode = "wsi" # or tile
        
    if args['--model_checkpoint']:
        checkpoint_path = args['--model_checkpoint']
    else:
        checkpoint_path = "/data/ANTICIPATE/outcome_prediction/ODYN_inference/weights/transunet_external.tar"
        
    colour_dict = {
        "nolabel": [0, [0  ,   0,   0]],
        "dysplasia": [1, [255, 0,   0]]
    }
    
    segment_dysplasia(
        input_wsi_dir=input_wsi_dir,
        output_dir=output_dir,
        model_weights=checkpoint_path,
        colour_dict=colour_dict,
        mode=mode,
        nr_loader_workers=int(args['--nr_loader_workers']),
        nr_post_proc_workers=int(args['--nr_post_proc_workers']),
        batch_size=int(args['--batch_size']),
        )