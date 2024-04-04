"""
Generate IEL scores using masks and nuclei segmentations.
Note, within this work we use a dysplasia mask that is the overlap of the Tranformer-produced dysplasia map and the HoVer-Net+ produced epithelium map.
We additionally get the IEL nuclear segmentations from the HoVer-Net+ model.

Warning! HoVer-Net+ output MPP and nuclear types are hardcoded. I.e. MPP=0.5 and IEL nuclei are type 1.

Usage:
  iel_scoring.py [options] [--help] [<args>...]
  iel_scoring.py --version
  iel_scoring.py (-h | --help)
  
Options:
  -h --help                   Show this string.
  --version                   Show version.

  --input_wsi_dir=<string>    Path to input directory containing slides or images.
  --input_mask_dir=<string>   Path to input directory containing slides or images.
  --input_nuc_dir=<string>    Path to input directory containing slides or images.
  --output_dir=<string>       Path to output directory to save results.
  --mode=<string>             Tile-level or WSI-level mode. [default: wsi]
  --patch_size=<n>            Patch size. [default: 128]
  --patch_stride=<n>          Patch stride. [default: 64]
  --nr_workers=<n>            Number of workers. [default: 10]

Use `iel_scoring.py --help` to show their options and usage
"""

from docopt import docopt
import os
import glob
import shutil

import csv
import joblib
import multiprocessing
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import ndimage
import cv2
from skimage import morphology
from shapely.geometry import shape, Point, Polygon, box, MultiPolygon
from shapely.strtree import STRtree

from matplotlib import pyplot as plt

from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.utils.misc import imread, imwrite
from tiatoolbox.tools.patchextraction import get_patch_extractor

from utils.viz_utils import decolourise, colourise
from utils.shapely_utils import get_nuclear_polygons, get_mask_polygons


def iel_scoring(
    wsi_path: str,
    mask_dir: str,
    nuc_dir: str,
    output_dir: str,
    mode: str,
    color_dict: dict,
    patch_size: int | int = 128,
    patch_stride: int | int = 64,
    viz_mpp: float | float = 2.0
    ) -> None:
    """
    Generate IEL scores using masks and nuclei segmentations.
    Note, within this work we use a dysplasia mask that is the overlap of the Tranformer-produced dysplasia map and the HoVer-Net+ produced epithelium map.
    We additionally get the IEL nuclear segmentations from the HoVer-Net+ model.
    """
    # Get file name
    basename = os.path.basename(wsi_path).split(".")[0]
    mask_path = os.path.join(mask_dir, basename + ".png")
    nuc_path = os.path.join(nuc_dir, basename + ".dat")
    
    score_path = os.path.join(output_dir, "scores.csv")
    if os.path.exists(score_path):
        scores_tmp = pd.read_csv(score_path)
        scores_tmp['slide'] = scores_tmp["slide"].astype('str')
        if len(scores_tmp.loc[(scores_tmp["slide"]==basename)]) == 1:
            print(f"IEL score exists for {basename}, skipping")
            return
        print(f"Processing {basename}")
    
    else:
        scores_tmp = pd.DataFrame(columns=["slide", "iel_index", "iel_count", "iel_peak_index", "iel_peak_count"])
    
    # Read WSI
    try:
        wsi = WSIReader.open(wsi_path)
    except:
        print(f"Failed to read {wsi_path}")
        return
    
    base_dims = wsi.slide_dimensions(resolution=0, units="level")
    base_mpp = wsi.info.mpp
    thumb = wsi.slide_thumbnail(resolution=viz_mpp, units="mpp")
    thumb_dims = np.asarray(thumb.shape[:2][::-1])
    
    # Read mask
    mask = imread(mask_path)
    mask = decolourise(mask, color_dict)
    mask_dims = mask.shape[::-1]
    
    # Mask area
    area = np.unique(mask, return_counts=True)[1][1] # area of mask value 1 in pixels at lower res
    mask_mpp = (base_dims / mask_dims) * base_mpp
    area_microns = area * (mask_mpp[0] * mask_mpp[1]) # area of mask in microns

    # Read nuclei files. HoNerNet+ is at 0.5 mpp.
    nuc_data = joblib.load(nuc_path)
    nuc_mpp = 0.5 # hardcoded!
    nuc_dims = wsi.slide_dimensions(resolution=nuc_mpp, units="mpp")
    thumb_ds = nuc_dims / thumb_dims

    # Transform mask dictionary into shapely polygons. Process at nuc dims
    # mask_polygons = get_mask_polygons(wsi, mask, nuc_mpp, "mpp")
    mask_polygons = get_mask_polygons(wsi, mask, viz_mpp, "mpp")
    
    ## Viz mask to ensure correct
    # overlay3 = np.zeros_like(thumb)
    # for geom in mask_polygons.geoms:
    #     contour_points = np.asarray(geom.exterior.coords, dtype='int').tolist()
    #     contour_points = np.array(contour_points).reshape((-1, 1, 2))
    #     # overlay3 = cv2.drawContours(overlay3, [contour_points], 0, (0, 0, 255), 2)
    #     overlay3 = cv2.fillPoly(overlay3, [contour_points], (0, 0, 255))
    #     for hole in geom.interiors:
    #         contour_points = np.asarray(hole.coords, dtype='int').tolist()
    #         contour_points = np.array(contour_points).reshape((-1, 1, 2))
    #         overlay3 = cv2.fillPoly(overlay3, [contour_points], (0, 0, 0))
    # 
    # plt.figure();plt.imshow(overlay3)
    
    # Transform nuclear dictionary into shapely geometries with lits of contours and types
    # nuc_geometries, inst_cntrs, inst_types = get_nuclear_polygons(wsi, {"mpp": nuc_mpp, "nuc": nuc_data}, nuc_mpp, "mpp")
    nuc_geometries, inst_cntrs, inst_types = get_nuclear_polygons(wsi, {"mpp": nuc_mpp, "nuc": nuc_data}, viz_mpp, "mpp")

    spatial_indexer = STRtree(nuc_geometries)
    geo_indices = [
        idx
        for poly in mask_polygons.geoms
        for idx in spatial_indexer.query(poly)
        if nuc_geometries[idx].intersects(poly)
        # if nuc_geometries[idx].within(poly)
        ]

    # Now, iterate through nuclear geometries and check if they lie within any of the outer polygons but not in holes
    nuc_overlay = thumb.copy()
    nr_iels = 0
    nr_epith = 0
    for g_i in tqdm(geo_indices, desc="Processing Nuclei"): 
        # inst_c = inst_cntrs[nuc_geometries.index(obj)]
        geo = nuc_geometries[g_i]
        inst_t = inst_types[g_i]
        # centroid_x = int(geo.centroid.x / thumb_ds[0])
        # centroid_y = int(geo.centroid.y / thumb_ds[1])
        centroid_x = int(geo.centroid.x)
        centroid_y = int(geo.centroid.y)
        if inst_t == 1: # hardcoded that iels are 1, and epith are else
            nuc_overlay = cv2.circle(nuc_overlay, (centroid_x, centroid_y), 4, (255, 0, 0), -1) # 3
            nr_iels += 1
        else:
            nuc_overlay = cv2.circle(nuc_overlay, (centroid_x, centroid_y), 2, (0, 255, 0), -1) # 3
            nr_epith += 1
    
    # IEL scoring
    iel_index = nr_iels / area_microns # microns
    iel_count = (nr_iels / nr_epith) * 100
    iel_peak_index = "na"
    iel_peak_count = "na"
    
    # save overlay
    output_dir_viz = os.path.join(output_dir, "visualisation")
    os.makedirs(output_dir_viz, exist_ok=True)
    imwrite(os.path.join(output_dir_viz, basename + ".png"), nuc_overlay)
    print(f"Saved {basename} viz to {output_dir_viz}")
       
    # Peak scores
    img_patches = get_patch_extractor(
        input_img=wsi,
        input_mask=mask,
        method_name="slidingwindow",
        patch_size=patch_size,
        resolution=viz_mpp,
        units="mpp", #"power",
        stride=patch_stride,
        pad_mode="constant",
        pad_constant_values=255,
        within_bound=False,
    )
    
    iels_index = []
    iels_count = []
    iels_bounds = []
    for patch in tqdm(img_patches):
        item = img_patches.n-1
        bounds = img_patches.coordinate_list[item]
        bounds_ = box(*bounds)
        
        bounds_mask = [
            bounds_.intersection(poly)
            for poly in mask_polygons.geoms
            if bounds_.intersects(poly)
        ]
        box_mask_area_pixels = sum([b_m.area for b_m in bounds_mask])
        if box_mask_area_pixels == 0:
            continue
        box_mask_area_microns = box_mask_area_pixels * (viz_mpp * viz_mpp)
        
        nuc_indices = [
            geo
            for geo in spatial_indexer.query(bounds_)
            if bounds_.contains(nuc_geometries[geo])
            for poly in mask_polygons.geoms
            if poly.intersects(nuc_geometries[geo])
            ]
        patch_nuc_types = [inst_types[nuc] for nuc in nuc_indices]
        patch_nr_iels = [1 for n in patch_nuc_types if n == 1]
        patch_nr_epith = [1 for n in patch_nuc_types if n != 1]
        patch_nr_iels = sum(patch_nr_iels)
        patch_nr_epith = sum(patch_nr_epith)
        
        if (box_mask_area_microns < 1000) or (patch_nr_epith == 0): # HARDCODED! 1000 is specific to 2.0 mpp!
            # 1. Ensure enough tissue present
            # 2. If no epithelial nuclei are present then likely to be an error in processing
            iels_count.append(0)
            iels_index.append(0)
        else:
            # Peak Index
            iel_patch_index = patch_nr_iels / box_mask_area_microns
        
            # Peak Count
            iel_patch_count = (patch_nr_iels / patch_nr_epith) * 100
            
            if iel_patch_count > 500:
            # If > 500 make 0 as unlikely to be real. Alternatively cap at 500?
                iel_patch_count = 0
                iel_patch_index = 0
            
            iels_count.append(iel_patch_count)
            iels_index.append(iel_patch_index)
        
        iels_bounds.append(bounds)

    ds = base_dims / thumb_dims
    slide_level_score_df = pd.DataFrame({
        'start_x': (np.asarray(iels_bounds)[:,0]*ds[0]).astype(int),
        'start_y': (np.asarray(iels_bounds)[:,1]*ds[1]).astype(int),
        'end_x': (np.asarray(iels_bounds)[:,2]*ds[0]).astype(int),
        'end_y': (np.asarray(iels_bounds)[:,3]*ds[1]).astype(int),
        'iel_index': iels_index,
        'iel_count': iels_count
        })
    
    output_csv_dir = os.path.join(output_dir, "scores")
    csv_path = os.path.join(output_csv_dir, f"{basename}.csv")
    os.makedirs(output_csv_dir, exist_ok=True)
    slide_level_score_df.to_csv(csv_path, index=False)    
    
    # Find max scores and visualise tiles
    max_score_slide_1 = slide_level_score_df.nlargest(1, "iel_index")
    max_1_bounds = int(max_score_slide_1.iloc[0]["start_x"]), int(max_score_slide_1.iloc[0]["start_y"]), int(max_score_slide_1.iloc[0]["end_x"]), int(max_score_slide_1.iloc[0]["end_y"])
    iel_peak_index = max_score_slide_1.iloc[0]["iel_index"]
    max_score_slide_2 = slide_level_score_df.nlargest(1, "iel_count")
    max_2_bounds = int(max_score_slide_2.iloc[0]["start_x"]), int(max_score_slide_2.iloc[0]["start_y"]), int(max_score_slide_2.iloc[0]["end_x"]), int(max_score_slide_2.iloc[0]["end_y"])
    iel_peak_count = max_score_slide_2.iloc[0]["iel_count"] 
    
    # save scores to csv
    scores_tmp.loc[len(scores_tmp)] = [basename, iel_index, iel_count, iel_peak_index, iel_peak_count]    
    scores_tmp.to_csv(score_path, index=False)
    print(f"Saved {basename} scores to {score_path}")
    
    # TODO: Save best patches!

    # # VIZ contours for patches
    # overlay = patch.copy()
    # for b_m in bounds_mask:
    # coords = np.array(b_m.exterior.coords)
    # coords[:,0] = coords[:,0] - bounds[0]
    # coords[:,1] = coords[:,1] - bounds[1]
    # overlay = cv2.drawContours(overlay, [np.array(coords, dtype='int')], 0, (0, 0, 255), 2)
    # for hole in b_m.interiors:
    #     coords = np.array(hole.coords)
    #     coords[:,0] = coords[:,0] - bounds[0]
    #     coords[:,1] = coords[:,1] - bounds[1]        
    #     overlay = cv2.fillPoly(overlay, [np.array(coords, dtype='int')], 0, (0, 0, 0), 2)
        
    # plt.figure()
    # plt.imshow(overlay)

    # overlay = patch.copy()
    # for nuc in nuc_indices:
    # geom = nuc_geometries[nuc]
    # if geom.type != "Polygon":
    #     print(nuc)
    #     continue
    # nuc_t = inst_types[nuc]
    # coords = np.array(geom.exterior.coords)
    # coords[:,0] = coords[:,0] - bounds[0]
    # coords[:,1] = coords[:,1] - bounds[1]
    # if nuc_t == 1:
    #     overlay = cv2.drawContours(overlay, [np.array(coords, dtype='int')], 0, (255, 0, 0), 2)
    # else:
    #     overlay = cv2.drawContours(overlay, [np.array(coords, dtype='int')], 0, (0, 255, 0), 2)

    # plt.figure()
    # plt.imshow(overlay)
    
    return


def iel_scoring_wrapper(
    input_wsi_dir: str,
    input_mask_dir: str,
    input_nuc_dir: str,
    output_dir: str,
    mode: str,
    color_dict: dict,
    patch_size: int | int = 128,
    patch_stride: int | int = 64, 
    nr_workers: int | int = 10,
    ) -> None:
    """
    Wrapper function for iel_scoring.py.
    This function is used to parallelize the iel_scoring.py script.
    It takes in the input_wsi_dir, input_mask_dir, input_nuc_dir, output_dir, mode, and nr_workers as arguments.
    It then calls iel_scoring on each file in the input_wsi_dir.
    It then closes the pool and waits for all processes to finish.
    It then returns None.
    
    Args:
        input_wsi_dir (str): Path to input directory containing slides or images.
        input_mask_dir (str): Path to input directory containing slides or images.
        input_nuc_dir (str): Path to input directory containing slides or images.
        output_dir (str): Path to output directory to save results.
        mode (str): Tile-level or WSI-level mode.
        nr_workers (int): Number of workers.
    
    Returns:
        None.
    
    Raises:
        ValueError: If mode is not "tile" or "wsi".
    """
    wsi_file_list = glob.glob(input_wsi_dir + "*.*")#[1:]
    for wsi_file in wsi_file_list:
        iel_scoring(
            wsi_file, input_mask_dir, input_nuc_dir, output_dir, mode,
            color_dict, patch_size, patch_stride
        )
    
    # pool = multiprocessing.Pool(processes=nr_workers)
    # pool.starmap(iel_scoring, [(
    #     wsi_file, input_mask_dir, input_nuc_dir, output_dir, mode,
    #     patch_size, patch_stride, color_dict
    #     ) for wsi_file in wsi_file_list])
    # # Close the pool to free resources
    # pool.close()
    # # Wait for all processes to finish
    # pool.join()


if __name__ == '__main__':
    args = docopt(__doc__, help=False, options_first=True, 
                    version='IEL scoring')

    if args['--help']:
        print(__doc__)
        exit()

    if args['--input_wsi_dir']:
        input_wsi_dir = args['--input_wsi_dir']
    else:      
        input_wsi_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/wsis/"

    if args['--input_mask_dir']:
        input_mask_dir = args['--input_mask_dir']
    else:      
        input_mask_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/output_epith/combined/"
        
    if args['--input_nuc_dir']:
        input_nuc_dir = args['--input_nuc_dir']
    else:      
        input_nuc_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/output_epith/nuclei/"
    
    if args['--output_dir']:
        output_dir = args['--output_dir']
    else:
        output_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/output_epith/iels/"
    
    if args['--mode']:
        mode = args['--mode']
        if mode not in ["tile", "wsi"]:
            raise ValueError("Mode must be tile or wsi")
    else:
        mode = "wsi" # or tile
 
    color_dict_dysp = {
        "bkgd": [0, (0, 0, 0)],
        "dysplasia": [1, (255, 0, 0)],
    }   
    
    iel_scoring_wrapper(
        input_wsi_dir=input_wsi_dir,
        input_mask_dir=input_mask_dir,
        input_nuc_dir=input_nuc_dir,
        output_dir=output_dir,
        mode=mode,
        nr_workers=int(args['--nr_workers']),
        patch_size=int(args['--patch_size']),
        patch_stride=int(args['--patch_stride']),
        color_dict=color_dict_dysp,
        )
    