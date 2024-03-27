"""
Code for generating Intra-epithelial lymphocyte (IEL) scores.
Script requires WSI images for a case, and dysplasia segmentations.
We also need HoVer-Net+ segmentations for the intra-epithelial lymphocytes.

Target:
1) Nr IELs per unit dysplasia area (in microns)
2) Nr IELs per 100 epithelial cells
3) Nr IELs per unit dysplasia area (in microns) in most dense field of view
4) Nr IELs per 100 epithelial cells in most dense field of view

Created by: AS
Creation Date: 29/06/2023
Project: ANTICIPATE
"""

from torch.multiprocessing import Pool, RLock, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

import os
import glob
import json
import numpy as np
import pandas as pd
import cv2
import csv
from matplotlib import pyplot as plt
from tqdm import tqdm

from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.utils.misc import imread, imwrite
from tiatoolbox.tools.patchextraction import get_patch_extractor

from utils.viz_utils import decolourise


PATCH_SIZE = 128#256
PATCH_MPP = 2
PATCH_STRIDE = PATCH_SIZE // 2 #// 4


# Functions
def count_iels_mask(wsi, hovernet_json, mask):
    """
    Counts nr of IELs and epithelial cells in a mask.
    Note. hoverent nuclei are at 0.5mpp, mask is different.
    """
    nuc_mpp = hovernet_json["mpp"]
    nuc_dims = wsi.slide_dimensions(resolution=nuc_mpp, units="mpp")
    mask_dims = mask.shape[::-1]
    ds = mask_dims / nuc_dims
    nuc_dict = hovernet_json["nuc"]
    iels = 0
    eps = 0
    for nuc in nuc_dict.values():
        centroid = (nuc["centroid"] * ds).astype('int')
        nuc_type = nuc["type"]
        if (mask[centroid[1], centroid[0]] == 1) and nuc_type == 1:
            iels += 1
        elif (mask[centroid[1], centroid[0]] == 1) and nuc_type in [2, 3, 4]: # since hovernet masks have nuclei from 2-4
            eps += 1
    return iels, eps

def vis_iels(wsi, hovernet_json, mask, res=2.5):
    """
    Visualise the mask and the IELs at 2.5X.
    """
    nuc_mpp = hovernet_json["mpp"]
    nuc_dims = wsi.slide_dimensions(resolution=nuc_mpp, units="mpp")
    mask_dims = mask.shape[::-1]
    ds = mask_dims / nuc_dims
    nuc_dims = wsi.slide_dimensions(units="mpp", resolution=nuc_mpp)
    thumb = wsi.slide_thumbnail(units="power", resolution=res)
    thumb_dims = wsi.slide_dimensions(units="power", resolution=res)
    thumb_ds = thumb_dims / nuc_dims
    nuc_dict = hovernet_json["nuc"]
    iels = 0
    for nuc in nuc_dict.values():
        centroid = (nuc["centroid"] * ds).astype('int')
        centroid2 = (nuc["centroid"] * thumb_ds).astype('int')
        nuc_type = nuc["type"]
        if (mask[centroid[1], centroid[0]] == 1) and nuc_type == 1:
            thumb = cv2.circle(thumb, (centroid2[0], centroid2[1]), int(2 * res), (255, 0, 0), 2)
            iels += 1
    return thumb, iels

def count_iels(wsi, hovernet_json, patch, patch_mask, bounds, mpp=2):
    nr_iels = 0
    nr_eps = 0
    nuc_mpp = hovernet_json["mpp"]
    nuc_dims = wsi.slide_dimensions(resolution=nuc_mpp, units="mpp")
    mask_dims = wsi.slide_dimensions(resolution=mpp, units="mpp")
    ds = nuc_dims / mask_dims
    nuc_dict = hovernet_json["nuc"]
    # overlay = patch.copy()
    for nuc in nuc_dict.values():
        centroid = (nuc["centroid"] / ds).astype('int')
        centroid[0] = centroid[0] - bounds[0]
        centroid[1] = centroid[1] - bounds[1]   
        nuc_type = nuc["type"]
        if (0 < (centroid[1]) < patch_mask.shape[0]) and (0 < (centroid[0]) < patch_mask.shape[1]):
            if (patch_mask[centroid[1], centroid[0]] == 1):
                # contours_x = np.asarray(nuc["contour"])[:,0]
                # contours_y = np.asarray(nuc["contour"])[:,1]
                # contours = np.array([contours_x/ds[0]-bounds[0], contours_y/ds[1]-bounds[1]]).T.astype('int')
                if nuc_type == 1:
                    nr_iels += 1
                    # overlay = cv2.drawContours(overlay, [contours], -1, (255,0,0), thickness=1)
                elif nuc_type in [2, 3, 4]: # since hovernet masks have nuclei from 2-4
                    nr_eps += 1
                    # overlay = cv2.drawContours(overlay, [contours], -1, (0,255,0), thickness=1)
        # plt.figure(); plt.imshow(overlay); plt.show(block=False)
    # return overlay, nr_iels, nr_eps
    return nr_iels, nr_eps

def count_iels_highres(wsi, hovernet_json, patch_mask, bounds, viz_mpp=0.5, mask_mpp=2):
    nr_iels = 0
    nr_eps = 0
    # take bounds to nuc dims
    nuc_mpp = hovernet_json["mpp"]
    nuc_dims = wsi.slide_dimensions(resolution=nuc_mpp, units="mpp")
    mask_dims = wsi.slide_dimensions(resolution=mask_mpp, units="mpp")
    ds = nuc_dims / mask_dims
    nuc_dict = hovernet_json["nuc"]
    viz_dims = wsi.slide_dimensions(units="mpp", resolution=viz_mpp)
    viz_ds = nuc_dims / viz_dims
    mask_ds = mask_dims / viz_dims
    base_dims = wsi.slide_dimensions(units="level", resolution=0)
    base_ds = base_dims / mask_dims# from mask to base
    roi = wsi.read_bounds(
        bounds=[int(bounds[0]*base_ds[0]), int(bounds[1]*base_ds[1]), int(bounds[2]*base_ds[0]), int(bounds[3]*base_ds[1])],
        units="mpp", 
        resolution=viz_mpp
        )
    if patch_mask.shape != roi.shape[:2]:
        patch_mask = cv2.resize(patch_mask, roi.shape[:2][::-1])
    overlay = roi.copy()
    for nuc in nuc_dict.values():
        centroid = (nuc["centroid"] / viz_ds).astype('int')
        centroid[0] = centroid[0] - int(bounds[0]/mask_ds[0])
        centroid[1] = centroid[1] - int(bounds[1]/mask_ds[1])  
        nuc_type = nuc["type"]
        if (0 < (centroid[1]) < patch_mask.shape[0]) and (0 < (centroid[0]) < patch_mask.shape[1]):
            if (patch_mask[centroid[1], centroid[0]] == 1):
                contours_x = np.asarray(nuc["contour"])[:,0]
                contours_y = np.asarray(nuc["contour"])[:,1]
                contours = np.array([(contours_x/viz_ds[0])-(bounds[0]/mask_ds[0]), (contours_y/viz_ds[1])-(bounds[1]/mask_ds[1])]).T.astype('int')
                if nuc_type == 1:
                    nr_iels += 1
                    overlay = cv2.drawContours(overlay, [contours], -1, (255,0,0), thickness=int(1*(1/viz_mpp)))
                elif nuc_type in [2, 3, 4]: # since hovernet masks have nuclei from 2-4
                    nr_eps += 1
                    overlay = cv2.drawContours(overlay, [contours], -1, (0,255,0), thickness=int(1*(1/viz_mpp)))
        # plt.figure(); plt.imshow(overlay); plt.show(block=False)
    return overlay, nr_iels, nr_eps

def process(wsi_path, tissue, cohort, input_dysplasia, input_hovernet, color_dict, color_dict_2, cohorts, output_folder, img_out_coh_dir, patch_level_csv_dir, patch_level_1_dir, patch_level_2_dir):
    wsi_name = os.path.basename(wsi_path)
    wsi_name = wsi_name.split(".")[0]
    patch_size = PATCH_SIZE

    # Check if output exists, skip if it does
    # if os.path.exists(os.path.join(patch_level_1_dir, f"{wsi_name}.png")):
    #     print(f"Viz output exists for {wsi_name}, skipping")
    #     return
    score_path = os.path.join(output_folder, f"iel_scoring.csv")
    if os.path.exists(score_path):
        scores_tmp = pd.read_csv(score_path)
        scores_tmp['cohort'] = scores_tmp["cohort"].astype('str')
        scores_tmp['slide'] = scores_tmp["slide"].astype('str')
        # if (scores_tmp["slide"] == wsi_name) and (scores_tmp["cohort"] == cohort):
        if len(scores_tmp.loc[(scores_tmp["slide"]==wsi_name) & (scores_tmp["cohort"] == str(cohorts[cohort]))]) == 1:
            print(f"IEL score exists for {wsi_name}, skipping")
            return
    print(f"Processing {wsi_name}")

    # Read WSI
    try:
        wsi = WSIReader.open(wsi_path)
    except:
        print(f"Failed to read {wsi_path}")
        return
    
    base_dims = wsi.slide_dimensions(resolution=0, units="level")
    base_mpp = wsi.info.mpp

    if tissue == "dysplasia":
        # Read Dysplasia
        dysplasia_path = os.path.join(input_dysplasia[cohort], wsi_name + ".png")
        try:
            dysplasia_mask = imread(dysplasia_path)
            mask = decolourise(dysplasia_mask, color_dict)
            mask_dims = mask.shape[::-1]
        except:
            print(f"No dysplasia mask for case {wsi_name}, skipping")
            return
    
    # Read HoVer-Net+ Epith
    epith_path = os.path.join(input_hovernet[cohort], "layers", wsi_name + ".png")
    try:
        epith_mask = imread(epith_path)
        epith_mask = decolourise(epith_mask, color_dict_2)
        epith_mask_dims = epith_mask.shape[::-1]
    except:
        print(f"No HoVer-Net+ epithelium mask for case {wsi_name}, skipping")
        return    
    
    epith_mask[epith_mask == 1] = 0
    epith_mask[epith_mask >= 2] = 1
    if tissue == "dysplasia":
        if mask_dims != epith_mask_dims:
            epith_mask = cv2.resize(epith_mask, mask_dims)
        mask = mask * epith_mask # ensure dysplasia anno is within epithelium
    else:
        mask = epith_mask
        mask_dims = epith_mask_dims
    
    try:
        dys_area = np.unique(mask, return_counts=True)[1][1] # area of dysplasia in pixels at lower res
        mask_mpp = (base_dims / mask_dims) * base_mpp
        dys_area_microns = dys_area * (mask_mpp[0] * mask_mpp[1]) # area of dysplasia in microns

        # Read HoverNet - note this is at 0.5 mpp
        try:
            hovernet_path = os.path.join(input_hovernet[cohort], "json", wsi_name + ".json")
            with open(hovernet_path, "r") as f:
                hovernet_json = json.load(f)
        except:
            print(f"No hovernet output for case {wsi_name}, skipping")
            return
            
        nuc_mpp = hovernet_json["mpp"]
        nuc_dims = wsi.slide_dimensions(resolution=nuc_mpp, units="mpp")

        # First score is nr iels per unit dysplasia area (in microns)
        nr_iels, nr_eps = count_iels_mask(wsi, hovernet_json, mask)
        # Visualise IELs
        viz_iels, _ = vis_iels(wsi, hovernet_json, mask, res=1.25)
        img_out_path = os.path.join(img_out_coh_dir, f"{wsi_name}.png")
        imwrite(img_out_path, viz_iels)        

        iels_per_dys_area_microns = nr_iels / dys_area_microns
        # print(f"IELs per unit dysplasia area: {iels_per_dys_area_microns}")

        # Second score is nr iels per 100 epithelial cells
        iels_per_100_epithelial_cells = (nr_iels / nr_eps) * 100
        # print(f"IELs per 100 epithelial cells: {iels_per_100_epithelial_cells}")
    
    except:
        print(f"No dysplasia found for {wsi_name}, defaulting scores to 0.")
        iels_per_dys_area_microns = 0
        iels_per_100_epithelial_cells = 0
        max_score_1 = 0
        max_score_2 = 0
        med5_max_score_1 = 0
        med5_max_score_2 = 0
        # Save scores
        if tissue == "dysplasia":
            slide_score_dict = {
                "slide": wsi_name,
                "cohort": cohorts[cohort],
                "nr_iels_per_unit_dysplasia_area": iels_per_dys_area_microns,
                "nr_iels_per_100_epithelial_cells": iels_per_100_epithelial_cells,
                "max_nr_iels_per_unit_dysplasia_area": max_score_1,
                "max_iels_per_100_epithelial_cells": max_score_2,
                "med5_max_nr_iels_per_unit_dysplasia_area": med5_max_score_1,
                "med5_max_iels_per_100_epithelial_cells": med5_max_score_2,                
            }
        else:
            slide_score_dict = {
                "slide": wsi_name,
                "cohort": cohorts[cohort],
                "nr_iels_per_unit_epithelial_area": iels_per_dys_area_microns,
                "nr_iels_per_100_epithelial_cells": iels_per_100_epithelial_cells,
                "max_nr_iels_per_unit_epithelial_area": max_score_1,
                "max_iels_per_100_epithelial_cells": max_score_2,
                "med5_max_nr_iels_per_unit_epithelial_area": med5_max_score_1,
                "med5_max_iels_per_100_epithelial_cells": med5_max_score_2,       
            }

        # If file exists, append to it, otherwise create new file
        score_path = os.path.join(output_folder, f"iel_scoring.csv")
        if os.path.exists(score_path):
            with open(score_path, "a") as f:
                writer = csv.DictWriter(f, fieldnames=slide_score_dict.keys())
                writer.writerow(slide_score_dict)
                f.close()
                print("Written to file")
        else:
            df = pd.DataFrame(data=slide_score_dict, index=[0])
            df.to_csv(score_path, index=False)
        return 

    # Third score is nr iels per unit dysplasia area (in microns) in most dense field of view
    # Will take a sliding window approach here over the epithelium using the tiatoolbox patch extractor

    img_patches = get_patch_extractor(
        input_img=wsi,
        input_mask=mask,
        method_name="slidingwindow",
        patch_size=patch_size,
        resolution=PATCH_MPP,
        units="mpp", #"power",
        stride=PATCH_STRIDE,
        pad_mode="constant",
        pad_constant_values=255,
        within_bound=False,
    )

    # Loop over patches and count nr iels and nr eps in each patch
    new_dims = wsi.slide_dimensions(resolution=2, units="mpp")
    ds = mask_dims / new_dims
    bounds_all = []
    iels_area = []
    iels_per_100 = []
    for patch in tqdm(img_patches):
        item = img_patches.n-1
        bounds = img_patches.coordinate_list[item]
        mask_bounds = [int(bounds[0]*ds[0]), int(bounds[1]*ds[1]), int(bounds[2]*ds[0]), int(bounds[3]*ds[1])]
        patch_mask = mask[mask_bounds[1]:mask_bounds[3], mask_bounds[0]:mask_bounds[2]]
        # get dysplastic area in microns
        vals, counts = np.unique(patch_mask, return_counts=True)
        if 1 in vals:
            dys_idx = np.argwhere(vals==1)[0][0] #np.where(vals==1)
            dys_area = counts[dys_idx]
        else:
            dys_area = 0
            continue
        # dys_area = np.unique(patch_mask, return_counts=True)[1][1]
        dys_area_microns = dys_area * (mask_mpp[0] * mask_mpp[1])
        if patch_mask.shape != patch.shape[:2]:
            patch_mask = cv2.resize(patch_mask, patch.shape[:2][::-1])
        # Count nr iels and nr eps in patch_mask
        # overlay, nr_iels, nr_eps = count_iels(wsi, hovernet_json, patch, patch_mask, bounds, mpp=2)
        nr_iels, nr_eps = count_iels(wsi, hovernet_json, patch, patch_mask, bounds, mpp=2)
        # overlay, nr_iels, nr_eps = count_iels_highres(wsi, hovernet_json, patch_mask, bounds, viz_mpp=0.5, mask_mpp=2)
        iels_per_dys_area_microns_ = nr_iels / dys_area_microns
        # if nr_eps == 0:
        #     nr_eps = 1 # to avoid zero division
        # if (dys_area < ((patch_size*patch_size) / 50)) or (nr_eps == 0):
        if (dys_area < 1000) or (nr_eps == 0):
            # 1. Ensure enough tissue present
            # 2. If no epithelial nuclei are present then likely to be an error in processing
            iels_area.append(0)
            iels_per_100.append(0)
        else:
            iels_per_100_epithelial_cells_ = (nr_iels / nr_eps) * 100
            if (iels_per_100_epithelial_cells_ > 500):
                iels_area.append(0)
                iels_per_100.append(0)
                # make 0 as unlikely to be real. Alternatively cap at 500?
            else:
                iels_area.append(iels_per_dys_area_microns_)
                iels_per_100.append(iels_per_100_epithelial_cells_)
        bounds_all.append(bounds)

    # Save scores
    # ds = base_dims / mask_dims
    ds = base_dims / new_dims
    slide_level_score_df = pd.DataFrame({
        'start_x': (np.asarray(bounds_all)[:,0]*ds[0]).astype(int),
        'start_y': (np.asarray(bounds_all)[:,1]*ds[1]).astype(int),
        'end_x': (np.asarray(bounds_all)[:,2]*ds[0]).astype(int),
        'end_y': (np.asarray(bounds_all)[:,3]*ds[1]).astype(int),
        'iels_per_dys_area_microns': iels_area,
        'iels_per_100_epithelial_cells': iels_per_100
        })
    
    patch_path = os.path.join(patch_level_csv_dir, f"{wsi_name}.csv")
    slide_level_score_df.to_csv(patch_path, index=False)
    
    # Find max scores and visualise tiles
    new_dims = wsi.slide_dimensions(resolution=1, units="mpp")
    ds = base_dims / new_dims # base to 1 mpp
    mask_ds = mask_dims / base_dims
    max_score_slide_1 = slide_level_score_df.nlargest(1, 'iels_per_dys_area_microns')
    max_1_bounds = int(max_score_slide_1.iloc[0]["start_x"]), int(max_score_slide_1.iloc[0]["start_y"]), int(max_score_slide_1.iloc[0]["end_x"]), int(max_score_slide_1.iloc[0]["end_y"])
    max_score_1 = max_score_slide_1.iloc[0]["iels_per_dys_area_microns"]
    mask_bounds = [int(max_1_bounds[0]*mask_ds[0]), int(max_1_bounds[1]*mask_ds[1]), int(max_1_bounds[2]*mask_ds[0]), int(max_1_bounds[3]*mask_ds[1])]
    patch_mask = mask[mask_bounds[1]:mask_bounds[3], mask_bounds[0]:mask_bounds[2]]   
    roi_1 = wsi.read_bounds((max_1_bounds[0], max_1_bounds[1], max_1_bounds[2], max_1_bounds[3]), resolution=1, units="mpp")
    patch_mask = cv2.resize(patch_mask, roi_1.shape[:2][::-1])
    max_1_bounds_05 = int(max_1_bounds[0]/ds[0]), int(max_1_bounds[1]/ds[1]), int(max_1_bounds[2]/ds[0]), int(max_1_bounds[3]/ds[1])
    overlay, nr_iels, nr_eps = count_iels_highres(wsi, hovernet_json, patch_mask, max_1_bounds_05, viz_mpp=1, mask_mpp=1)
    img_path_1 = os.path.join(patch_level_1_dir, f"{wsi_name}.png")
    imwrite(img_path_1, overlay)
    topn_max_score_slide_1 = slide_level_score_df.nlargest(5, 'iels_per_dys_area_microns')
    med5_max_score_1 = topn_max_score_slide_1["iels_per_dys_area_microns"].median()
    
    # Now for per 100 eps
    max_score_slide_2 = slide_level_score_df.nlargest(1, 'iels_per_100_epithelial_cells')
    max_2_bounds = int(max_score_slide_2.iloc[0]["start_x"]), int(max_score_slide_2.iloc[0]["start_y"]), int(max_score_slide_2.iloc[0]["end_x"]), int(max_score_slide_2.iloc[0]["end_y"])
    max_score_2 = max_score_slide_2.iloc[0]["iels_per_100_epithelial_cells"]
    mask_bounds = [int(max_2_bounds[0]*mask_ds[0]), int(max_2_bounds[1]*mask_ds[1]), int(max_2_bounds[2]*mask_ds[0]), int(max_2_bounds[3]*mask_ds[1])]
    roi_2 = wsi.read_bounds((max_2_bounds[0], max_2_bounds[1], max_2_bounds[2], max_2_bounds[3]), resolution=1, units="mpp")
    patch_mask = mask[mask_bounds[1]:mask_bounds[3], mask_bounds[0]:mask_bounds[2]]   
    patch_mask = cv2.resize(patch_mask, roi_2.shape[:2][::-1])
    max_2_bounds_05 = int(max_2_bounds[0]/ds[0]), int(max_2_bounds[1]/ds[1]), int(max_2_bounds[2]/ds[0]), int(max_2_bounds[3]/ds[1])
    overlay, nr_iels, nr_eps = count_iels_highres(wsi, hovernet_json, patch_mask, max_2_bounds_05, viz_mpp=1, mask_mpp=1)
    img_path_2 = os.path.join(patch_level_2_dir, f"{wsi_name}.png")
    imwrite(img_path_2, overlay)
    topn_max_score_slide_2 = slide_level_score_df.nlargest(5, 'iels_per_100_epithelial_cells')
    med5_max_score_2 = topn_max_score_slide_2["iels_per_100_epithelial_cells"].median()

    # Save scores
    if tissue == "dysplasia":
        slide_score_dict = {
            "slide": wsi_name,
            "cohort": cohorts[cohort],
            "nr_iels_per_unit_dysplasia_area": iels_per_dys_area_microns,
            "nr_iels_per_100_epithelial_cells": iels_per_100_epithelial_cells,
            "max_nr_iels_per_unit_dysplasia_area": max_score_1,
            "max_iels_per_100_epithelial_cells": max_score_2,
            "med5_max_nr_iels_per_unit_dysplasia_area": med5_max_score_1,
            "med5_max_iels_per_100_epithelial_cells": med5_max_score_2,
        }
    else:
        slide_score_dict = {
            "slide": wsi_name,
            "cohort": cohorts[cohort],
            "nr_iels_per_unit_epithelial_area": iels_per_dys_area_microns,
            "nr_iels_per_100_epithelial_cells": iels_per_100_epithelial_cells,
            "max_nr_iels_per_unit_epithelial_area": max_score_1,
            "max_iels_per_100_epithelial_cells": max_score_2,
            "med5_max_nr_iels_per_unit_epithelial_area": med5_max_score_1,
            "med5_max_iels_per_100_epithelial_cells": med5_max_score_2,
            }

    # If file exists, append to it, otherwise create new file
    score_path = os.path.join(output_folder, f"iel_scoring.csv")
    if os.path.exists(score_path):
        with open(score_path, "a") as f:
            writer = csv.DictWriter(f, fieldnames=slide_score_dict.keys())
            writer.writerow(slide_score_dict)
            f.close()
            print("Written to file")
    else:
        df = pd.DataFrame(data=slide_score_dict, index=[0])
        df.to_csv(score_path, index=False)
    return


### Problems regarding incorrect names for some slides. E.g.:
#235it [00:42,  5.33it/s]No hovernet output for case 13- 0536-031, skipping

if __name__ == '__main__':

    # Inputs
    num_processes = 8 #2

    # participant_file = "/data/data/ANTICIPATE/sheffield/new_clinical_data_221222/sheffield_data_c1,3,4,7_oed_5-folds.csv"
    participant_file = "/data/data/ANTICIPATE/train_test_splits/oed_data_train_test_new_noqc_corrected_dyspdet_c7_brazil.csv"

    input_wsis = [
        "/data/data/ANTICIPATE/sheffield/old_oed_wsi/tmp/", # cohort 1
        "/data/data/ANTICIPATE/sheffield/all/oed/", # cohort 3
        "/data/data/ANTICIPATE/sheffield/cohort_4/wsis/", # cohort 4
        "/data/data/ANTICIPATE/sheffield/3dhistech_oed_case_controls/Histech - cases/wsis/", # cohort 7
        "/data/data/ANTICIPATE/birmingham/wsis/", # birmingham
        "/data/data/ANTICIPATE/belfast/wsis/", # belfast
        "/data/data/ANTICIPATE/brazil/wsis/", # brazil
    ]

    input_dysplasia = [
        "/data/ANTICIPATE/dyplasia_detection/inference/dysplasia_detector_output/sheffield/cohort_1/layers/",
        "/data/ANTICIPATE/dyplasia_detection/inference/dysplasia_detector_output/sheffield/cohort_3/oed/layers/",
        "/data/ANTICIPATE/dyplasia_detection/inference/dysplasia_detector_output/sheffield/cohort_4/layers/",
        "/data/ANTICIPATE/dyplasia_detection/inference/dysplasia_detector_output/sheffield/cohort_7/oed/layers/",
        "/data/ANTICIPATE/dyplasia_detection/inference/dysplasia_detector_output/birmingham/layers/",
        "/data/ANTICIPATE/dyplasia_detection/inference/dysplasia_detector_output/belfast/layers/",
        "/data/ANTICIPATE/dyplasia_detection/inference/dysplasia_detector_output/brazil/layers/",
    ]

    input_hovernet = [
        "/data/ANTICIPATE/nuclei_segmentation/hovernetplus_segmentations/sheffield/cohort_1/",#json/",
        "/data/ANTICIPATE/nuclei_segmentation/hovernetplus_segmentations/sheffield/cohort_3/oed/",#json",
        "/data/ANTICIPATE/nuclei_segmentation/hovernetplus_segmentations/sheffield/cohort_4/",#json/",
        "/data/ANTICIPATE/nuclei_segmentation/hovernetplus_segmentations/sheffield/cohort_7/oed/",#json/",  
        "/data/ANTICIPATE/nuclei_segmentation/hovernetplus_segmentations/birmingham/",
        "/data/ANTICIPATE/nuclei_segmentation/hovernetplus_segmentations/belfast/",
        "/data/ANTICIPATE/nuclei_segmentation/hovernetplus_segmentations/brazil/",      
    ]

    color_dict_dysp = {
        "bkgd": [0, (0, 0, 0)],
        "dysplasia": [1, (255, 0, 0)],
    }
    color_dict_hover = {
        "bkgd": [0, (0, 0, 0)],
        "other": [1, (255, 155, 0)],
        "basal": [2, (255, 0, 0)],
        "epith": [3, (0, 255, 0)],
        "keratin": [4, (0, 0, 255)],
    }
    cohorts = {
        0: 1,
        1: 3,
        2: 4,
        3: 7,
        4: 5,
        5: 6,
        6: 7,
    }
    external_dict = {
        5: "birmingham",
        6: "belfast",
        7: "brazil",
    }
    tissue = "dysplasia" #"epithelium" #"epithelium" #"dysplasia"
   
    # Outputs
    # output_folder = f"/data/ANTICIPATE/outcome_prediction/digital_scoring/output2/iels_2/{tissue}_{PATCH_MPP}mpp_{PATCH_SIZE}_{PATCH_STRIDE}/"
    output_folder = f"/data/ANTICIPATE/outcome_prediction/digital_scoring/output2/iels_external2/{tissue}_{PATCH_MPP}mpp_{PATCH_SIZE}_{PATCH_STRIDE}/"

    # Make relevant directories
    os.makedirs(output_folder, exist_ok=True)
    img_out_dir = os.path.join(output_folder, "viz_iels")
    os.makedirs(img_out_dir, exist_ok=True)

    # First loop over WSIs in folder
    # for cohort in range(0, len(input_wsis)):
    for cohort in [4]:#range(1, len(input_wsis)):
        if cohort <= 3:
            img_out_coh_dir = os.path.join(output_folder, "viz_iels", "cohort_" + str(cohorts[cohort]))
            patch_level_csv_dir = os.path.join(output_folder, "patch_level_iels", "csvs", "cohort_" + str(cohorts[cohort]))
            patch_level_1_dir = os.path.join(output_folder, "patch_level_iels", "score_1", "cohort_" + str(cohorts[cohort]))
            patch_level_2_dir = os.path.join(output_folder, "patch_level_iels", "score_2", "cohort_" + str(cohorts[cohort]))
        else:
            img_out_coh_dir = os.path.join(output_folder, "viz_iels", str(external_dict[cohorts[cohort]]))
            patch_level_csv_dir = os.path.join(output_folder, "patch_level_iels", "csvs", str(external_dict[cohorts[cohort]]))
            patch_level_1_dir = os.path.join(output_folder, "patch_level_iels", "score_1", str(external_dict[cohorts[cohort]]))
            patch_level_2_dir = os.path.join(output_folder, "patch_level_iels", "score_2", str(external_dict[cohorts[cohort]]))      
        os.makedirs(img_out_coh_dir, exist_ok=True)
        os.makedirs(patch_level_csv_dir, exist_ok=True)
        os.makedirs(patch_level_1_dir, exist_ok=True)
        os.makedirs(patch_level_2_dir, exist_ok=True)

        file_list = sorted(glob.glob(input_wsis[cohort] + "*.*"))

        pbar_format = "Processing cases... |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        pbarx = tqdm(
            total=len(file_list), bar_format=pbar_format, ascii=True, position=0
        )

        # Start multi-processing
        argument_list = file_list
        num_jobs = len(argument_list)

        pool = Pool(processes=num_processes, initargs=(RLock(),), initializer=tqdm.set_lock)

        def pbarx_update(*a):
            pbarx.update()

        jobs = [pool.apply_async(process, args=(n, tissue, cohort, input_dysplasia, input_hovernet, color_dict_dysp, color_dict_hover, cohorts, output_folder, img_out_coh_dir, patch_level_csv_dir, patch_level_1_dir, patch_level_2_dir), callback=pbarx_update) for n in argument_list]
        pool.close()
        result_list = [job.get() for job in jobs]
        pbarx.close()