import pandas as pd
import json
import numpy as np
import cv2
from collections import OrderedDict
from skimage.color import rgb2hed
from tqdm import tqdm
from matplotlib import pyplot as plt
from tiatoolbox.utils.transforms import rgb2od


def get_cellularity(wsi, mask, nuc_path, mask_classes=[1], nr_classes=4):
    """
    Get cellularity of nuclei in mask.
    """
    with open(nuc_path) as f:
        jsdata = json.load(f)
    proc_mpp = jsdata["mpp"]
    proc_dims = wsi.slide_dimensions(resolution=proc_mpp, units="mpp")
    mask_dims = mask.shape[::-1]
    base_dims = wsi.slide_dimensions(resolution=0, units="level")
    base_ds = base_dims/proc_dims
    proc_ds = proc_dims/mask_dims
    # nuc_types = {}
    nuc_types = {str(i+1):0 for i in range(nr_classes)}
    # for n in tqdm(jsdata["nuc"], desc="Processing Nuclei"):
    for n in jsdata["nuc"]:
        nuc = jsdata["nuc"][n]
        cX, cY = nuc["centroid"]
        nuc_type = nuc["type"]
        if nuc_type == 0:
            continue
        if mask[int(cY/proc_ds[1]), int(cX/proc_ds[0])] in mask_classes:#!= 0:
            # if str(nuc_type) not in nuc_types:
            #     nuc_types[str(nuc_type)] = 1
            # else:
            #     nuc_types[str(nuc_type)] += 1
            nuc_types[str(nuc_type)] += 1
    mask_area = np.sum(np.isin(mask, mask_classes))
    mask_area_proc = mask_area * proc_ds[0] * proc_ds[1]
    mask_area_microns = mask_area_proc * proc_mpp * proc_mpp
    # mask_area_mm = mask_area_proc * proc_mpp * proc_mpp / 10**6
    cellularity = {k: v/mask_area_microns for k,v in nuc_types.items()}
    # cellularity = {k: v/mask_area_mm for k,v in nuc_types.items()}
    # nuc_types_2 = {
    #     '1': nuc_types['1'],
    #     '2': np.sum([nuc_types['2'], nuc_types['3'], nuc_types['4']]),
    # }
    # cellularity_2 = {k: v/mask_area_microns for k,v in nuc_types_2.items()}
    return cellularity

def hematoxylin_od(image, mask):
    """
    Get Hematoxylin OD of nuclei in mask.
    """
    hed = rgb2hed(image)
    hed = (hed*255).astype('int')
    hed_od = rgb2od(hed)
    mask_inv = mask.copy()
    mask_inv[mask_inv == 0] = 2
    mask_inv[mask_inv == 1] = 0
    mask_inv[mask_inv == 2] = 1
    h_od_avg = np.ma.array(hed_od[...,0], mask=mask_inv).mean()    
    return h_od_avg 
    
def get_nuc_features(wsi, ds, inst_dict):
    inst_cnt = np.array(inst_dict['contour']).astype(np.float32)
    inst_box = np.array([
        [np.min(inst_cnt[:,1]), np.min(inst_cnt[:,0])],
        [np.max(inst_cnt[:,1]), np.max(inst_cnt[:,0])]
    ])
    bbox_h, bbox_w = inst_box[1] - inst_box[0]
    bbox_aspect_ratio = float(bbox_w / bbox_h)
    bbox_area = bbox_h * bbox_w
    contour_area = cv2.contourArea(inst_cnt)
    extent = float(contour_area) / bbox_area
    convex_hull = cv2.convexHull(inst_cnt)
    convex_area = cv2.contourArea(convex_hull)
    convex_area = convex_area if convex_area != 0 else 1
    solidity = float(contour_area) / convex_area
    equiv_diameter = np.sqrt(4 * contour_area / np.pi)
    start_y, start_x = (inst_box[0]*np.asarray(ds)).astype('int')
    end_y, end_x = (inst_box[1]*np.asarray(ds)).astype('int')
    image = wsi.read_region((start_x, start_y), 0, (end_x-start_x, end_y-start_y))
    # plt.imshow(image)
    cntr = inst_cnt.copy()
    cntr[:,0] = (cntr[:,0]*ds[0]).astype('int') - start_x 
    cntr[:,1] = (cntr[:,1]*ds[1]).astype('int') - start_y
    cntr = cntr.astype('int')
    mask = cv2.fillPoly(np.zeros((end_y-start_y, end_x-start_x)), [cntr], 1)
    # mask = cv2.drawContours(image, [cntr], 1)
    hem_od = hematoxylin_od(image, mask)
    if inst_cnt.shape[0] > 4:
        _, axes, orientation = cv2.fitEllipse(inst_cnt)
        major_axis_length = max(axes)
        minor_axis_length = min(axes)
    else:
        orientation = 0
        major_axis_length = 1
        minor_axis_length = 1
    perimeter = cv2.arcLength(inst_cnt, True)
    _, radius = cv2.minEnclosingCircle(inst_cnt)
    eccentricity = np.sqrt(1- (minor_axis_length / major_axis_length)**2)
    circularity = (4 * np.pi * radius**2) / (perimeter**2)
    stat_dict = OrderedDict()
    stat_dict['type'] = inst_dict['type']
    stat_dict['hematoxylin_od'] = hem_od
    stat_dict['eccentricity'] = eccentricity
    stat_dict['circularity'] = circularity
    stat_dict['convex_area'] = convex_area
    stat_dict['contour_area'] = contour_area
    stat_dict['equiv_diameter'] = equiv_diameter
    stat_dict['extent'] = extent
    stat_dict['major_axis_length'] = major_axis_length
    stat_dict['minor_axis_length'] = minor_axis_length
    stat_dict['perimeter'] = perimeter
    stat_dict['solidity'] = solidity
    stat_dict['orientation'] = orientation
    stat_dict['radius'] = radius
    stat_dict['bbox_area'] = bbox_area
    stat_dict['bbox_aspect_ratio'] = bbox_aspect_ratio
    return stat_dict

def get_morph_features(wsi, mask, hovernetplus_nuc_path):#, nr_classes=2):
    """
    Get morphological features of nuclei in mask. Ignores nuclear class
    """
    with open(hovernetplus_nuc_path) as f:
        jsdata = json.load(f)
    proc_mpp = jsdata["mpp"]
    proc_dims = wsi.slide_dimensions(resolution=proc_mpp, units="mpp")
    base_dims = wsi.slide_dimensions(resolution=0, units="level")
    mask_dims = mask.shape[::-1]
    # ds = (base_dims/proc_dims)/mask_dims
    ds = proc_dims/mask_dims
    base_ds = (base_dims/proc_dims)
    nuc_features_all = []
    for n in jsdata["nuc"]:
        nuc = jsdata["nuc"][n]
        cX, cY = nuc["centroid"]
        # nuc_type = nuc["type"]
        if mask[int(cY/ds[1]), int(cX/ds[0])] == 0:
            continue
        nuc_features = get_nuc_features(wsi, base_ds, nuc)
        nuc_features_all.append(nuc_features)
    nuc_features_all = pd.DataFrame.from_records(nuc_features_all)
    return nuc_features_all
