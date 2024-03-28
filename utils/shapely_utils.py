import numpy as np
import cv2
from shapely.geometry import shape, Point, Polygon, box


def get_mask_polygons(wsi, mask, resolution, units="mpp"):
    """
    Get mask polygons from mask. Mask is HW numpy array.
    Returns:
        geometries: list of shapely polygons
    """
    mask_dims = mask.shape[::-1]
    out_dims = wsi.slide_dimensions(resolution=resolution, units=units)
    
    # Get contours, including holes
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Scale down the contours
    mask_scale_factor = np.asarray(out_dims) / np.asarray(mask_dims)
    outer_polygons = []
    # Scale up the contours and create Shapely polygons
    for contour in contours:
        contour = contour.squeeze() * mask_scale_factor
        contour = contour.tolist() 
        contour = [(x, y) for x, y in contour]  # Convert array elements to tuples
        if len(contour) < 4:
            continue  # Skip polygons with fewer than 4 vertices
        outer_polygon = Polygon(contour)
        for inner_contour in contours:
            inner_contour = inner_contour.squeeze() * mask_scale_factor
            inner_contour = inner_contour.tolist() 
            inner_contour = [(x, y) for x, y in inner_contour]
            if len(inner_contour) < 4:
                continue  # Skip polygons with fewer than 4 vertices 
            if inner_contour != contour:
                hole_polygon = Polygon(inner_contour)
                if outer_polygon.contains(hole_polygon): # create holes
                    outer_polygon = outer_polygon.difference(hole_polygon)
        outer_polygons.append(outer_polygon)

    return outer_polygons


def get_nuclear_polygons(wsi, nuc_data, resolution, units="mpp"):
    """
    Get nuclear polygons from nuc_data dict.
    Returns:
        geometries: list of shapely polygons
        inst_types: list of nuclear types
        inst_cntrs: list of nuclear contours (in pixel units)
    """
    
    inst_mpp = nuc_data["mpp"]
    inst_pred = nuc_data["nuc"]
    inst_res = wsi.convert_resolution_units(input_res=inst_mpp, input_unit="mpp", output_unit=units)
    
    proc_scale_factor = inst_res[0] / resolution
    inst_pred = list(inst_pred.values())
    inst_cntrs = [np.rint(np.asarray(v["contour"])*proc_scale_factor).astype('int') for v in inst_pred]
    geometries = [Polygon(cntr) for cntr in inst_cntrs]
    inst_types = [v["type"] for v in inst_pred]
    
    # Check and attempt to fix invalid geometries
    for i, geometry in enumerate(geometries):
        if not geometry.is_valid:
            geometries[i] = geometry.buffer(0)
    return geometries, inst_cntrs, inst_types