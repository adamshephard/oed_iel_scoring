import numpy as np
import cv2
from shapely.geometry import shape, Point, Polygon, box, MultiPolygon


def get_mask_polygons(wsi, mask, resolution, units="mpp"):
    """
    Get mask polygons from mask. Mask is HW numpy array.
    Returns:
        geometries: list of shapely polygons
    """
    mask_dims = mask.shape[::-1]
    out_dims = wsi.slide_dimensions(resolution=resolution, units=units)
    
    # Get contours, including holes
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = np.squeeze(hierarchy)

    # Scale down the contours
    mask_scale_factor = np.asarray(out_dims) / np.asarray(mask_dims)
    
    # outer_polygons = []
    # # Scale up the contours and create Shapely polygons
    # contours_2 = list(contours)
    # for contour in contours:
    #     contour = contour.squeeze() * mask_scale_factor
    #     contour = contour.tolist() 
    #     if len(contour) < 4:
    #         continue  # Skip polygons with fewer than 4 vertices
    #     contour = [(x, y) for x, y in contour]  # Convert array elements to tuples
    #     outer_polygon = Polygon(contour)
    #     for i, inner_c in enumerate(contours_2):
    #         inner_contour = inner_c.squeeze() * mask_scale_factor
    #         inner_contour = inner_contour.tolist() 
    #         if len(inner_contour) < 4:
    #             contours_2.pop(i)
    #             continue  # Skip polygons with fewer than 4 vertices 
    #         inner_contour = [(x, y) for x, y in inner_contour]
    #         if inner_contour != contour:
    #             hole_polygon = Polygon(inner_contour)
    #             if outer_polygon.contains(hole_polygon): # create holes
    #                 outer_polygon = outer_polygon.difference(hole_polygon)
    #                 contours_2.pop(i)
    #         else:
    #             contours_2.pop(i)
    #     outer_polygons.append(outer_polygon)
    
    # Define function to merge individual contours into one MultiPolygon
    def merge_polygons(polygon:MultiPolygon,idx:int,add:bool) -> MultiPolygon:
        """
        polygon: Main polygon to which a new polygon is added
        idx: Index of contour
        add: If this contour should be added (True) or subtracted (False)
        """
        # Get contour from global list of contours
        contour = np.squeeze(contours[idx]) * mask_scale_factor
        # cv2.findContours() sometimes returns a single point -> skip this case
        if len(contour) > 2:
            # Convert contour to shapely polygon
            new_poly = Polygon(contour)
            # Not all polygons are shapely-valid (self intersection, etc.)
            if not new_poly.is_valid:
                # Convert invalid polygon to valid
                new_poly = new_poly.buffer(0)
            # Merge new polygon with the main one
            if add: polygon = polygon.union(new_poly)
            else:   polygon = polygon.difference(new_poly)
        # Check if current polygon has a child
        child_idx = hierarchy[idx][2]
        if child_idx >= 0:
            # Call this function recursively, negate `add` parameter
            polygon = merge_polygons(polygon,child_idx,not add)
        # Check if there is some next polygon at the same hierarchy level
        next_idx = hierarchy[idx][0]
        if next_idx >= 0:
            # Call this function recursively
            polygon = merge_polygons(polygon,next_idx,add)
        return polygon

    # Call the function with an initial empty polygon and start from contour 0
    outer_polygons = merge_polygons(MultiPolygon(),0,True)


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