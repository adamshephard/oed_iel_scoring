import numpy as np
import cv2

def decolourise(image, col_dict):
    image_col = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')
    for _, col in col_dict.items():
        image_col[np.where(np.all(image == col[1], axis=-1))] = col[0]
    return image_col

def colourise(image, col_dict):
    image_col = np.zeros((image.shape[0], image.shape[1], 3), dtype='uint8')
    for _, col in col_dict.items():
        try:
            image_col[image==col[0]] = col[1]
        except:
            continue
    return image_col

def draw_layers(roi, layer_map, layer_ds, write_ds, colour_dict, bounds=None, thickness=7, exclude_tissue=True):
    overlay = roi.copy()
    if bounds is not None:
        layers = layer_map[int(bounds[1]/layer_ds[1]):int(bounds[3]/layer_ds[1]), int(bounds[0]/layer_ds[0]):int(bounds[2]/layer_ds[0])]
    else:
        layers = layer_map
    for layer_id, colour in colour_dict.items():
        if exclude_tissue:
            if layer_id in [1]:
                continue # makes viz easier
        if layer_id == 0:
            continue # makes viz easier
        layer = (layers == layer_id).astype('uint8')
        contours, _ = cv2.findContours(layer, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) > 3:
                contour_adj = (np.squeeze(contour) * layer_ds / write_ds).astype('int')
                overlay = cv2.drawContours(overlay, [contour_adj], -1, colour, thickness=thickness)
    return overlay
