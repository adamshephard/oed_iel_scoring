"""
Script to convert HoVer-Net+ nuclei DAT file outputs to annotation stores (DB).
"""

import numpy as np
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.utils.misc import store_from_dat
import glob
import os


def dat2db(input_wsi_path, input_dat_file, output_db_file, proc_res=0.5):
    """
    Convert DAT file to annotation store (DB).
    """
    # Get scale factor for annotation store
    wsi = WSIReader.open(input_wsi_path)
    base_dims = wsi.slide_dimensions(units='level', resolution=0)
    proc_dims = wsi.slide_dimensions(units='mpp', resolution=proc_res)

    scale_factor = np.asarray(base_dims) / np.asarray(proc_dims)

    # Convert DAT file to annotation store
    annos_db = store_from_dat(input_dat_file, scale_factor=scale_factor)

    # Save annotation store to DB
    annos_db.dump(output_db_file)
    return


if __name__ == "__main__":

    input_wsi_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/wsis_3/"
    input_dat_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/output_epith4/nuclei/"
    output_db_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/output_epith4/nuclei/"
    
    for input_wsi_path in glob.glob(input_wsi_dir + "*.*"):
        basename = os.path.basename(input_wsi_path).split(".")[0]
        input_dat_file = os.path.join(input_dat_dir, basename + ".dat")
        output_db_file = os.path.join(output_db_dir, basename + ".db")
        print(f"Processing {basename}")
        dat2db(input_wsi_path, input_dat_file, output_db_file)
