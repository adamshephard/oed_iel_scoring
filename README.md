# Intra-Epithelial Lymphocyte (IEL) Scoring for Predicting Oral Epithelial Dysplasia Malignant Transformation

This repository provides the code for the models used for predicting slide-level malignancy transformation in OED, based on H&E-stained whole-slide images. Link to preprint [here](n/a).

The first step in this pipeline is to use HoVer-Net+ (see original paper [here](https://openaccess.thecvf.com/content/ICCV2021W/CDPath/html/Shephard_Simultaneous_Nuclear_Instance_and_Layer_Segmentation_in_Oral_Epithelial_Dysplasia_ICCVW_2021_paper.html)) to segment the epithelium and nuclei. We have used the TIAtoolbox (see paper [here](https://www.nature.com/articles/s43856-022-00186-5)) implementation of HoVer-Net+ in the below scripts. Next, we have used a Transformer-based model to segment the dyplastic regions of the WSIs (see paper here).

## Set Up Environment

We use Python 3.11 with the [tiatoolbox](https://github.com/TissueImageAnalytics/tiatoolbox) package installed. By default this uses PyTorch 2.2.

```
conda create -n odyn python=3.11 cudatoolkit=11.8
conda activate odyn
pip install tiatoolbox
pip uninstall torch
conda install pytorch
pip install h5py
pip install docopt
pip install ml-collections
```

## Repository Structure

Below are the main directories in the repository: 

- `utils/`: scripts for metric, patch generation
- `models/`: model definition

Below are the main executable scripts in the repository:

- `dysplasia_segmentation.py`: transformer inference script
- `epithelium_segmentation.py`: hovernetplus inference script
- `iel_scoring.py`: IEL scoring script
- `viz_iels.py`: Visualise IEL/nuclei and dysplasi regions in cases.
