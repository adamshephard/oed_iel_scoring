[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-orange.svg)](https://www.gnu.org/licenses/gpl-3.0)
  <a href="#cite-this-repository"><img src="https://img.shields.io/badge/Cite%20this%20repository-BibTeX-brightgreen" alt="DOI"></a> <a href="https://doi.org/10.1101/2024.03.27.24304967"><img src="https://img.shields.io/badge/DOI-10.1101%2F2024.03.27.24304967-blue" alt="DOI"></a>
<br>


# Intra-Epithelial Lymphocyte (IEL) Scoring in Oral Epithelial Dysplasia

This repository provides the code for the models used to generate intra-epithelial lymphocyte (IEL) scores in OED, based on H&E-stained whole-slide images. Link to journal article [here](https://www.nature.com/articles/s41416-024-02916-z).

The first step in this pipeline is to use HoVer-Net+ (see original paper [here](https://openaccess.thecvf.com/content/ICCV2021W/CDPath/html/Shephard_Simultaneous_Nuclear_Instance_and_Layer_Segmentation_in_Oral_Epithelial_Dysplasia_ICCVW_2021_paper.html)) to segment the epithelium and nuclei. We have used the TIAToolbox (see paper [here](https://www.nature.com/articles/s43856-022-00186-5)) implementation of HoVer-Net+ in the below scripts. Next, we have used a Transformer-based model to segment the dyplastic regions of the WSIs (see paper [here](https://arxiv.org/abs/2311.05452)).

Following this, we count the number of IELs in the combined dysplasia-epithelium mask using various different IEL scores, as per the paper:

1. IEL Index (II) – the number of IELs per unit area of dysplasia, within the entire dysplastic region of the WSI
2. IEL Peak Index (IPI) – the maximum number of IELs per unit area of dysplasia in any given area of dysplasia (here, chosen to be a patch of size 512 x 512, at 1.0 mpp resolution)
3. IEL Count (IC) – the number of IELs per 100 dysplastic epithelial cells, in any given area of dysplasia (here, chosen to be a patch of size 512 x 512, at 1.0 mpp resolution)
4. IEL Peak Count (IPC) – the maximum number of IELs per 100 dysplastic epithelial cells, within the entire dysplastic region of the WSI In Figure 1, we provide an overview of the proposed analytical pipeline used to generate our IEL scores.

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
- `combined_masks.py`: combine dysplasia-epithelium masks script
- `iel_scoring.py`: IEL scoring script

## Inference

### Data Format
Input: <br />
- WSIs supported by [OpenSlide](https://openslide.org/), including `svs`, `tif`, `ndpi` and `mrxs`.

Output: <br />
- HoVer-Net nuclei and epithelium segmentations as `dat` and `png` files, respectively. These segmentations are saved at 0.5 mpp resolution. Nuclei `dat` files have a key as the ID for each nucleus, which then contain a dictionary with the keys:
  - 'box': bounding box coordinates for each nucleus
  - 'centroid': centroid coordinates for each nucleus
  - 'contour': contour coordinates for each nucleus 
  - 'prob': per class probabilities for each nucleus
  - 'type': prediction of category for each nucleus
- Transformer dysplasia segmentations as `png` files. These segmentations are saved at 1 mpp resolution.
- IEL scores csv file
- IEL scores visualisation per slide (visualisaton at 2 mpp resolution).

### Model Weights

We use the following weights in this work. If any of the models or checkpoints are used, please ensure to cite the corresponding paper.

- The Transformer model weights (for dyplasia segmentation) obtained from training on the Sheffield OED dataset: [OED Transformer checkpoint](https://drive.google.com/file/d/1EF3ItKmYhtdOy5aV9CJZ0a-g03LDaVy4/view?usp=sharing). Note, these weights are obtained from this [paper](https://arxiv.org/abs/2311.05452).
- The HoVer-Net+ model weights (for epithelium segmentation) obtained from training on the Sheffield OED dataset: [OED HoVer-Net+ checkpoint](https://drive.google.com/file/d/1D2OQhHv-5e9ncRfjv2QM8HE7PAWoS79h/view?usp=sharing). Note, these weights are updated compared to TIAToolbox's and are those obtained in this [paper](https://arxiv.org/abs/2307.03757).

### Usage

#### Dysplasia Segmentation with Transformer

The first stage is to run the Transformer-based model on the WSIs to generate dysplasia segmentations. This is relatively fast and is run at 1.0mpp. Note, the `model_checkpoint` is the path to the Transformer segmentation weights available to download from above.

Usage: <br />
```
  python dysplasia_segmentation.py --input_dir="/path/to/input/slides/or/images/dir/" --output_dir="/path/to/transformer/output/dir/" --model_checkpoint="/path/to/transformer/checkpoint/"
```
#### Epithelium Segmentation with HoVer-Net+

The second stage is to run HoVer-Net+ on the WSIs to generate epithelial and nuclei segmentations. This can be quite slow as run at 0.5mpp. Note, the `model_checkpoint` is the path to the HoVer-Net+ segmentation weights available to download from above. However, if none are provided then the default version of HoVer-Net+ used with TIAToolbox, will be used.

Usage: <br />
```
  python epithelium_segmentation.py --input_dir="/path/to/input/slides/or/images/dir/" --output_dir="/path/to/epithelium/output/dir/" --model_checkpoint="/path/to/hovernetplus/checkpoint/"
```

#### Combined Masks

Next, we combine the epithelium mask generated by HoVer-Net+ with the dysplasia map generated by the Transformer. This is to ensure we have conservative dysplastic epithelium masks.

Usage: <br />
```
  python combine_masks.py --input_epith="/path/to/input/epithelium/segmentation/dir/" --input_dysplasia= "/path/to/input/dysplasia/segmentation/dir/" --output_dir="/path/to/combined/mask/output/dir/"
```

#### IEL Scoring

The next stage is to generate IEL scores for each WSI using the nuclei/layer segmentations. Note the `input_mask_dir` is the output directory from the previous step.

Usage: <br />
```
  python iel_scoring.py --input_wsi_dir="/path/to/input/slides/or/images/dir/" --input_mask_dir="/path/to/combined/mask/dir/" --output_dir="/path/to/output/feature/dir/"
```

## Interactive Demo

We have made an interactive demo to help visualise the output of our model. Note, this is not optimised for mobile phones and tablets. The demo was built using the TIAToolbox [tile server](https://tia-toolbox.readthedocs.io/en/latest/_autosummary/tiatoolbox.visualization.tileserver.TileServer.html).

Check out the demo [here](https://tiademos.dcs.warwick.ac.uk/bokeh_app?demo=oed_iels). 

In the demo, we provide multiple examples of WSI-level results. These include:
- Dysplasia segmentations (using the Transformer model). Here, dysplasia is in red.
- Intra-epithelial layer segmentation (using HoVer-Net+). Here, orange is stroma, red is the basal layer, green the (core) epithelial layer, and blue keratin.
- Combined mask (from the above two segmentations). Here, orange is stroma, red is dysplasia, and green is non-dysplastic epithelium.
- Nuclei segmentations (using HoVer-Net+). Here, orange is "other" nuclei (i.e. connective/inflammatory), whilst the epithelial nuclei are coloured green).
- Masked nuclei segmentations. Here, orange is IEL nuclei, and green is epithelial nuclei.

Each histological object can be toggled on/off by clicking the appropriate buton on the right hand side. Also, the colours and the opacity can be altered.

https://github.com/adamshephard/oed_iel_scoring/assets/39619155/4ae9a645-8a7b-4675-b14e-62d62f891750

## License

Code is under a GPL-3.0 license. See the [LICENSE](https://github.com/adamshephard/oed_iel_scoring/blob/main/LICENSE) file for further details.

Model weights are licensed under [Attribution-NonCommercial-ShareAlike 4.0 International](http://creativecommons.org/licenses/by-nc-sa/4.0/). Please consider the implications of using the weights under this license. 

## Cite this repository
```
@article{Shephard2024IELs,
	author = {Adam J Shephard and Hanya Mahmood and Shan E Ahmed Raza and Syed Ali Khurram and Nasir M Rajpoot},
	title = {A Novel AI-based Score for Assessing the Prognostic Value of Intra-Epithelial Lymphocytes in Oral Epithelial Dysplasia},
	year = {2024},
	doi = {10.1101/2024.03.27.24304967},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2024/03/28/2024.03.27.24304967},
	journal = {medRxiv}
}
```
