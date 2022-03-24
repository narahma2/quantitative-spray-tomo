# quantitative-spray-4D
This repo contains the scripts required for tomographic reconstructions of the liquid spray datasets showcased in the manuscript entitled "Quantitative polychromatic 4D and monochromatic 2D X-ray tomography of liquid jet breakup dynamics."

| <img width="60%" src="images/reconstruction_comparison.png"> | 
|:--:| 
| *Slice-based comparisons of tomographic reconstructions of an impinging jet spray using a temporally-resolved polychromatic beam setup (PB-4D), temporally averaged polychromatic beam setup (PB-3D), and a temporally averaged monochromatic beam setup (MB-2D). For further detail, refer to the accompanying journal publication.* |

## Project setup
The analyses shown here walk the user through how to obtain quantitative temporally resolved volumetric reconstructions (4D) of liquid mass distribution in an impinging jet spray using a full-view cone beam tube source radiography imaging experimental setup (labelled as polychromatic beam, **PB**) and quantitative temporally averaged planar reconstructions (2D) of liquid mass distributions from the same spray using a focused parallel beam synchrotron source scanning experimental setup (labelled as monochromatic beam, **MB**). A small subset of the experimental data is included in the `inputs` folder to serve as a showcase for how the analyses are carried out—all raw datasets collected in this work are archived separately (DOI: [10.4231/G20X-4Z27](https://doi.org/10.4231/G20X-4Z27)).

The workflow in this project is split between high-level Jupyter notebooks (\*.ipynb) that serve as a detailed guide on how the analysis is done on the relevant datasets and low-level Python scripts (\*.py) that do the bulk of the actual computations. The entire repo can be downloaded to be run locally, however pre-computed static renders through [nbviewer.org](https://nbviewer.org) and interactive notebooks through [mybinder.org](https://mybinder.org) (may take a while to load!) are available below through the clickable badges.

### Monochromatic Beam (MB) processing
The workflow for quantitative tomographic reconstructions of the MB datasets is fully contained within the one notebook. The datasets for this workflow are the `*monochromatic*` files in the `inputs` folder.

| File | Description | Static | Interactive |
| ---  | --          | ---    | ---         |
| [mb_processing.ipynb](./mb_processing.ipynb) | Full processing notebook for the MB datasets. | [![nbviewer](./images/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/narahma2/quantitative-spray-tomo/blob/main/mb_processing.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/narahma2/quantitative-spray-tomo/main?filepath=mb_processing.ipynb) |
| [mb_utilities.py](./mb_utilities.py) | Python functions used in the MB notebook. | - | - |

### Polychromatic Beam (PB) processing
The workflow for quantitative tomographic reconstructions of the PB datasets is split between multiple notebooks that are to be run in the order below. The datasets for this workflow are the `*polychromatic*` files in the `inputs` folder. Due to the requirement of a GPU for 3D reconstruction, only a previously run static view is shown for the refinement step—the interactive view for the reconstruction notebook loads in previously computed results purely for visualization.

| File | Description | Static | Interactive |
| ---  | --          | ---    | ---         |
| [pb01_calibration.ipynb](./pb01_calibration.ipynb) | Initial camera calibration for each line of sight using dot targets. | [![nbviewer](./images/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/narahma2/quantitative-spray-tomo/blob/main/pb01_calibration.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/narahma2/quantitative-spray-tomo/main?filepath=pb01_calibration.ipynb) |
| [pb02_preprocessing.ipynb](./pb02_preprocessing.ipynb) | Conversion of raw X-ray radiographs into projected density maps. | [![nbviewer](./images/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/narahma2/quantitative-spray-tomo/blob/main/pb02_preprocessing.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/narahma2/quantitative-spray-tomo/main?filepath=pb02_preprocessing.ipynb) |
| [pb03_refinement.ipynb](./pb03_refinement.ipynb) | Tomography-based refinement of calibration parameters. | [![nbviewer](./images/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/narahma2/quantitative-spray-tomo/blob/main/pb03_refinement.ipynb) | - |
| [pb04_reconstruction.ipynb](./pb04_reconstruction.ipynb) | Reconstruction of the multi-view projections into volumes. | [![nbviewer](./images/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/narahma2/quantitative-spray-tomo/blob/main/pb04_reconstruction.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/narahma2/quantitative-spray-tomo/main?filepath=pb04_reconstruction.ipynb) |
| [calib_utilities.py](./calib_utilities.py) | Python functions for PB camera calibration. | - | - |
| [preproc_utilities.py](./preproc_utilities.py) | Python functions for PB image pre-processing. | - | - |
| [refine_utilities.py](./refine_utilities.py) | Python functions for PB calibration refinement. | - | - |
| [recon_utilities.py](./recon_utilities.py) | Python functions for PB volume reconstruction. | - | - |

## Requirements
The Python packages used and version information for each are detailed within the Jupyter notebooks. The cone beam reconstructions implemented through the ASTRA toolbox (https://www.astra-toolbox.com) require the use of an NVIDIA GPU. Due to the use of an out-dated GPU (see [Hardware information](#hardware-information)), the PB reconstruction scripts utilized v1.8.3 of the ASTRA toolbox, however the MB reconstruction script requires v >= 2.0.0 of the ASTRA toolbox to use the newer rotation offset functionality and was therefore run on the CPU instead in a separate Python environment. Depending on your hardware configuration, you may be able to process all notebooks/scripts without need of separate environments.
