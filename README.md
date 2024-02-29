# Sundial
Workflow / workstation for machine learning with data sourced from Google Earth Engine on a bare metal machine.

## Introduction

Google Earth Engine provides an API endpoint which allows you to download chips up to 48 MB +- a few MB. This is relatively small as far as spatial science goes but for machine learning it is an ideal size for training with image chips. This repo contains scripts to download chips via polygons sourced from shapefiles and run them through a neural network of your own choosing. Although google already provides powerful tools for deep learning integrated with GEE, it's nice to do things for little cost on your own machine if you already own a GPU.

## Getting Started

### Download and Install packages and dependencies

- Create a conda environment from provided environment file or install as you go. Linux 64bit is required!!

```
conda env create -f environment.yml -n sundial
```

### Authenticate on Google Earth Engine

- [Earth Engine Authentication and Initialization](https://developers.google.com/earth-engine/guides/auth)

```
earthengine authenticate
```

## Usage Examples

- Getting help for commands.

```Shell
make

Welcome to Sundial!

        To run Sundial, use the following commands:

    Sundial Methods:
        sample:      Retrieves chip sample from Google Earth Engine.                                        'Alpha'
        train:       Train Sundial model with optional config.                                              'Alpha'
        validate:    Validate Sundial model with optional config.                                           'Alpha'
        test:        Test Sundial model with optional config.                                               'Alpha'
        predict:     Predict using Sundial. Must provide image path in SUNDIAL_CONFIG.                      'Alpha'
        nuke:        Removes all data from run. Must provide sample name in env var SUNDIAL_SAMPLE_NAME.    'Alpha'

    Variables:
        SUNDIAL_BASE_PATH:           Base path for Sundial scripts. Default: 'shell pwd' of this file
        SUNDIAL_SAMPLE_NAME:         Sample name. Default: 'blm_or_wa_bounds'
        SUNDIAL_EXPERIMENT_SUFFIX:   Sundial experiment name to be appened to sample name. Default: ''
        SUNDIAL_ENV_NAME:            Sundial environment name. Default: 'sundial'
        SUNDIAL_PROCESSING:          Sundial processing method. Default: 'hpc'
        SUNDIAL_CONFIG:              Sundial config file path. Default: 'src/settings.py'
        SUNDIAL_CKPT_PATH:           Sundial checkpoint path. Default: 'null'
```

- Sampling using a shapefile. The scripts will read the file name stored in data/shapes and generate train, validate,test,predict splits as well as metadata files in zarr format.
```Shell
make SUNDIAL_PROCESSING=local SUNDIAL_CONFIG=./configs/bug_ads.sample.yaml SUNDIAL_SAMPLE_NAME=ads_damage_1990-2022 sample
```

- Downloading using the generated chip samples. Download limits apply so use your own discretion. I found the limit to be chips of shape (256 bands, 256 pixels, 256 pixels) to be the the upper limit but this may vary depending on processing, scale, etc.
```Shell
make SUNDIAL_PROCESSING=local SUNDIAL_CONFIG=./configs/bug_ads.download.yaml SUNDIAL_SAMPLE_NAME=ads_damage_1990-2022 download
```

- Train a model using the provided chips. Included is simple segmentation model built on Prithvi's foundation model using pytorch lightning. Using the CLI, you can mix and match models. See their [docs](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) for more details on config files. For now the runner will import models by name from the models module. Write additional models there or use one that is prebuilt in lightning. Backbones are also stored in the backbones directory in src.
```Shell
make SUNDIAL_PROCESSING=local SUNDIAL_CONFIG=./configs/bug_ads.run.yaml SUNDIAL_SAMPLE_NAME=ads_damage_1990-2022 fit
```

## Config files

While the framework for deep learning is done via [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html), the pipeline down from GEE can be configured using a similar format. See src/pipeline/settings.py for more details. In addition, images are generated using an annual medoid composite since for now, that is our focus, but another image generator function can be provided to the class if you instantiate it separately.

The configs worth looking at are:
  - SAMPLER.method: Method to generate samples options: ["convering_grid" | "random" | "stratified" | "single"]
  - SAMPLER.num_points: Number of points to sample from original shapefile
  - SAMPLER.num_strata: Number of strata to generate from statistics
  - SAMPLER.meter_edge_size: Edge size of squares in meters
  - SAMPLER.strata_columns: Columns found in shapefile to generate strata for training

  - DOWNLOADER.file_type: File type to download ["NPY" | "NUMPY_NDARRAY" | "ZARR" | "GEO_TIFF"]
  - DOWNLOADER.scale: Scale to generate image
  - DOWNLOADER.reprojection: Reprojection string (EPSG:****)
  - DOWNLOADER.overlap_band: Wether to include an additional band that notes wether the pixel in the generated square overlaps the original polygon
  - DOWNLOADER.pixel_edge_size: Edge size of chip image in pixels
