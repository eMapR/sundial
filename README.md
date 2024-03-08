# Sundial
Workflow / workstation for machine learning with data sourced from Google Earth Engine on a bare metal machine.

## Introduction

Google Earth Engine provides an API endpoint which allows you to download chips up to 48 MB +- a few MB. This is relatively small as far as spatial science goes but for machine learning it is an ideal size for training with a large number of image chips. This repo contains scripts to download chips via polygons sourced from shapefiles and run them through a neural network of your own choosing. Although google already provides powerful tools for deep learning integrated with GEE, it's nice to do things for little cost on your own machine if you already own a GPU.

## Getting Started

### Download and Install packages and dependencies

- Clone the repository and cd into it. 

```
git clone https://github.com/eMapR/sundial.git
cd sundial
```

- Create a conda environment from provided environment file or install as you go. Linux 64bit is required! WSL will work as well.

```
conda env create --prefix {PREFIX} --file environment.yaml
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

    Methods:
        sample:      Generates chip sample polygons using Google Earth Engine and provided shapefile.
        download:    Downloads chip sample images from Google Earth Engine.
        fit:         Train model using subset of data from sample and download.
        validate:    Validate model subset of using data from sample and download.
        test:        Test model using subset of data from sample and download.
        predict:     Predict and image from subset of data from sample and download.
        clean:       Removes all logs and sample data.
        nuke:        Removes all data from run.

    Variables:
        SUNDIAL_BASE_PATH:           Base path for Sundial scripts. Default: 'shell pwd' of this file
        SUNDIAL_SAMPLE_NAME:         Sample name. Default: ''
        SUNDIAL_EXPERIMENT_SUFFIX:   Sundial experiment name. Default: ''
        SUNDIAL_ENV_NAME:            Sundial environment name. Default: 'sundial'
        SUNDIAL_PROCESSING:          Sundial processing method. Default: 'hpc'
```

- Generate config files for running make commands. These will be generated with the appropriate paths for all files that need to be created / read into each script and automatically overwrite any default configs found in settings.py when used. This must be run before any other submake.
```Shell
make SUNDIAL_PROCESSING=local SUNDIAL_SAMPLE_NAME=ads_damage_1990-2022_14k config
```

- Sampling using a shapefile. The scripts will read the file with the name in variable $(SUNDIAL_SAMPLE_NAME) stored in data/shapes and generate train, validate, test, predict splits if specified, as well as metadata files in zarr format. Features in the shapefile can also contain columns which can be used to create annotations for training using the SAMPLER.strata_columns setting.
```Shell
make SUNDIAL_PROCESSING=local SUNDIAL_SAMPLE_NAME=ads_damage_1990-2022_14k sample
```

- Downloading using the generated chip samples to create GEE images. Download limits apply so use your own discretion. I found the limit to be chips of shape (256 bands, 256 pixels, 256 pixels) to be the the upper limit but this may vary depending on processing, scale, etc. The information used to download images is found in the meta data file generated from the sampler. Annotations can be saved as their own image. This may not be as storage efficient but makes for loading into Pytorch simpler for now.
```Shell
make SUNDIAL_PROCESSING=local SUNDIAL_SAMPLE_NAME=ads_damage_1990-2022_14k download
```

- Train a model using the downloaded chips. Included in this repo is a simple segmentation model using a fully convolutional network built on Prithvi's foundation model as a backbone. Using the CLI, you can mix and match models. See [Pytorch Lightning Docs](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) for more details on config files. For now the runner will import models by name from the models module. Write additional models there or use one that is prebuilt in lightning. Backbones are also stored in the backbones directory in src and can be imported in a similar way.
```Shell
make SUNDIAL_PROCESSING=local SUNDIAL_SAMPLE_NAME=ads_damage_1990-2022_14k fit
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
  - DOWNLOADER.overlap_band: Whether to include an additional band that notes if the pixel in the generated square overlaps the original polygon
  - DOWNLOADER.pixel_edge_size: Edge size of chip image in pixels
