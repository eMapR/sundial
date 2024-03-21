# Sundial
Workflow / framework for machine learning with data sourced from Google Earth Engine on a bare metal machine. The intended use is on an HPC however, the scripts will work on any machine with an nvidia gpu.

## Introduction

Google Earth Engine provides an API endpoint which allows you to download chips up to 48 MB +- a few MB. This is relatively small as far as spatial science goes but for machine learning it is an ideal size for training with a large number of image chips. This repo contains scripts to download chips via polygons sourced from shapefiles and run them through a neural network of your own choosing. Although google already provides powerful tools for deep learning integrated with GEE, it's nice to do things for little cost on your own machine if you already own a GPU.

## Getting Started

### Download and Install packages and dependencies

- Clone the repository and cd into it. 

```Shell
git clone https://github.com/eMapR/sundial.git
cd sundial
```

- Create a conda environment from provided environment file or install using the submake below. Linux 64bit is required! WSL will work as well.

```Shell
conda env create --prefix $(CONDA_PREFIX) --file environment.yaml

or 

make setup
```

### Authenticate on Google Earth Engine

- [Earth Engine Authentication and Initialization](https://developers.google.com/earth-engine/guides/auth)

```Shell
earthengine authenticate
```

## Usage Examples

- Getting help for commands.

```console
make

Welcome to Sundial!

    Methods:

        setup_env:      Sets up Sundial directories and environment. The existing environment will be overwritten.
        setup:          Sets up Sundial experiment directories and config files using defaults found in src/settings.py.
        clink:          Copies Sundial experiment configs from source experiment to new experiment and links sample configs and data. Requires SSOURCE variable.

        sample:         Generates chip sample polygons using Google Earth Engine and provided shapefile.
        annotate:       Collects image annotations for experiment using chip polygons.
        download:       Downloads image chips for experiment using chip polygons.

        fit:            Train model using subset of data from sample.
        validate:       Validate model subset of using data from sample.
        test:           Test model using subset of data from sample.
        predict:        Predict and image from subset of data from sample.
        package:        Compresses experiment logs to tar to export. The tar file will be saved in home directory and overwrite already existing archives.
        status:         Check status of all jobs for experiment.
        vars:           Print all Sundial variables.
	    (method)_out:   Watch stdout and stderr or logs for experiment.
        (method)_err:   Print stderr from save file. Only available on HPC."
        
	clean:          Removes all logs, checkpoints, and predictions for experiment.
        clean_logs:     Removes all logs for experiment.
        clean_samp:     Removes all sample data for experiment.
        clean_dnld:     Removes all image chip data for experiment.
        clean_anno:     Removes all annotation data for experiment.
        clean_ckpt:     Removes all checkpoints for experiment.
        clean_pred:     Removes all predictions for experiment.
        clean_expt:     Removes all logs, sample data, checkpoints, and predictions for experiment.
        clean_all:      Removes all data for experiment including configs.

    Environment Variables:
    These may be set at submake (see below).

        SUNDIAL_BASE_PATH:           Base path for Sundial scripts. Default: 'shell pwd -P' of makefile
        SUNDIAL_SAMPLE_NAME:         Sample name. REQUIRED
        SUNDIAL_EXPERIMENT_PREFIX:   Sundial experiment name prefix. Default: ''
        SUNDIAL_ENV_NAME:            Sundial environment name. Default: 'sundial'
        SUNDIAL_PROCESSING:          Sundial processing method. Default: 'hpc'
        SUNDIAL_NODES:               Node within hpc to run job. Default: any node
```

### 1. Setup directories and conda environment.
This will create a conda environment name "sundial" using the environment.yaml file in the current directory. This only needs to be performed once. Base file paths will also be created. WARNING: this only works on x86_64 machines. Additional steps must be taken for arm architectures. 
```console
make setup_env
```

### 2. Generate directories and config files for running submakes.
These will be generated with the appropriate paths for all files that need to be created / read into each script per experiement. Default configs found in settings.py will be automatically overwritten if changes are made to the generated config files. Config files must be created before any other experiment submake.
```console
make \
    SUNDIAL_PROCESSING=local \
    SUNDIAL_SAMPLE_NAME=ads_damage_1990-2022 \
    SUNDIAL_EXPERIMENT_PREFIX=14k \
    setup_exp
```

### 3. Sampling using a shapefile. 
The scripts will read the file with the name in variable $(SUNDIAL_SAMPLE_NAME) stored in data/shapes and generate train, validate, test, predict index splits in npy format if specified, as well as metadata files in shp format.
```console
make \
    SUNDIAL_PROCESSING=local \
    SUNDIAL_SAMPLE_NAME=ads_damage_1990-2022 \
    SUNDIAL_EXPERIMENT_PREFIX=14k \
    sample
```

### 4. Generate annotation images from original shapefile and sample metadata.
Features in the shapefile are used to create annotations by pixel using the SAMPLER.strata_columns setting. The image shapes are (N C H W). This step is only necessary for superivsed learning. Note: Annotations are saved as their own image. This may not be as storage efficient but makes for loading into Pytorch simpler for now.
```console
make \
    SUNDIAL_PROCESSING=local \
    SUNDIAL_SAMPLE_NAME=ads_damage_1990-2022 \
    SUNDIAL_EXPERIMENT_PREFIX=14k \
    annotate
```

### 5. Downloading using the generated chip samples via GEE.
Download limits apply so use your own discretion. I found the limit to be chips of shape (256 bands, 256 pixels, 256 pixels) to be the the upper limit but this may vary depending on processing, scale, etc. The information used to download images is found in the meta data file generated from the sample submake.
```console
make \
    SUNDIAL_PROCESSING=local \
    SUNDIAL_SAMPLE_NAME=ads_damage_1990-2022 \
    SUNDIAL_EXPERIMENT_PREFIX=14k \
    download
```

### 6. Train a model using the downloaded chips.
Included in this repo is a simple segmentation model using a fully convolutional network built on Prithvi's foundation model as a backbone. Using the CLI, you can mix and match models. See [Pytorch Lightning Docs](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) for more details on config files. For now the runner will import models by name from the models module. Write additional models there or use one that is prebuilt in lightning. Backbones are also stored in the backbones directory in src and can be imported in a similar way.
```console
make \
    SUNDIAL_PROCESSING=local \
    SUNDIAL_SAMPLE_NAME=ads_damage_1990-2022 \
    SUNDIAL_EXPERIMENT_PREFIX=14k \
    fit
```

## Config files

While the framework for deep learning is done via [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html), the pipeline down from GEE can be configured using a similar format. See src/pipeline/settings.py for more details. In addition, images are generated using an annual medoid composite since for now, that is our focus, but another image generator function can be provided to the class if you instantiate it separately.

The configs worth looking at are:
| CONFIG | DESCRIPTION | OPTIONS |
| --- | --- | --- |
| SAMPLER.method | Method to generate samples options with polygon | "convering_grid", "random", "gee_stratified", "centroid" |
| SAMPLER.num_points | Number of points to sample from original shapefile | |
| SAMPLER.num_strata | Number of strata to generate from statistics | |
| SAMPLER.meter_edge_size | Edge size of squares in meters | |
| SAMPLER.strata_columns | Columns found in shapefile to generate strata for training | |
| - | - | - |
| DOWNLOADER.file_type | File type to download | "NPY", "NUMPY_NDARRAY", "ZARR", "GEO_TIFF" |
| DOWNLOADER.scale | Scale to generate image | |
| DOWNLOADER.projection | Reprojection string | (EPSG:****) |
| DOWNLOADER.pixel_edge_size | Edge size of chip image in pixels | |
