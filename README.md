# Sundial
Workflow / framework for machine learning with data sourced from Google Earth Engine on a bare metal machine.

## Introduction

Google Earth Engine provides an API endpoint which allows you to quickly download images up to 48 MB, give or take a few MB. This is relatively small as far as spatial science goes, but for machine learning, it is an ideal tool for getting a large number of training chips for deep learning research. GEE also provides easy access to a plethora of pre-processsed image collections.

This repo contains scripts to download square chips via polygons sourced from shapefiles and run them through a neural network of your own choosing. Although Google already provides powerful tools for deep learning integrated with GEE, it can be beneficial and/or cost-effective to perform analysis on your own GPU(s). The intended use is time series analysis on an HPC however, the scripts will work on any machine with Nvidia GPU(s).

## Basic Usage

```console
make

Welcome to Sundial!

    Methods:

        setup_env:      Sets up Sundial directories and environment. The existing environment will be overwritten.
        setup:          Sets up Sundial experiment directories and config files using defaults found in src/settings.py.
        clink:          Copies Sundial experiment configs from source experiment to new experiment and links sample configs and data. Requires SSOURCE variable.

        sample:         Generates chip sample polygons using Google Earth Engine and provided shapefile.
        annotate:       Collects image annotations for experiment using sample polygons.
        download:       Downloads image chips for experiment using sample polygons.
        calculate:      Calculate means and stds for experiment sample.

        fit:            Train model using subset of data from sample.
        validate:       Validate model subset of using data from sample.
        test:           Test model using subset of data from sample.
        predict:        Predict an image from subset of data from sample.
        package:        Convert predictions to GeoTIFF if not already in format and compress for download.

        status:         Check status of all jobs for user.
        vars:           Print all Sundial variables.
        (method)_err:   Print ERRORs and CRITICALs in log file and print stderr from file on HPC.

        clean:          Removes all logs, checkpoints, and predictions for experiment.
        clean_outs:     Removes all stdouts for experiment.
        clean_logs:     Removes all logs for experiment.
        clean_samp:     Removes all sample data for experiment.
        clean_dnld:     Removes all image chip data for experiment.
        clean_anno:     Removes all annotation data for experiment.
        clean_ckpt:     Removes all checkpoints for experiment.
        clean_pred:     Removes all predictions for experiment.
        clean_expt:     Removes all logs, sample data, checkpoints, and predictions for experiment.
        clean_nuke:     Removes all data for experiment including configs.

    Variables:
    These may be set at submake or as environment variables.

        SUNDIAL_BASE_PATH:           Base path for Sundial scripts. Default: 'shell pwd' of makefile
        SUNDIAL_SAMPLE_NAME:         Sample name. REQUIRED
        SUNDIAL_EXPERIMENT_PREFIX:   Sundial experiment name prefix used for naming. Default: ''
        SUNDIAL_EXPERIMENT_SUFFIX:   Experiment suffix used only for logging.
        SUNDIAL_ENV_NAME:            Sundial environment name. Default: 'sundial'
        SUNDIAL_PROCESSING:          Sundial processing method. Default: 'hpc'
        SUNDIAL_NODES:               Node within hpc to run job. Default: any node
```

### 1. Setup directories and conda environment.
This only works on x86_64 machines. Additional steps must be taken for ARM architectures. 
```console
# Clone repository cd into it. Omit --recurse-submodules if you do not plan to use provided backbone.
git clone --recurse-submodules https://github.com/eMapR/sundial.git
cd sundial

# Create environment in one of two ways
# 1. Install using yaml file directly
conda env create --prefix $(CONDA_PREFIX) --file environment.yaml --name sundial
# 2. Use submake. This will also create the necessary base directories.
make setup_env

# activate environment
conda activate sundial

# Authenticate on Google Earth Engine
earthengine authenticate
```

### 2. Generate experiment directories and config files for running submakes.
Experiment directories will be generated with the appropriate paths for all files that need to be created and read per experiement. Default configs found in settings.py will copied to the appropriate config location and any edits to them will automatically supercede settings.py values. Config files must be created before any other experiment submake.
```console
# Set environment variables for experiment. These variables can also be set at submake.
conda env config vars set \
    SUNDIAL_PROCESSING=local \
    SUNDIAL_SAMPLE_NAME=ads_1990-2022 \
    SUNDIAL_EXPERIMENT_PREFIX=5c

make setup
```

### 3. Sampling using a shapefile. 
The scripts will read the shapefile with the name in variable SUNDIAL_SAMPLE_NAME stored in shapes/SUNDIAL_SAMPLE_NAME and generate metadata files in shapefile format as well as index splits for training if specified. Sample files can be found in samples/SUNDIAL_EXPERIMENT_PREFIX_SUNDIAL_SAMPLE_NAME.
```console
# shapefile with geometry column must already be in shapes/
make sample
```

### 4. Generate annotation images from original shapefile and sample metadata.
Features in the shapefile are used to create annotations by pixel using the SAMPLER_CONFIG.strata_columns setting. The annotation image shape is (N C H W). This step is only necessary for superivsed learning. Note: Annotations are saved as their own image. This may not be as storage efficient but makes for loading into Pytorch much simpler and quicker.
```console
make annotate
```

### 5. Downloading using the generated chip samples via GEE.
GEE is used to generate raster data based on polygons from the shapefile given in the sample. Download limits apply so use your own discretion. I found the limit to be chips of shape (256, 256, 256) to be the the upper limit but this may vary depending on processing, scale, projection, etc. Depending on your internet speed, ~32k chips of shape (32, 256, 256) can be downloaded in just a few hours. This is usually more than enough data. Rasters are saved to samples/SUNDIAL_EXPERIMENT_PREFIX_SUNDIAL_SAMPLE_NAME.
```console
make download
```

### 6. Train a model using the downloaded chips.
Included in this repo is a simple segmentation model using a fully convolutional network built on Prithvi's foundation model as a backbone that needs to be finetuned. Using the CLI, you can mix and match models. See [Pytorch Lightning Docs](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) for more details on config files. For now the runner will import models by name from the models module. Write additional models there or use one that is prebuilt in lightning. Backbones are also stored in the backbones directory in src and can be imported in a similar way. Any paths to chip or annotation data created in the previous steps are automatically loaded.
```console
make fit
```

### 7. View logs.
Comet, an ML cloud reporting service offered free for research, is used as the default logging package for images and training data. See [Lightning Comet Docs](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.comet.html#module-lightning.pytorch.loggers.comet) and [Comet Experiment Docs](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/Experiment/#experimentlog_metric) for more details. Environment variables needed can be found in src/settings.py. Alternatively, a separate logger can be written and included via config files.

## Config files

While the framework for deep learning is build on [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html), the pipeline down from GEE can be configured using a similar format. See src/pipeline/settings.py for more details and configs directory for examples. In addition, images are generated using an annual medoid composite by default since for now, that is our focus, but another image generator function can be provided to the class if you instantiate it separately.

The configs worth looking at are:
| CONFIG | DESCRIPTION | OPTIONS |
| --- | --- | --- |
| SAMPLER_CONFIG.method | Method to generate samples options with polygon | "convering_grid", "random", "gee_stratified", "centroid" |
| SAMPLER_CONFIG.num_points | Number of points to sample from original shapefile | |
| SAMPLER_CONFIG.strata_columns | Columns found in shapefile to generate strata for training | |
| SAMPLER_CONFIG.file_type | File type to download | "NPY", "NUMPY_NDARRAY", "ZARR", "GEO_TIFF" |
| SAMPLER_CONFIG.scale | Scale to generate image raster | |
| SAMPLER_CONFIG.projection | Reprojection string | (EPSG:****) |
| SAMPLER_CONFIG.pixel_edge_size | Edge size of chip image in pixels | |
