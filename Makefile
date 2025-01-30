.ONE_SHELL:
SHELL := /bin/bash

.EXPORT_ALL_VARIABLES:
.SILENT:
.PHONY:

ifndef SUNDIAL_BASE_PATH
SUNDIAL_BASE_PATH := $(shell pwd -P)
endif

ifndef SUNDIAL_ENV_NAME 
SUNDIAL_ENV_NAME := sundial
endif

ifndef SUNDIAL_PROCESSING
SUNDIAL_PROCESSING := hpc
endif

ifndef SUNDIAL_NODES
SUNDIAL_NODES :=
endif

ifndef SUNDIAL_PACKAGE_FORMAT
SUNDIAL_PACKAGE_FORMAT := tar
endif

ifndef SUNDIAL_BASE_PATH
SUNDIAL_BASE_PATH := $(shell pwd)
endif

SUNDIAL_EXPERIMENT_BASE_NAME := $(SUNDIAL_EXPERIMENT_PREFIX)_$(SUNDIAL_SAMPLE_NAME)
ifndef SUNDIAL_EXPERIMENT_SUFFIX
SUNDIAL_EXPERIMENT_FULL_NAME := $(SUNDIAL_EXPERIMENT_BASE_NAME)
else
SUNDIAL_EXPERIMENT_FULL_NAME := $(SUNDIAL_EXPERIMENT_BASE_NAME)_$(SUNDIAL_EXPERIMENT_SUFFIX)
endif

default:
	echo "Welcome to Sundial!"
	echo
	echo "    Methods:"
	echo
	echo "        setup_env:      Sets up Sundial directories and environment. The existing environment will be overwritten."
	echo "        setup:          Sets up Sundial experiment directories and config files using defaults found in src/settings.py."
	echo "        clink:          Copies Sundial experiment configs from source experiment to new experiment and links sample configs and data. Requires SSOURCE variable."
	echo
	echo "        sample:         Generates chip sample polygons using Google Earth Engine and provided shapefile."
	echo "        annotate:       Collects image annotations for experiment using sample polygons."
	echo "        download:       Downloads image chips for experiment using sample polygons."
	echo "        index:          Creates indicies for training from chip and anno data."
	echo
	echo "        fit:            Train model using subset of data from sample."
	echo "        validate:       Validate model using subset of data from sample."
	echo "        test:           Test model using subset of data from sample."
	echo "        predict:        Predict an image using subset of data from sample."
	echo "        package:        Convert predictions to GeoTIFF if not already in format and compress for download."
	echo
	echo "        status:         Check status of all jobs for user."
	echo "        vars:           Print all Sundial variables."
	echo "        (method)_err:   Print ERRORs and CRITICALs in log file and print stderr from file on HPC."
	echo
	echo "        clean:          Removes all logs, checkpoints, and predictions for experiment."
	echo "        clean_outs:     Removes all stdouts for experiment."
	echo "        clean_logs:     Removes all logs for experiment."
	echo "        clean_samp:     Removes all sample data for experiment."
	echo "        clean_dnld:     Removes all image chip data for experiment."
	echo "        clean_anno:     Removes all annotation data for experiment."
	echo "        clean_ckpt:     Removes all checkpoints for experiment."
	echo "        clean_pred:     Removes all predictions for experiment."
	echo "        clean_expt:     Removes all logs, sample data, checkpoints, and predictions for experiment."
	echo "        clean_nuke:     Removes all data for experiment including configs."
	echo
	echo "    Variables:"
	echo "    These may be set at submake or as environment variables."
	echo
	echo "        SUNDIAL_BASE_PATH:           Base path for Sundial scripts. Default: 'shell pwd' of makefile"
	echo "        SUNDIAL_SAMPLE_NAME:         Sample name. REQUIRED"
	echo "        SUNDIAL_EXPERIMENT_PREFIX:   Sundial experiment name prefix used for naming. REQUIRED"
	echo "        SUNDIAL_EXPERIMENT_SUFFIX:   Experiment suffix used only for logging."
	echo "        SUNDIAL_ENV_NAME:            Sundial environment name. Default: 'sundial'"
	echo "        SUNDIAL_PROCESSING:          Sundial processing method. Default: 'hpc'"
	echo "        SUNDIAL_NODES:               Node within hpc to run job. Default: any node"
	echo "        SUNDIAL_GPU_PARTITION:       Partition within hpc to run gpu based jobs."
	echo "        SUNDIAL_CPU_PARTITION:       Partition within hpc to run cpu based jobs."
	echo "        SUNDIAL_PACKAGE_FORMAT:      Format to package predictions. Default: 'tar'"
	echo

setup_env:
	echo "Setting up directories for $(SUNDIAL_BASE_PATH) and conda environment "$(SUNDIAL_ENV_NAME)".";
	mkdir -p $(SUNDIAL_BASE_PATH)/logs;
	mkdir -p $(SUNDIAL_BASE_PATH)/samples;
	mkdir -p $(SUNDIAL_BASE_PATH)/checkpoints;
	mkdir -p $(SUNDIAL_BASE_PATH)/predictions;
	mkdir -p $(SUNDIAL_BASE_PATH)/configs;
	mkdir -p $(SUNDIAL_BASE_PATH)/src/models/backbones;
	if [[ "$(ARCH)" == aarch64 ]]; then \
		conda env create -f $(SUNDIAL_BASE_PATH)/environment_a64.yaml -n $(SUNDIAL_ENV_NAME) -y; \
	else \
		conda env create -f $(SUNDIAL_BASE_PATH)/environment_x86.yaml -n $(SUNDIAL_ENV_NAME) -y; \
	fi; \
	echo "Sundial setup complete!";

setup: _experiment_name_check
	echo "Setting up Sundial experiment for $(SUNDIAL_EXPERIMENT_BASE_NAME)...";
	mkdir -p $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_BASE_NAME);
	mkdir -p $(SUNDIAL_BASE_PATH)/samples/$(SUNDIAL_EXPERIMENT_BASE_NAME);
	mkdir -p $(SUNDIAL_BASE_PATH)/checkpoints/$(SUNDIAL_EXPERIMENT_BASE_NAME);
	mkdir -p $(SUNDIAL_BASE_PATH)/predictions/$(SUNDIAL_EXPERIMENT_BASE_NAME);
	if [[ -d $(SUNDIAL_BASE_PATH)/configs/$(SUNDIAL_EXPERIMENT_BASE_NAME) ]]; then \
		echo "WARNING: Configs directory found. To restart experiment from scratch, use 'make clean_nuke' then 'make setup'..."; \
	else \
		echo "Generating Sundial config files for experiment with values from settings.py..."; \
		python $(SUNDIAL_BASE_PATH)/src/settings.py; \
	fi;

clink: _experiment_name_check
	if [[ -z "$(SSOURCE)" ]]; then \
		echo "Please provide a source experiment to copy from in variable "SSOURCE" (eg 'make clink SSOURCE=14k')..."; \
		exit 1; \
	fi; \
	source_exp=$(SSOURCE)_$(SUNDIAL_SAMPLE_NAME); \
	echo "Copying Sundial experiment $$source_exp configs to $(SUNDIAL_EXPERIMENT_BASE_NAME)..."; \
	cp -r $(SUNDIAL_BASE_PATH)/configs/$$source_exp $(SUNDIAL_BASE_PATH)/configs/$(SUNDIAL_EXPERIMENT_BASE_NAME); \
	rm -rfv $(SUNDIAL_BASE_PATH)/configs/$(SUNDIAL_EXPERIMENT_BASE_NAME)/sample.yaml; \
	echo "WARNING: This will create soft links to the source sammple data and sample config file."; \
	ln -s $(SUNDIAL_BASE_PATH)/samples/$$source_exp $(SUNDIAL_BASE_PATH)/samples/$(SUNDIAL_EXPERIMENT_BASE_NAME); \
	ln -s $(SUNDIAL_BASE_PATH)/configs/$$source_exp/sample.yaml $(SUNDIAL_BASE_PATH)/configs/$(SUNDIAL_EXPERIMENT_BASE_NAME)/sample.yaml;

vars: _experiment_name_check
	echo "SUNDIAL_BASE_PATH: $(SUNDIAL_BASE_PATH)";
	echo "SUNDIAL_SAMPLE_NAME: $(SUNDIAL_SAMPLE_NAME)";
	echo "SUNDIAL_EXPERIMENT_PREFIX: $(SUNDIAL_EXPERIMENT_PREFIX)";
	echo "SUNDIAL_EXPERIMENT_SUFFIX: $(SUNDIAL_EXPERIMENT_SUFFIX)";
	echo "SUNDIAL_EXPERIMENT_BASE_NAME: $(SUNDIAL_EXPERIMENT_BASE_NAME)";
	echo "SUNDIAL_EXPERIMENT_FULL_NAME: $(SUNDIAL_EXPERIMENT_FULL_NAME)";
	echo "SUNDIAL_ENV_NAME: $(SUNDIAL_ENV_NAME)";
	echo "SUNDIAL_PROCESSING: $(SUNDIAL_PROCESSING)";
	echo "SUNDIAL_NODES: $(SUNDIAL_NODES)";
	echo "SUNDIAL_GPU_PARTITION: $(SUNDIAL_GPU_PARTITION)";
	echo "SUNDIAL_CPU_PARTITION: $(SUNDIAL_CPU_PARTITION)";
	echo "SUNDIAL_PACKAGE_FORMAT: $(SUNDIAL_PACKAGE_FORMAT)";

rename: _experiment_name_check
	if [[ -z "$(SNEW)" ]]; then \
		echo "Please provide a new experiment name in variable "SNEW" (eg 'make rename SNEW=15k')..."; \
		exit 1; \
	fi; \
	new_exp=$(SNEW)_$(SUNDIAL_SAMPLE_NAME); \
	echo "Renaming Sundial experiment $(SUNDIAL_EXPERIMENT_BASE_NAME) to $$new_exp..."; \
	mv $(SUNDIAL_BASE_PATH)/configs/$(SUNDIAL_EXPERIMENT_BASE_NAME) $(SUNDIAL_BASE_PATH)/configs/$$new_exp; \
	mv $(SUNDIAL_BASE_PATH)/samples/$(SUNDIAL_EXPERIMENT_BASE_NAME) $(SUNDIAL_BASE_PATH)/samples/$$new_exp; \
	mv $(SUNDIAL_BASE_PATH)/checkpoints/$(SUNDIAL_EXPERIMENT_BASE_NAME) $(SUNDIAL_BASE_PATH)/checkpoints/$$new_exp; \
	mv $(SUNDIAL_BASE_PATH)/predictions/$(SUNDIAL_EXPERIMENT_BASE_NAME) $(SUNDIAL_BASE_PATH)/predictions/$$new_exp; \
	mv $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_BASE_NAME) $(SUNDIAL_BASE_PATH)/logs/$$new_exp;

sample: _sample
	echo "Processing polygon sample from $(SUNDIAL_SAMPLE_NAME) shapefile for $(SUNDIAL_EXPERIMENT_PREFIX). Uno momento...";
	$(eval export SUNDIAL_PARTITION=$(SUNDIAL_CPU_PARTITION))
	$(MAKE) -s _run

annotate: _annotate
	echo "Collecting image annotations for $(SUNDIAL_EXPERIMENT_FULL_NAME). This might take a sec...";
	$(eval export SUNDIAL_PARTITION=$(SUNDIAL_CPU_PARTITION))
	$(MAKE) -s _run

download: _download
	echo "Downloading image chips for $(SUNDIAL_EXPERIMENT_FULL_NAME). Sit tight...";
	$(eval export SUNDIAL_PARTITION=$(SUNDIAL_CPU_PARTITION))
	$(MAKE) -s _run

index: _index
	echo "Creating indicies for $(SUNDIAL_EXPERIMENT_FULL_NAME)...";
	$(eval export SUNDIAL_PARTITION=$(SUNDIAL_CPU_PARTITION))
	$(MAKE) -s _run

fit: _fit
	echo "Training model for $(SUNDIAL_EXPERIMENT_FULL_NAME)... This may take a while...";
	$(eval export SUNDIAL_PARTITION=$(SUNDIAL_GPU_PARTITION))
	$(MAKE) -s _run

validate: _validate
	echo "Validating model for $(SUNDIAL_EXPERIMENT_FULL_NAME)... Go ahead and hold your breath...";
	$(eval export SUNDIAL_PARTITION=$(SUNDIAL_GPU_PARTITION))
	$(MAKE) -s _run

test: _test
	echo "Testing model for $(SUNDIAL_EXPERIMENT_FULL_NAME)... Use 'make test_err' to check for critical errors...";
	$(eval export SUNDIAL_PARTITION=$(SUNDIAL_GPU_PARTITION))
	$(MAKE) -s _run

predict: _predict
	echo "Making predictions for $(SUNDIAL_EXPERIMENT_FULL_NAME)... Lets see if this works!";
	$(eval export SUNDIAL_PARTITION=$(SUNDIAL_GPU_PARTITION))
	$(MAKE) -s _run

package: _package
	echo "Packaging predictions for $(SUNDIAL_EXPERIMENT_FULL_NAME)...";
	$(eval export SUNDIAL_PARTITION=$(SUNDIAL_CPU_PARTITION))
	$(MAKE) -s _run

status: _hpc_check
	squeue -u $(USER) --format="%.18i %.9P %.40j %.8u %.8T %.10M %.9l %.6D %R";

sample_err: _sample
	$(MAKE) -s _read_err;

annotate_err: _annotate
	$(MAKE) -s _read_err;

download_err: _download
	$(MAKE) -s _read_err;

index_err: _index
	$(MAKE) -s _read_err;

fit_err: _fit
	$(MAKE) -s _read_err;

validate_err: _validate
	$(MAKE) -s _read_err;

test_err: _test
	$(MAKE) -s _read_err;

predict_err: _predict
	$(MAKE) -s _read_err;

package_err: _package
	$(MAKE) -s _read_err;

clean: _experiment_name_check
	echo "Cleaning up logs, checkpoints, and predictions for $(SUNDIAL_EXPERIMENT_BASE_NAME).";
	rm -rfv $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_BASE_NAME)/*;
	rm -rfv $(SUNDIAL_BASE_PATH)/checkpoints/$(SUNDIAL_EXPERIMENT_BASE_NAME)/*;
	rm -rfv $(SUNDIAL_BASE_PATH)/predictions/$(SUNDIAL_EXPERIMENT_BASE_NAME)/*;

clean_logs: _experiment_name_check
	echo "Cleaning up logs for $(SUNDIAL_EXPERIMENT_BASE_NAME).";
	rm -rfv $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_BASE_NAME)/*;

clean_samp: _experiment_name_check
	echo "Cleaning up all sample data for $(SUNDIAL_EXPERIMENT_BASE_NAME).";
	rm -rfv $(SUNDIAL_BASE_PATH)/samples/$(SUNDIAL_EXPERIMENT_BASE_NAME)/*;

clean_dnld: _experiment_name_check
	echo "Cleaning up chip data for $(SUNDIAL_EXPERIMENT_BASE_NAME).";
	rm -rfv $(SUNDIAL_BASE_PATH)/samples/$(SUNDIAL_EXPERIMENT_BASE_NAME)/chip_data*;

clean_anno: _experiment_name_check
	echo "Cleaning up annotation data for $(SUNDIAL_EXPERIMENT_BASE_NAME).";
	rm -rfv $(SUNDIAL_BASE_PATH)/samples/$(SUNDIAL_EXPERIMENT_BASE_NAME)/anno_data*;

clean_ckpt: _experiment_name_check
	echo "Cleaning up checkpoints generated for $(SUNDIAL_EXPERIMENT_BASE_NAME).";
	rm -rfv $(SUNDIAL_BASE_PATH)/checkpoints/$(SUNDIAL_EXPERIMENT_BASE_NAME)/*;

clean_pred: _experiment_name_check
	echo "Cleaning up predictions generated for $(SUNDIAL_EXPERIMENT_BASE_NAME).";
	rm -rfv $(SUNDIAL_BASE_PATH)/predictions/$(SUNDIAL_EXPERIMENT_BASE_NAME)/*;

clean_expt: _experiment_name_check
	echo "Cleaning up logs, sample data, checkpoints, and predictions for $(SUNDIAL_EXPERIMENT_BASE_NAME).";
	rm -rfv $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_BASE_NAME)/*;
	rm -rfv $(SUNDIAL_BASE_PATH)/samples/$(SUNDIAL_EXPERIMENT_BASE_NAME)/*;
	rm -rfv $(SUNDIAL_BASE_PATH)/checkpoints/$(SUNDIAL_EXPERIMENT_BASE_NAME)/*;
	rm -rfv $(SUNDIAL_BASE_PATH)/predictions/$(SUNDIAL_EXPERIMENT_BASE_NAME)/*;

clean_nuke: _experiment_name_check
	echo "Cleaning up all data for $(SUNDIAL_EXPERIMENT_BASE_NAME).";
	rm -rfv $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_BASE_NAME);
	rm -rfv $(SUNDIAL_BASE_PATH)/samples/$(SUNDIAL_EXPERIMENT_BASE_NAME);
	rm -rfv $(SUNDIAL_BASE_PATH)/checkpoints/$(SUNDIAL_EXPERIMENT_BASE_NAME);
	rm -rfv $(SUNDIAL_BASE_PATH)/predictions/$(SUNDIAL_EXPERIMENT_BASE_NAME);
	rm -rfv $(SUNDIAL_BASE_PATH)/configs/$(SUNDIAL_EXPERIMENT_BASE_NAME);

_config_dir_check:
	if [[ ! -d $(SUNDIAL_BASE_PATH)/configs/$(SUNDIAL_EXPERIMENT_BASE_NAME) ]]; then \
		echo "No configs found. Please run make setup first..."; \
		exit 1; \
	fi;

_experiment_name_check:
	if [[ -z "$(SUNDIAL_SAMPLE_NAME)" ]]; then \
		echo "Please provide sample name. (eg 'make setup SUNDIAL_SAMPLE_NAME=shapefile_name') "; \
		exit 1; \
	fi; \
	if [[ -z "$(SUNDIAL_EXPERIMENT_PREFIX)" ]]; then \
		echo "Please provide a Sundial experiment prefix. (eg 'make setup SUNDIAL_EXPERIMENT_PREFIX=14k')..."; \
		exit 1; \
	fi;


_hpc_check:
	if [[ "$(SUNDIAL_PROCESSING)" != hpc ]]; then \
		echo "This method is only available on an HPC with slurm..."; \
		exit 1; \
	fi;

_sample:
	$(eval export SUNDIAL_METHOD=sample)

_annotate:
	$(eval export SUNDIAL_METHOD=annotate)

_download:
	$(eval export SUNDIAL_METHOD=download)

_index:
	$(eval export SUNDIAL_METHOD=index)

_fit:
	$(eval export SUNDIAL_METHOD=fit)

_validate:
	$(eval export SUNDIAL_METHOD=validate)

_test:
	$(eval export SUNDIAL_METHOD=test)

_predict:
	$(eval export SUNDIAL_METHOD=predict)

_package:
	$(eval export SUNDIAL_METHOD=package)

_run: _experiment_name_check
	if [[ "$(SUNDIAL_PROCESSING)" == hpc ]]; then \
		job_name=$(SUNDIAL_METHOD)_$(SUNDIAL_EXPERIMENT_FULL_NAME); \
		if [[ -z "$(SUNDIAL_NODES)" ]]; then \
			nodes=$(shell sinfo -h -p $(SUNDIAL_PARTITION) -o "%N"); \
		else \
			nodes=$(SUNDIAL_NODES); \
		fi; \
		echo "Running on HPC..."; \
		sbatch \
			--job-name=$$job_name \
			--output=$(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_BASE_NAME)/$(SUNDIAL_METHOD).o \
			--error=$(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_BASE_NAME)/$(SUNDIAL_METHOD).e \
			--chdir=$(SUNDIAL_BASE_PATH) \
			--partition=$(SUNDIAL_PARTITION) \
			--nodelist=$$nodes \
			--export=ALL \
			$(SUNDIAL_BASE_PATH)/slurm/$(SUNDIAL_METHOD).slurm; \
	else \
		echo "Running on local machine..."; \
		python $(SUNDIAL_BASE_PATH)/src/runner.py; \
	fi;

_read_err: _experiment_name_check
	if [[ -f $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_BASE_NAME)/$(SUNDIAL_METHOD).e ]]; then \
		cat $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_BASE_NAME)/$(SUNDIAL_METHOD).e; \
	fi; \
	if [[ -f $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_BASE_NAME)/$(SUNDIAL_METHOD).log ]]; then \
		cat $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_BASE_NAME)/$(SUNDIAL_METHOD).log | grep ERROR; \
		cat $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_BASE_NAME)/$(SUNDIAL_METHOD).log | grep CRITICAL; \
		tail $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_BASE_NAME)/$(SUNDIAL_METHOD).log; \
	fi;
