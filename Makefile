SHELL := /bin/bash

.ONE_SHELL:
.EXPORT_ALL_VARIABLES:
.SILENT:
.PHONY:

ifndef SUNDIAL_BASE_PATH
SUNDIAL_BASE_PATH := $(shell pwd -P)
endif

ifndef SUNDIAL_SAMPLE_NAME
SUNDIAL_SAMPLE_NAME :=
endif

ifndef SUNDIAL_EXPERIMENT_NAME
ifndef SUNDIAL_EXPERIMENT_SUFFIX
SUNDIAL_EXPERIMENT_SUFFIX :=
SUNDIAL_EXPERIMENT_NAME := $(SUNDIAL_SAMPLE_NAME)
else
SUNDIAL_EXPERIMENT_NAME := $(SUNDIAL_SAMPLE_NAME)_$(SUNDIAL_EXPERIMENT_SUFFIX)
endif
endif

ifndef SUNDIAL_ENV_NAME 
SUNDIAL_ENV_NAME := sundial
endif

ifndef SUNDIAL_PROCESSING
SUNDIAL_PROCESSING := hpc
endif

default:
	@echo "Welcome to Sundial!"
	@echo
	@echo "    Methods:"
	@echo "        setup:          Sets up Sundial directories and environment. The existing environment will be overwritten."
	@echo "        setup_exp:      Sets up Sundial experiment directories and config files using defaults found in src/settings.py."
	@echo
	@echo "        sample:         Generates chip sample polygons using Google Earth Engine and provided shapefile."
	@echo "        annotate:       Collects image annotations for experiment using sample polygons."
	@echo "        download:       Downloads image chips for experiment using sample polygons."
	@echo
	@echo "        fit:            Train model using subset of data from sample."
	@echo "        validate:       Validate model subset of using data from sample."
	@echo "        test:           Test model using subset of data from sample."
	@echo "        predict:        Predict and image from subset of data from sample."
	@echo "        package:        Compresses experiment to tar to export. The tar file will be saved in home directory and overwrite already existing archives."
	@echo "        status:         Check status of all jobs for experiment."
	@echo "        vars:           Print all Sundial variables."
	@echo
	@echo "        clean:          Removes all logs, checkpoints, and predictions for experiment."
	@echo "        clean_logs:     Removes all logs for experiment."
	@echo "        clean_sample:   Removes all sample data for experiment."
	@echo "        clean_download: Removes all image chip data for experiment."
	@echo "        clean_anno:     Removes all annotation data for experiment."
	@echo "        clean_ckpt:     Removes all checkpoints for experiment."
	@echo "        clean_predict:  Removes all predictions for experiment."
	@echo "        clean_exp:      Removes all logs, sample data, checkpoints, and predictions for experiment."
	@echo "        clean_all:      Removes all data for experiment including configs."
	@echo
	@echo "    Variables:"
	@echo "        SUNDIAL_BASE_PATH:           Base path for Sundial scripts. Default: 'shell pwd' of makefile"
	@echo "        SUNDIAL_SAMPLE_NAME:         Sample name. REQUIRED"
	@echo "        SUNDIAL_EXPERIMENT_SUFFIX:   Sundial experiment name. Default: ''"
	@echo "        SUNDIAL_ENV_NAME:            Sundial environment name. Default: 'sundial'"
	@echo "        SUNDIAL_PROCESSING:          Sundial processing method. Default: 'hpc'"
	@echo

setup: _experiment_name_check
	echo "Setting up directories for $(SUNDIAL_BASE_PATH) and conda environment "$(SUNDIAL_ENV_NAME)". The existing environment will be overwritten..."; \
	mkdir -p $(SUNDIAL_BASE_PATH)/logs; \
	mkdir -p $(SUNDIAL_BASE_PATH)/samples; \
	mkdir -p $(SUNDIAL_BASE_PATH)/checkpoints; \
	mkdir -p $(SUNDIAL_BASE_PATH)/predictions; \
	mkdir -p $(SUNDIAL_BASE_PATH)/configs; \
	conda env create -f $(SUNDIAL_BASE_PATH)/environment.yml -n $(SUNDIAL_ENV_NAME) -p $(CONDA_PREFIX) -y --force; \
	conda activate $(SUNDIAL_ENV_NAME); \
	echo "Sundial setup complete!"; \

setup_exp: _experiment_name_check
	echo "Setting up Sundial experiment for $(SUNDIAL_EXPERIMENT_NAME)..."; \
	mkdir -p $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME); \
	mkdir -p $(SUNDIAL_BASE_PATH)/samples/$(SUNDIAL_EXPERIMENT_NAME); \
	mkdir -p $(SUNDIAL_BASE_PATH)/checkpoints/$(SUNDIAL_EXPERIMENT_NAME); \
	mkdir -p $(SUNDIAL_BASE_PATH)/predictions/$(SUNDIAL_EXPERIMENT_NAME); \
	if [[ -d $(SUNDIAL_BASE_PATH)/configs/$(SUNDIAL_EXPERIMENT_NAME) ]]; then \
		echo "Configs directory found. To restart experiment from scratch, use make clean_all then make config..."; \
		exit 1; \
	else \
		echo "Generateing Sundial config files for experiment with values from sample and download..."; \
		python $(SUNDIAL_BASE_PATH)/src/pipeline/settings.py; \
	fi; \

sample: _sample
	echo "Processing polygon sample from $(SUNDIAL_SAMPLE_NAME) shapefile. Uno momento..."; \
	$(eval export SUNDIAL_PARTITION=$(CPU_PARTITION))
	$(MAKE) -s _run

annotate: _annotate
	echo "Collecting image annotations for $(SUNDIAL_EXPERIMENT_NAME). This might take a sec..."; \
	$(eval export SUNDIAL_PARTITION=$(CPU_PARTITION))
	$(MAKE) -s _run

download: _download
	echo "Downloading image chips for $(SUNDIAL_EXPERIMENT_NAME). Sit tight..."; \
	$(eval export SUNDIAL_PARTITION=$(CPU_PARTITION))
	$(MAKE) -s _run

fit: _fit
	echo "Training model... This may take a while..."; \
	$(eval export SUNDIAL_PARTITION=$(GPU_PARTITION))
	$(MAKE) -s _run

validate: _validate
	echo "Validating model... Go ahead and hold your breath..."; \
	$(eval export SUNDIAL_PARTITION=$(GPU_PARTITION))
	$(MAKE) -s _run

test: _test
	echo "Testing model... Please check $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME)/test.e..."; \
	$(eval export SUNDIAL_PARTITION=$(GPU_PARTITION))
	$(MAKE) -s _run

predict: _predict
	echo "Predicting image using model... Lets see if this works!"; \
	$(eval export SUNDIAL_PARTITION=$(GPU_PARTITION))
	$(MAKE) -s _run

status: _experiment_name_check _hpc_check
	squeue -u $(USER) --format="%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R" \

vars: _experiment_name_check
	echo "SUNDIAL_BASE_PATH: $(SUNDIAL_BASE_PATH)"; \
	echo "SUNDIAL_SAMPLE_NAME: $(SUNDIAL_SAMPLE_NAME)"; \
	echo "SUNDIAL_EXPERIMENT_SUFFIX: $(SUNDIAL_EXPERIMENT_SUFFIX)"; \
	echo "SUNDIAL_EXPERIMENT_NAME: $(SUNDIAL_EXPERIMENT_NAME)"; \
	echo "SUNDIAL_ENV_NAME: $(SUNDIAL_ENV_NAME)"; \
	echo "SUNDIAL_PROCESSING: $(SUNDIAL_PROCESSING)"; \

package: _experiment_name_check
	echo "Compressing logs, checkpoints, and predictions for $(SUNDIAL_EXPERIMENT_NAME) to tar. Tar file will be saved in home directory and overwrite already existing archives."; \
	tar -cvzf $(HOME)/$(SUNDIAL_EXPERIMENT_NAME).tar.gz \
		logs/$(SUNDIAL_EXPERIMENT_NAME) \
		checkpoints/$(SUNDIAL_EXPERIMENT_NAME) \
		predictions/$(SUNDIAL_EXPERIMENT_NAME) \

sample_watch: _sample _hpc_check
	$(MAKE) -s _watch

annotate_watch: _annotate _hpc_check
	$(MAKE) -s _watch

download_watch: _download _hpc_check
	$(MAKE) -s _watch

fit_watch: _fit _hpc_check
	$(MAKE) -s _watch

validate_watch: _validate _experiment_name_check _hpc_check
	$(MAKE) -s _watch

test_watch: _test _hpc_check
	$(MAKE) -s _watch

predict_watch: _predict _hpc_check
	$(MAKE) -s _watch

_watch: _experiment_name_check _hpc_check
	echo "Watching $(SUNDIAL_METHOD) for $(SUNDIAL_EXPERIMENT_NAME)..."; \
	while true; do \
		tput clear; \
		awk 'END {print}' $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME)/$(SUNDIAL_METHOD).o; \
		sleep 2; \
	done; \

_sample: _experiment_name_check
	$(eval export SUNDIAL_METHOD=sample)

_annotate: _experiment_name_check
	$(eval export SUNDIAL_METHOD=annotate)

_download: _experiment_name_check
	$(eval export SUNDIAL_METHOD=download)

_fit: _experiment_name_check
	$(eval export SUNDIAL_METHOD=fit)

_validate: _experiment_name_check
	$(eval export SUNDIAL_METHOD=validate)

_test: _experiment_name_check
	$(eval export SUNDIAL_METHOD=test)

_predict: _experiment_name_check
	$(eval export SUNDIAL_METHOD=predict)

clean: _experiment_name_check
	echo "Cleaning up logs, checkpoints, and predictions for $(SUNDIAL_EXPERIMENT_NAME)."; \
	rm -rf $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME); \
	rm -rf $(SUNDIAL_BASE_PATH)/checkpoints/$(SUNDIAL_EXPERIMENT_NAME); \
	rm -rf $(SUNDIAL_BASE_PATH)/predictions/$(SUNDIAL_EXPERIMENT_NAME); \

clean_logs: _experiment_name_check
	echo "Cleaning up logs for $(SUNDIAL_EXPERIMENT_NAME)."; \
	rm -rf $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME); \

clean_sample: _experiment_name_check
	echo "Cleaning up all sample data for $(SUNDIAL_EXPERIMENT_NAME)."; \
	rm -rf $(SUNDIAL_BASE_PATH)/samples/$(SUNDIAL_EXPERIMENT_NAME); \

clean_download: _experiment_name_check
	echo "Cleaning up chip data for $(SUNDIAL_EXPERIMENT_NAME)."; \
	rm -rf $(SUNDIAL_BASE_PATH)/samples/$(SUNDIAL_EXPERIMENT_NAME)/chip_data*; \

clean_anno: _experiment_name_check
	echo "Cleaning up annotation data for $(SUNDIAL_EXPERIMENT_NAME)."; \
	rm -rf $(SUNDIAL_BASE_PATH)/samples/$(SUNDIAL_EXPERIMENT_NAME)/anno_data*; \

clean_ckpt: _experiment_name_check
	echo "Cleaning up checkpoints generated for $(SUNDIAL_EXPERIMENT_NAME)."; \
	rm -rf $(SUNDIAL_BASE_PATH)/checkpoints/$(SUNDIAL_EXPERIMENT_NAME); \

clean_predict: _experiment_name_check
	echo "Cleaning up predictions generated for $(SUNDIAL_EXPERIMENT_NAME)."; \
	rm -rf $(SUNDIAL_BASE_PATH)/predictions/$(SUNDIAL_EXPERIMENT_NAME); \

clean_exp: _experiment_name_check
	echo "Cleaning up logs, sample data, and checkpoints for $(SUNDIAL_EXPERIMENT_NAME)."; \
	rm -rf $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME); \
	rm -rf $(SUNDIAL_BASE_PATH)/samples/$(SUNDIAL_EXPERIMENT_NAME); \
	rm -rf $(SUNDIAL_BASE_PATH)/checkpoints/$(SUNDIAL_EXPERIMENT_NAME); \
	rm -rf $(SUNDIAL_BASE_PATH)/predictions/$(SUNDIAL_EXPERIMENT_NAME); \

clean_all: _experiment_name_check
	echo "Deleting logs, sample data, and configs for $(SUNDIAL_EXPERIMENT_NAME)."; \
	rm -rf $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME); \
	rm -rf $(SUNDIAL_BASE_PATH)/samples/$(SUNDIAL_EXPERIMENT_NAME); \
	rm -rf $(SUNDIAL_BASE_PATH)/checkpoints/$(SUNDIAL_EXPERIMENT_NAME); \
	rm -rf $(SUNDIAL_BASE_PATH)/predictions/$(SUNDIAL_EXPERIMENT_NAME); \
	rm -rf $(SUNDIAL_BASE_PATH)/configs/$(SUNDIAL_EXPERIMENT_NAME); \

_config_dir_check:
	if [[ ! -d $(SUNDIAL_BASE_PATH)/configs/$(SUNDIAL_EXPERIMENT_NAME) ]]; then \
		echo "No configs found. Please run make setup_exp first..."; \
		exit 1; \
	fi; \

_experiment_name_check:
	if [[ -z "$(SUNDIAL_EXPERIMENT_NAME)" ]]; then \
		echo "Please provide sample name and experiment suffix to delete."; \
		exit 1; \
	fi; \

_hpc_check:
	if [[ "$(SUNDIAL_PROCESSING)" != hpc ]]; then \
		echo "This method is only available on an HPC with slurm..."; \
		exit 1; \
	fi; \

_run:
	if [[ "$(SUNDIAL_PROCESSING)" == hpc ]]; then \
		echo "Running on HPC..."; \
		sbatch \
			--job-name=$(SUNDIAL_METHOD).$(SUNDIAL_EXPERIMENT_NAME) \
			--output=$(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME)/$(SUNDIAL_METHOD).o \
			--error=$(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME)/$(SUNDIAL_METHOD).e \
			--chdir=$(SUNDIAL_BASE_PATH) \
			--partition=$(SUNDIAL_PARTITION) \
			--export=ALL \
			$(SUNDIAL_BASE_PATH)/slurm/$(SUNDIAL_METHOD).slurm; \
	else \
		echo "Running on local machine..."; \
		python $(SUNDIAL_BASE_PATH)/src/runner.py; \
	fi; \
