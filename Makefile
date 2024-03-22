.ONE_SHELL:
SHELL := /bin/bash

.EXPORT_ALL_VARIABLES:
.SILENT:
.PHONY:

ifndef SUNDIAL_BASE_PATH
SUNDIAL_BASE_PATH := $(shell pwd -P)
endif

ifndef SUNDIAL_SAMPLE_NAME
SUNDIAL_SAMPLE_NAME :=
endif

ifndef SUNDIAL_EXPERIMENT_PREFIX
SUNDIAL_EXPERIMENT_PREFIX :=
SUNDIAL_EXPERIMENT_NAME := $(SUNDIAL_SAMPLE_NAME)
else
SUNDIAL_EXPERIMENT_NAME := $(SUNDIAL_EXPERIMENT_PREFIX)_$(SUNDIAL_SAMPLE_NAME)
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
	echo
	echo "        fit:            Train model using subset of data from sample."
	echo "        validate:       Validate model subset of using data from sample."
	echo "        test:           Test model using subset of data from sample."
	echo "        predict:        Predict an image from subset of data from sample."
	echo "        package:        Compresses experiment logs to tar to export. The tar file will be saved in home directory and overwrite already existing archives."
	echo "        status:         Check status of all jobs for user."
	echo "        vars:           Print all Sundial variables."
	echo "        (method)_out:   Watch stdout and stderr or logs for experiment."
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
	echo "        SUNDIAL_EXPERIMENT_PREFIX:   Sundial experiment name prefix. Default: ''"
	echo "        SUNDIAL_ENV_NAME:            Sundial environment name. Default: 'sundial'"
	echo "        SUNDIAL_PROCESSING:          Sundial processing method. Default: 'hpc'"
	echo "        SUNDIAL_NODES:               Node within hpc to run job. Default: any node"
	echo

setup_env: _experiment_name_check
	echo "Setting up directories for $(SUNDIAL_BASE_PATH) and conda environment "$(SUNDIAL_ENV_NAME)". The existing environment will be overwritten...";
	mkdir -p $(SUNDIAL_BASE_PATH)/logs;
	mkdir -p $(SUNDIAL_BASE_PATH)/samples;
	mkdir -p $(SUNDIAL_BASE_PATH)/checkpoints;
	mkdir -p $(SUNDIAL_BASE_PATH)/predictions;
	mkdir -p $(SUNDIAL_BASE_PATH)/configs;
	conda env create -f $(SUNDIAL_BASE_PATH)/environment.yml -n $(SUNDIAL_ENV_NAME) -p $(CONDA_PREFIX) -y --force;
	conda activate $(SUNDIAL_ENV_NAME);
	echo "Sundial setup complete!";

setup: _experiment_name_check
	echo "Setting up Sundial experiment for $(SUNDIAL_EXPERIMENT_NAME)...";
	mkdir -p $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME);
	mkdir -p $(SUNDIAL_BASE_PATH)/samples/$(SUNDIAL_EXPERIMENT_NAME);
	mkdir -p $(SUNDIAL_BASE_PATH)/checkpoints/$(SUNDIAL_EXPERIMENT_NAME);
	mkdir -p $(SUNDIAL_BASE_PATH)/predictions/$(SUNDIAL_EXPERIMENT_NAME);
	if [[ -d $(SUNDIAL_BASE_PATH)/configs/$(SUNDIAL_EXPERIMENT_NAME) ]]; then \
		echo "WARNING: Configs directory found. To restart experiment from scratch, use 'make clean_nuke' then make config..."; \
	else \
		echo "Generateing Sundial config files for experiment with values from settings.py..."; \
		python $(SUNDIAL_BASE_PATH)/src/pipeline/settings.py; \
	fi;

clink: _experiment_name_check
	if [[ -z "$(SSOURCE)" ]]; then \
		echo "Please provide a source experiment to copy from in variable "SSOURCE" (eg 'make clink SSOURCE=14k')..."; \
		exit 1; \
	fi; \
	echo "Copying Sundial experiment $(SSOURCE) configs to $(SUNDIAL_EXPERIMENT_NAME)...";
	echo "WARNING: This will create soft links to the source sammple data and sample config file.";
	source_exp = $(SSOURCE)_$(SUNDIAL_SAMPLE_NAME);
	cp -r $(SUNDIAL_BASE_PATH)/configs/$$source_exp $(SUNDIAL_BASE_PATH)/configs/$(SUNDIAL_EXPERIMENT_NAME);
	rm -rf $(SUNDIAL_BASE_PATH)/configs/$(SUNDIAL_EXPERIMENT_NAME)/sample.yaml;

	ln -s $(SUNDIAL_BASE_PATH)/samples/$$source_exp $(SUNDIAL_BASE_PATH)/samples/$(SUNDIAL_EXPERIMENT_NAME);
	ln -s $(SUNDIAL_BASE_PATH)/configs/$$source_exp/sample.yaml $(SUNDIAL_BASE_PATH)/configs/$(SUNDIAL_EXPERIMENT_NAME)/ sample.yaml;

sample: _sample
	echo "Processing polygon sample from $(SUNDIAL_SAMPLE_NAME) shapefile for $(SUNDIAL_EXPERIMENT_PREFIX). Uno momento...";
	$(eval export SUNDIAL_PARTITION=$(CPU_PARTITION))
	$(MAKE) -s _run

annotate: _annotate
	echo "Collecting image annotations for $(SUNDIAL_EXPERIMENT_NAME). This might take a sec...";
	$(eval export SUNDIAL_PARTITION=$(CPU_PARTITION))
	$(MAKE) -s _run

download: _download
	echo "Downloading image chips for $(SUNDIAL_EXPERIMENT_NAME). Sit tight...";
	$(eval export SUNDIAL_PARTITION=$(CPU_PARTITION))
	$(MAKE) -s _run

fit: _fit
	echo "Training model... This may take a while...";
	$(eval export SUNDIAL_PARTITION=$(GPU_PARTITION))
	$(MAKE) -s _run

validate: _validate
	echo "Validating model... Go ahead and hold your breath...";
	$(eval export SUNDIAL_PARTITION=$(GPU_PARTITION))
	$(MAKE) -s _run

test: _test
	echo "Testing model... Use 'make test_err' to check for critical errors...";
	$(eval export SUNDIAL_PARTITION=$(GPU_PARTITION))
	$(MAKE) -s _run

predict: _predict
	echo "Predicting image using model... Lets see if this works!";
	$(eval export SUNDIAL_PARTITION=$(GPU_PARTITION))
	$(MAKE) -s _run

status: _hpc_check
	squeue -u $(USER) --format="%.18i %.9P %.40j %.8u %.8T %.10M %.9l %.6D %R";

vars: _experiment_name_check
	echo "SUNDIAL_BASE_PATH: $(SUNDIAL_BASE_PATH)";
	echo "SUNDIAL_SAMPLE_NAME: $(SUNDIAL_SAMPLE_NAME)";
	echo "SUNDIAL_EXPERIMENT_PREFIX: $(SUNDIAL_EXPERIMENT_PREFIX)";
	echo "SUNDIAL_EXPERIMENT_NAME: $(SUNDIAL_EXPERIMENT_NAME)";
	echo "SUNDIAL_ENV_NAME: $(SUNDIAL_ENV_NAME)";
	echo "SUNDIAL_PROCESSING: $(SUNDIAL_PROCESSING)";
	echo "SUNDIAL_NODES: $(SUNDIAL_NODES)";
	echo "GPU_PARTITION: $(GPU_PARTITION)";
	echo "CPU_PARTITION: $(CPU_PARTITION)"; 

package: _experiment_name_check
	echo "Compressing logs for $(SUNDIAL_EXPERIMENT_NAME) to tar. Tar file will be saved in home directory and overwrite already existing archives.";
	tar -cvzf $(HOME)/$(SUNDIAL_EXPERIMENT_NAME).tar.gz logs/$(SUNDIAL_EXPERIMENT_NAME)
	echo "Logs saved at $(HOME)/$(SUNDIAL_EXPERIMENT_NAME).tar.gz";

sample_out: _sample
	$(MAKE) -s _watch_log

annotate_out: _annotate
	$(MAKE) -s _watch_log

download_out: _download
	$(MAKE) -s _watch_log

fit_out: _fit
	$(MAKE) -s _watch_std

validate_out: _validate
	$(MAKE) -s _watch_std

test_out: _test
	$(MAKE) -s _watch_std

predict_out: _predict
	$(MAKE) -s _watch_std

sample_err: _sample
	if [[ "$(SUNDIAL_PROCESSING)" == hpc ]]; then \
		cat $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME)/sample.e; \
	fi; \
	cat $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME)/sample.log | grep ERROR; \
	cat $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME)/sample.log | grep CRITICAL; \

annotate_err: _annotate
	if [[ "$(SUNDIAL_PROCESSING)" == hpc ]]; then \
		cat $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME)/annotate.e; \
	fi; \
	cat $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME)/annotate.log | grep ERROR; \
	cat $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME)/annotate.log | grep CRITICAL; \

download_err: _download
	if [[ "$(SUNDIAL_PROCESSING)" == hpc ]]; then \
		cat $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME)/download.e; \
	fi; \
	cat $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME)/download.log | grep ERROR; \
	cat $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME)/download.log | grep CRITICAL; \

fit_err: _fit _hpc_check
	cat $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME)/fit.e

validate_err: _validate _hpc_check
	cat $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME)/validate.e

test_err: _test _hpc_check
	cat $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME)/test.e

predict_err: _predict _hpc_check
	cat $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME)/predict.e

clean: _experiment_name_check
	echo "Cleaning up logs, checkpoints, and predictions for $(SUNDIAL_EXPERIMENT_NAME).";
	rm -rf $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME);
	rm -rf $(SUNDIAL_BASE_PATH)/checkpoints/$(SUNDIAL_EXPERIMENT_NAME);
	rm -rf $(SUNDIAL_BASE_PATH)/predictions/$(SUNDIAL_EXPERIMENT_NAME);

clean_outs: _experiment_name_check
	echo "Cleaning up outputs for $(SUNDIAL_EXPERIMENT_NAME).";
	rm -rf $(SUNDIAL_BASE_PATH)/logs/*.e
	rm -rf $(SUNDIAL_BASE_PATH)/logs/*.o
	rm -rf $(SUNDIAL_BASE_PATH)/logs/*.log

clean_logs: _experiment_name_check
	echo "Cleaning up logs for $(SUNDIAL_EXPERIMENT_NAME).";
	rm -rf $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME);

clean_samp: _experiment_name_check
	echo "Cleaning up all sample data for $(SUNDIAL_EXPERIMENT_NAME).";
	rm -rf $(SUNDIAL_BASE_PATH)/samples/$(SUNDIAL_EXPERIMENT_NAME);

clean_dnld: _experiment_name_check
	echo "Cleaning up chip data for $(SUNDIAL_EXPERIMENT_NAME).";
	rm -rf $(SUNDIAL_BASE_PATH)/samples/$(SUNDIAL_EXPERIMENT_NAME)/chip_data*;

clean_anno: _experiment_name_check
	echo "Cleaning up annotation data for $(SUNDIAL_EXPERIMENT_NAME).";
	rm -rf $(SUNDIAL_BASE_PATH)/samples/$(SUNDIAL_EXPERIMENT_NAME)/anno_data*;

clean_ckpt: _experiment_name_check
	echo "Cleaning up checkpoints generated for $(SUNDIAL_EXPERIMENT_NAME).";
	rm -rf $(SUNDIAL_BASE_PATH)/checkpoints/$(SUNDIAL_EXPERIMENT_NAME);

clean_pred: _experiment_name_check
	echo "Cleaning up predictions generated for $(SUNDIAL_EXPERIMENT_NAME).";
	rm -rf $(SUNDIAL_BASE_PATH)/predictions/$(SUNDIAL_EXPERIMENT_NAME);

clean_expt: _experiment_name_check
	echo "Cleaning up logs, sample data, checkpoints, and predictions for $(SUNDIAL_EXPERIMENT_NAME).";
	rm -rf $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME);
	rm -rf $(SUNDIAL_BASE_PATH)/samples/$(SUNDIAL_EXPERIMENT_NAME);
	rm -rf $(SUNDIAL_BASE_PATH)/checkpoints/$(SUNDIAL_EXPERIMENT_NAME);
	rm -rf $(SUNDIAL_BASE_PATH)/predictions/$(SUNDIAL_EXPERIMENT_NAME);

clean_nuke: _experiment_name_check
	echo "Cleaning up all data for $(SUNDIAL_EXPERIMENT_NAME).";
	rm -rf $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME);
	rm -rf $(SUNDIAL_BASE_PATH)/samples/$(SUNDIAL_EXPERIMENT_NAME);
	rm -rf $(SUNDIAL_BASE_PATH)/checkpoints/$(SUNDIAL_EXPERIMENT_NAME);
	rm -rf $(SUNDIAL_BASE_PATH)/predictions/$(SUNDIAL_EXPERIMENT_NAME);
	rm -rf $(SUNDIAL_BASE_PATH)/configs/$(SUNDIAL_EXPERIMENT_NAME);

_config_dir_check:
	if [[ ! -d $(SUNDIAL_BASE_PATH)/configs/$(SUNDIAL_EXPERIMENT_NAME) ]]; then \
		echo "No configs found. Please run make setup first..."; \
		exit 1; \
	fi;

_experiment_name_check:
	if [[ -z "$(SUNDIAL_EXPERIMENT_NAME)" ]]; then \
		echo "Please provide sample name and experiment suffix to delete."; \
		exit 1; \
	fi;

_hpc_check:
	if [[ "$(SUNDIAL_PROCESSING)" != hpc ]]; then \
		echo "This method is only available on an HPC with slurm..."; \
		exit 1; \
	fi;

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

_run:
	if [[ "$(SUNDIAL_PROCESSING)" == hpc ]]; then \
		if [[ -z "$(SUNDIAL_NODES)" ]]; then \
			nodes=$(shell sinfo -h -p $(SUNDIAL_PARTITION) -o "%N"); \
		else \
			nodes=$(SUNDIAL_NODES); \
		fi; \
		echo "Running on HPC..."; \
		sbatch \
			--job-name=$(SUNDIAL_METHOD).$(SUNDIAL_EXPERIMENT_NAME) \
			--output=$(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME)/$(SUNDIAL_METHOD).o \
			--error=$(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME)/$(SUNDIAL_METHOD).e \
			--chdir=$(SUNDIAL_BASE_PATH) \
			--partition=$(SUNDIAL_PARTITION) \
			--nodelist=$$nodes \
			--export=ALL \
			$(SUNDIAL_BASE_PATH)/slurm/$(SUNDIAL_METHOD).slurm; \
	else \
		echo "Running on local machine..."; \
		python $(SUNDIAL_BASE_PATH)/src/runner.py; \
	fi;

_watch_std: _experiment_name_check _hpc_check
	echo "Watching $(SUNDIAL_METHOD) for $(SUNDIAL_EXPERIMENT_NAME)...";
	for ((i=0; i<2048; i++)); do \
		tput clear; \
		cat $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME)/$(SUNDIAL_METHOD).e; \
		awk 'END {print}' $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME)/$(SUNDIAL_METHOD).o; \
		sleep 4; \
	done;

_watch_log: _experiment_name_check
	echo "Watching $(SUNDIAL_METHOD) for $(SUNDIAL_EXPERIMENT_NAME)...";
	for ((i=0; i<2048; i++)); do \
		tput clear; \
		tail $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME)/$(SUNDIAL_METHOD).log; \
		sleep 4; \
	done;
