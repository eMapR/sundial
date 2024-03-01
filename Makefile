SHELL := /bin/bash

.ONE_SHELL:
.SILENT:
.EXPORT_ALL_VARIABLES:

ifndef SUNDIAL_BASE_PATH
SUNDIAL_BASE_PATH := $(shell pwd)
endif
ifndef SUNDIAL_SAMPLE_NAME
SUNDIAL_SAMPLE_NAME :=
endif
ifndef SUNDIAL_EXPERIMENT_SUFFIX
SUNDIAL_EXPERIMENT_SUFFIX :=
SUNDIAL_EXPERIMENT_NAME := $(SUNDIAL_SAMPLE_NAME)
else
SUNDIAL_EXPERIMENT_NAME := $(SUNDIAL_SAMPLE_NAME)_$(SUNDIAL_EXPERIMENT_SUFFIX)
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
	@echo "	To run Sundial, use the following commands:"
	@echo
	@echo "    Sundial Methods:"
	@echo "        sample:      Retrieves chip sample from Google Earth Engine.                                        'Alpha'"
	@echo "        train:       Train Sundial model with optional config.                                              'Alpha'"
	@echo "        validate:    Validate Sundial model with optional config.                                           'Alpha'"
	@echo "        test:        Test Sundial model with optional config.                                               'Alpha'"
	@echo "        predict:     Predict using Sundial. Must provide image path in SUNDIAL_CONFIG.                      'Alpha'"
	@echo "        nuke:        Removes all data from run. Must provide sample name in env var SUNDIAL_SAMPLE_NAME.    'Alpha'"
	@echo
	@echo "    Variables:"
	@echo "        SUNDIAL_BASE_PATH:           Base path for Sundial scripts. Default: 'shell pwd' of this file"
	@echo "        SUNDIAL_SAMPLE_NAME:         Sample name. Default: 'blm_or_wa_bounds'"
	@echo "        SUNDIAL_EXPERIMENT_SUFFIX:   Sundial experiment name. Default: ''"
	@echo "        SUNDIAL_ENV_NAME:            Sundial environment name. Default: 'sundial'"
	@echo "        SUNDIAL_PROCESSING:          Sundial processing method. Default: 'hpc'"
	@echo


config:
	if [[ -d $(SUNDIAL_BASE_PATH)/configs/$(SUNDIAL_EXPERIMENT_NAME) ]]; then \
		echo "Configs folder found. To restart experiment from scratch, use make nuke..."; \
		exit 1; \
	fi; \
	echo "Generateing Sundial config files for experiment with values from sample and download..."; \
	python $(SUNDIAL_BASE_PATH)/src/pipeline/settings.py; \

sample:
	if [[ ! -d $(SUNDIAL_BASE_PATH)/configs/$(SUNDIAL_EXPERIMENT_NAME) ]]; then \
		echo "No configs found. Please run make config first..."; \
		exit 1; \
	fi; \
	echo "Retreiving polygon sample from Google Earth Engine. This may take a sec..."; \
	$(eval export SUNDIAL_METHOD=sample)

	if [[ $(SUNDIAL_PROCESSING) == hpc ]]; then \
		echo "Submitting job to HPC..."; \
		sbatch \
			--output=$(SUNDIAL_BASE_PATH)/logs/sample.o \
			--error=$(SUNDIAL_BASE_PATH)/logs/sample.e \
			--partition=$(X86_PARTITION) \
			--chdir=$(SUNDIAL_BASE_PATH) \
			--export=ALL \
			$(SUNDIAL_BASE_PATH)/utils/sample.slurm; \
	else\
		echo "Running on local machine..."; \
		python $(SUNDIAL_BASE_PATH)/src/pipeline/sampler.py; \
	fi; \

download:
	if [[ ! -d $(SUNDIAL_BASE_PATH)/configs/$(SUNDIAL_EXPERIMENT_NAME) ]]; then \
		echo "No configs found. Please run make config first..."; \
		exit 1; \
	fi; \
	echo "Downloading chip sample from Google Earth Engine. This may take a while..."; \
	$(eval export SUNDIAL_METHOD=download)

	if [[ "$(SUNDIAL_PROCESSING)" == hpc ]]; then \
		sbatch \
			--output=$(SUNDIAL_BASE_PATH)/logs/downloade.o \
			--error=$(SUNDIAL_BASE_PATH)/logs/downloade.e \
			--partition=$(X86_PARTITION) \
			--chdir=$(SUNDIAL_BASE_PATH) \
			--export=ALL \
			$(SUNDIAL_BASE_PATH)/utils/download.slurm; \
	else\
		echo "Running on local machine..."; \
		python $(SUNDIAL_BASE_PATH)/src/pipeline/downloader.py; \
	fi; \

fit:
	if [[ ! -d $(SUNDIAL_BASE_PATH)/configs/$(SUNDIAL_EXPERIMENT_NAME) ]]; then \
		echo "No configs found. Please run make config first..."; \
		exit 1; \
	fi; \
	echo "Training model... This may take a while..."; \
	$(eval export SUNDIAL_METHOD=fit)

	if [[ "$(SUNDIAL_PROCESSING)" == hpc ]]; then \
		sbatch \
			--output=$(SUNDIAL_BASE_PATH)/logs/fit.o \
			--error=$(SUNDIAL_BASE_PATH)/logs/fit.e \
			--partition=$(A64_PARTITION) \
			--chdir=$(SUNDIAL_BASE_PATH) \
			--export=ALL \
			$(SUNDIAL_BASE_PATH)/utils/run.slurm; \
	else \
		echo "Running on local machine..."; \
		python $(SUNDIAL_BASE_PATH)/src/runner.py; \
	fi; \

validate:
	if [[ ! -d $(SUNDIAL_BASE_PATH)/configs/$(SUNDIAL_EXPERIMENT_NAME) ]]; then \
		echo "No configs found. Please run make config first..."; \
		exit 1; \
	fi; \
	echo "Validating model... Go ahead and hold your breath..."; \
	$(eval export SUNDIAL_METHOD=validate)

	if [[ "$(SUNDIAL_PROCESSING)" == hpc ]]; then \
		sbatch \
			--output=$(SUNDIAL_BASE_PATH)/logs/validate.o \
			--error=$(SUNDIAL_BASE_PATH)/logs/validate.e \
			--partition=$(A64_PARTITION) \
			--chdir=$(SUNDIAL_BASE_PATH) \
			--export=ALL \
			$(SUNDIAL_BASE_PATH)/utils/run.slurm; \
	else \
		echo "Running on local machine..."; \
		python $(SUNDIAL_BASE_PATH)/src/runner.py; \
	fi; \

test:
	if [[ ! -d $(SUNDIAL_BASE_PATH)/configs/$(SUNDIAL_EXPERIMENT_NAME) ]]; then \
		echo "No configs found. Please run make config first..."; \
		exit 1; \
	fi; \
	echo "Testing model... Please check $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME)/test.e..."; \
	$(eval export SUNDIAL_METHOD=test)

	if [[ "$(SUNDIAL_PROCESSING)" == hpc ]]; then \
		sbatch \
			--output=$(SUNDIAL_BASE_PATH)/logs/test.o \
			--error=$(SUNDIAL_BASE_PATH)/logs/test.e \
			--partition=$(A64_PARTITION) \
			--chdir=$(SUNDIAL_BASE_PATH) \
			--export=ALL \
			$(SUNDIAL_BASE_PATH)/utils/run.slurm; \
	else \
		echo "Running on local machine..."; \
		python $(SUNDIAL_BASE_PATH)/src/runner.py; \
	fi; \


predict:
	if [[ ! -d $(SUNDIAL_BASE_PATH)/configs/$(SUNDIAL_EXPERIMENT_NAME) ]]; then \
		echo "No configs found. Please run make config first..."; \
		exit 1; \
	fi; \
	echo "Predicting image using model... Lets see if this works!"; \
	$(eval export SUNDIAL_METHOD=predict)

	if [[ "$(SUNDIAL_PROCESSING)" == hpc ]]; then \
		sbatch \
			--output=$(SUNDIAL_BASE_PATH)/logs/predict.o \
			--error=$(SUNDIAL_BASE_PATH)/logs/predict.e \
			--partition=$(A64_PARTITION) \
			--chdir=$(SUNDIAL_BASE_PATH) \
			--export=ALL \
			$(SUNDIAL_BASE_PATH)/utils/run.slurm; \
	else \
		echo "Running on local machine..."; \
		python $(SUNDIAL_BASE_PATH)/src/runner.py; \
	fi; \

nuke:
	if [[ -z $(SUNDIAL_EXPERIMENT_NAME) ]]; then \
		echo "Please provide sample name and experiment suffix to delete."; \
		exit 1; \
	fi; \
	echo "Deleting logs, meta data, and zarr chips datasets for $(SUNDIAL_EXPERIMENT_NAME)."; \
	rm -rf $(SUNDIAL_BASE_PATH)/data/samples/$(SUNDIAL_EXPERIMENT_NAME); \
	rm -rf $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME); \
	rm -rf $(SUNDIAL_BASE_PATH)/configs/$(SUNDIAL_EXPERIMENT_NAME); \
