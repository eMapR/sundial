SHELL := /bin/bash

.ONE_SHELL:
.SILENT:
.EXPORT_ALL_VARIABLES:

ifndef SUNDIAL_BASE_PATH
SUNDIAL_BASE_PATH := $(shell pwd)
endif
ifndef SUNDIAL_SAMPLE_NAME
SUNDIAL_SAMPLE_NAME := blm_or_wa_bounds
endif
ifndef SUNDIAL_EXPERIMENT_SUFFIX
SUNDIAL_EXPERIMENT_SUFFIX :=
endif
ifndef SUNDIAL_CKPT_PATH
SUNDIAL_CKPT_PATH := null
endif
ifndef SUNDIAL_PROCESSING
SUNDIAL_PROCESSING := hpc
endif
ifndef SUNDIAL_ENV_NAME 
SUNDIAL_ENV_NAME := sundial
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
	@echo "        SUNDIAL_CONFIG:              Sundial config file path. Default: 'src/settings.py'"
	@echo "        SUNDIAL_CKPT_PATH:           Sundial checkpoint path. Default: 'null'"
	@echo

tester:
	$(SUNDIAL_BASE_PATH)/utils/test.sh

sample:
	echo "Retreiving polygon sample from Google Earth Engine. This may take a sec..."; \
	if [[ $(SUNDIAL_PROCESSING) == hpc ]]; then \
		echo "Submitting job to HPC..."; \
		sbatch \
			--output=$(SUNDIAL_BASE_PATH)/logs/sundial.sampler.o \
			--error=$(SUNDIAL_BASE_PATH)/logs/sundial.sampler.e \
			--partition $(X86_PARTITION) \
			--chdir=$(SUNDIAL_BASE_PATH) \
			--export=ALL \
			$(SUNDIAL_BASE_PATH)/utils/sample.slurm; \
	else\
		echo "Running on local machine..."; \
		python $(SUNDIAL_BASE_PATH)/src/pipeline/sampler.py --config $(SUNDIAL_CONFIG); \
	fi \

download:
	echo "Downloading chip sample from Google Earth Engine. This may take a while..."; \
	if [[ "$(SUNDIAL_PROCESSING)" == hpc ]]; then \
		echo "Submitting job to HPC..."; \
		sbatch \
			--output=$(SUNDIAL_BASE_PATH)/logs/sundial.downloader.o \
			--error=$(SUNDIAL_BASE_PATH)/logs/sundial.downloader.e \
			--partition $(X86_PARTITION) \
			--chdir=$(SUNDIAL_BASE_PATH) \
			--export=ALL \
			$(SUNDIAL_BASE_PATH)/utils/download.slurm; \
	else\
		echo "Running on local machine..."; \
		python $(SUNDIAL_BASE_PATH)/src/pipeline/downloader.py --config $(SUNDIAL_CONFIG); \
	fi \

fit:
	echo "Training Sundial... This may take a while..."; \
	if [[ "$(SUNDIAL_PROCESSING)" == hpc ]]; then \
		echo "Submitting job to HPC..."; \
		sbatch \
			--output=$(SUNDIAL_BASE_PATH)/logs/sundial.train.o \
			--error=$(SUNDIAL_BASE_PATH)/logs/sundial.train.e \
			--partition $(A64_PARTITION) \
			--chdir=$(SUNDIAL_BASE_PATH) \
			--export=ALL \
			$(SUNDIAL_BASE_PATH)/utils/run.slurm; \
	else \
		echo "Running on local machine..."; \
		python $(SUNDIAL_BASE_PATH)/src/runner.py fit --config $(SUNDIAL_CONFIG) --ckpt_path $(SUNDIAL_CKPT_PATH); \
	fi \

validate:
	echo "Validating Sundial... Go ahead and hold your breath..."; \
	if [[ "$(SUNDIAL_PROCESSING)" == hpc ]]; then \
		echo "Submitting job to HPC..."; \
		sbatch \
			--output=$(SUNDIAL_BASE_PATH)/logs/sundial.validate.o \
			--error=$(SUNDIAL_BASE_PATH)/logs/sundial.validate.e \
			--partition $(A64_PARTITION) \
			--chdir=$(SUNDIAL_BASE_PATH) \
			--export=ALL \
			$(SUNDIAL_BASE_PATH)/utils/run.slurm; \
	else \
		echo "Running on local machine..."; \
		python $(SUNDIAL_BASE_PATH)/src/runner.py validate --config $(SUNDIAL_CONFIG) --ckpt_path $(SUNDIAL_CKPT_PATH); \
	fi \

test:
	echo "Testing Sundial... Please check $(SUNDIAL_BASE_PATH)/logs/sundial.test.e..."; \
	if [[ "$(SUNDIAL_PROCESSING)" == hpc ]]; then \
		echo "Submitting job to HPC..."; \
		sbatch \
			--output=$(SUNDIAL_BASE_PATH)/logs/sundial.test.o \
			--error=$(SUNDIAL_BASE_PATH)/logs/sundial.test.e \
			--partition $(A64_PARTITION) \
			--chdir=$(SUNDIAL_BASE_PATH) \
			--export=ALL \
			$(SUNDIAL_BASE_PATH)/utils/run.slurm; \
	else \
		echo "Running on local machine..."; \
		python $(SUNDIAL_BASE_PATH)/src/runner.py test --config $(SUNDIAL_CONFIG) --ckpt_path $(SUNDIAL_CKPT_PATH); \
	fi \


predict:
	echo "Predicting image using Sundial... Lets see if this works!"; \
	if [[ "$(SUNDIAL_PROCESSING)" == hpc ]]; then \
		echo "Submitting job to HPC..."; \
		sbatch \
			--output=$(SUNDIAL_BASE_PATH)/logs/sundial.test.o \
			--error=$(SUNDIAL_BASE_PATH)/logs/sundial.test.e \
			--partition $(A64_PARTITION) \
			--chdir=$(SUNDIAL_BASE_PATH) \
			--export=ALL \
			$(SUNDIAL_BASE_PATH)/utils/run.slurm; \
	else \
		echo "Running on local machine..."; \
		python $(SUNDIAL_BASE_PATH)/src/runner.py predict --config $(SUNDIAL_CONFIG) --ckpt_path $(SUNDIAL_CKPT_PATH); \
	fi \

nuke:
	echo "Deleting meta data and ZARR chips datasets."; \
	rm -rf $(SUNDIAL_BASE_PATH)/data/samples/$(SUNDIAL_SAMPLE_NAME); \
