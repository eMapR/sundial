SHELL := /bin/bash
ifndef SUNDIAL_BASE_PATH
	SUNDIAL_BASE_PATH := $(shell pwd)
	export SUNDIAL_BASE_PATH
endif
ifndef SUNDIAL_SAMPLE_NAME
	SUNDIAL_SAMPLE_NAME := blm_or_wa_bounds
	export SUNDIAL_SAMPLE_NAME
endif
ifndef SUNDIAL_EXPERIMENT_SUFFIX
	SUNDIAL_EXPERIMENT_SUFFIX := 
	export SUNDIAL_EXPERIMENT_SUFFIX
endif
ifndef SUNDIAL_PROCESSING
	SUNDIAL_PROCESSING := hpc
	export SUNDIAL_PROCESSING
endif
ifndef SUNDIAL_ENV_NAME 
	SUNDIAL_ENV_NAME := sundial
	export SUNDIAL_ENV_NAME
endif

.SILENT: sample download fit validate test predict nuke

default:
	@echo "Welcome to Sundial!"
	@echo
	@echo "	To run Sundial, use the following commands:"
	@echo
	@echo "    Sundial Methods:"
	@echo "        sample:      Retrieves chip sample from Google Earth Engine.                                        'Alpha'"
	@echo "        train:       Train Sundial model with optional config.                                              'IN-DEVELOPMENT'"
	@echo "        validate:    Validate Sundial model with optional config.                                           'IN-DEVELOPMENT'"
	@echo "        test:        Test Sundial model with optional config.                                               'IN-DEVELOPMENT'"
	@echo "        predict:     Predict using Sundial. Must provide image path in SUNDIAL_CONFIG.                      'IN-DEVELOPMENT'"
	@echo "        nuke:        Removes all data from run. Must provide sample name in env var SUNDIAL_SAMPLE_NAME.    'IN-DEVELOPMENT'"
	@echo
	@echo "    Variables:"
	@echo "        SUNDIAL_BASE_PATH:           Base path for Sundial scripts. Default: 'shell pwd' of this file"
	@echo "        SUNDIAL_SAMPLE_NAME:         Sample name. Default: 'blm_or_wa_bounds'"
	@echo "        SUNDIAL_EXPERIMENT_SUFFIX:   Sundial experiment name. Default: ''"
	@echo "        SUNDIAL_ENV_NAME:            Sundial environment name. Default: 'sundial'"
	@echo "        SUNDIAL_PROCESSING:          Sundial processing method. Default: 'hpc'"
	@echo "        SUNDIAL_CONFIG:              Sundial config file path. Default: 'src/settings.py'"
	@echo

sample:
	echo "Retreiving polygon sample from Google Earth Engine. This may take a sec..."; \
	if [[ $(SUNDIAL_PROCESSING) == hpc ]]; then \
		echo "Submitting job to HPC..."; \
		sbatch \
			--output=$(SUNDIAL_BASE_PATH)/data/logs/sundial.sample.o \
			--error=$(SUNDIAL_BASE_PATH)/data/logs/sundial.sample.e \
			--partition $(X86_PARTITION) \
			--chdir=$(SUNDIAL_BASE_PATH) \
			--export=SUNDIAL_CONFIG="$(SUNDIAL_CONFIG)",SUNDIAL_BASE_PATH="$(SUNDIAL_BASE_PATH)",SUNDIAL_ENV_NAME="$(SUNDIAL_ENV_NAME)",SUNDIAL_SAMPLE_NAME="$(SUNDIAL_SAMPLE_NAME)",SUNDIAL_EXPERIMENT_SUFFIX="$(SUNDIAL_EXPERIMENT_SUFFIX)" \
			$(SUNDIAL_BASE_PATH)/utils/sample.slurm; \
	else\
		echo "Running on local machine..."; \
		if [[ -n "$(SUNDIAL_CONFIG)" ]]; then \
			python $(SUNDIAL_BASE_PATH)/src/utils/sampler.py --config_path $(SUNDIAL_CONFIG); \
		else\
			python $(SUNDIAL_BASE_PATH)/src/utils/sampler.py; \
		fi \
	fi \

download:
	echo "Downloading chip sample from Google Earth Engine. This may take a while..."; \
	if [[ $(SUNDIAL_PROCESSING) == hpc ]]; then \
		echo "Submitting job to HPC..."; \
		sbatch \
			--output=$(SUNDIAL_BASE_PATH)/data/logs/sundial.sample.o \
			--error=$(SUNDIAL_BASE_PATH)/data/logs/sundial.sample.e \
			--partition $(X86_PARTITION) \
			--chdir=$(SUNDIAL_BASE_PATH) \
			--export=SUNDIAL_CONFIG="$(SUNDIAL_CONFIG)",SUNDIAL_BASE_PATH="$(SUNDIAL_BASE_PATH)",SUNDIAL_ENV_NAME="$(SUNDIAL_ENV_NAME)",SUNDIAL_SAMPLE_NAME="$(SUNDIAL_SAMPLE_NAME)",SUNDIAL_EXPERIMENT_SUFFIX="$(SUNDIAL_EXPERIMENT_SUFFIX)" \
			$(SUNDIAL_BASE_PATH)/utils/download.slurm; \
	else\
		echo "Running on local machine..."; \
		if [[ -n "$(SUNDIAL_CONFIG)" ]]; then \
			python $(SUNDIAL_BASE_PATH)/src/utils/downloader.py --config_path $(SUNDIAL_CONFIG); \
		else\
			python $(SUNDIAL_BASE_PATH)/src/utils/downloader.py; \
		fi \
	fi \

fit:
	echo "Training Sundial... This may take a while..."; \
	if [[ $(SUNDIAL_PROCESSING) == hpc ]]; then \
		echo "Submitting job to HPC..."; \
		sbatch \
			--output=$(SUNDIAL_BASE_PATH)/data/logs/sundial.train.o \
			--error=$(SUNDIAL_BASE_PATH)/data/logs/sundial.train.e \
			--partition $(A64_PARTITION) \
			--chdir=$(SUNDIAL_BASE_PATH) \
			--export=METHOD="fit",SUNDIAL_CONFIG="$(SUNDIAL_CONFIG)",SUNDIAL_BASE_PATH="$(SUNDIAL_BASE_PATH)",SUNDIAL_ENV_NAME="$(SUNDIAL_ENV_NAME)",SUNDIAL_SAMPLE_NAME="$(SUNDIAL_SAMPLE_NAME)",SUNDIAL_EXPERIMENT_SUFFIX="$(SUNDIAL_EXPERIMENT_SUFFIX)" \
			$(SUNDIAL_BASE_PATH)/utils/run.slurm; \
	else \
		echo "Running on local machine..."; \
		if [[ -n "$(SUNDIAL_CONFIG)" ]]; then \
			python $(SUNDIAL_BASE_PATH)/src/runner.py fit --config $(SUNDIAL_CONFIG); \
		else \
			python $(SUNDIAL_BASE_PATH)/src/runner.py fit; \
		fi \
	fi \

validate:
	echo "Validating Sundial... Go ahead and hold your breath..."; \
	if [[ $(SUNDIAL_PROCESSING) == hpc ]]; then \
		echo "Submitting job to HPC..."; \
		sbatch \
			--output=$(SUNDIAL_BASE_PATH)/data/logs/sundial.validate.o \
			--error=$(SUNDIAL_BASE_PATH)/data/logs/sundial.validate.e \
			--partition $(A64_PARTITION) \
			--chdir=$(SUNDIAL_BASE_PATH) \
			--export=METHOD="validate",SUNDIAL_CONFIG="$(SUNDIAL_CONFIG)",SUNDIAL_BASE_PATH="$(SUNDIAL_BASE_PATH)",SUNDIAL_ENV_NAME="$(SUNDIAL_ENV_NAME)",SUNDIAL_SAMPLE_NAME="$(SUNDIAL_SAMPLE_NAME)",SUNDIAL_EXPERIMENT_SUFFIX="$(SUNDIAL_EXPERIMENT_SUFFIX)" \
			$(SUNDIAL_BASE_PATH)/utils/run.slurm; \
	else \
		echo "Running on local machine..."; \
		if [[ -n "$(SUNDIAL_CONFIG)" ]]; then \
			python $(SUNDIAL_BASE_PATH)/src/runner.py validate --config $(SUNDIAL_CONFIG); \
		else \
			python $(SUNDIAL_BASE_PATH)/src/runner.py validate; \
		fi \
	fi \

test:
	echo "Testing Sundial... Please check $(SUNDIAL_BASE_PATH)/data/logs/sundial.test.e..."; \
	if [[ $(SUNDIAL_PROCESSING) == hpc ]]; then \
		echo "Submitting job to HPC..."; \
		sbatch \
			--output=$(SUNDIAL_BASE_PATH)/data/logs/sundial.test.o \
			--error=$(SUNDIAL_BASE_PATH)/data/logs/sundial.test.e \
			--partition $(A64_PARTITION) \
			--chdir=$(SUNDIAL_BASE_PATH) \
			--export=METHOD="test",SUNDIAL_CONFIG="$(SUNDIAL_CONFIG)",SUNDIAL_BASE_PATH="$(SUNDIAL_BASE_PATH)",SUNDIAL_ENV_NAME="$(SUNDIAL_ENV_NAME)",SUNDIAL_SAMPLE_NAME="$(SUNDIAL_SAMPLE_NAME)",SUNDIAL_EXPERIMENT_SUFFIX="$(SUNDIAL_EXPERIMENT_SUFFIX)" \
			$(SUNDIAL_BASE_PATH)/utils/run.slurm; \
	else \
		echo "Running on local machine..."; \
		if [[ -n "$(SUNDIAL_CONFIG)" ]]; then \
			python $(SUNDIAL_BASE_PATH)/src/runner.py test --config $(SUNDIAL_CONFIG); \
		else \
			python $(SUNDIAL_BASE_PATH)/src/runner.py test; \
		fi \
	fi \


predict:
	echo "Predicting image using Sundial... Lets see if this works!"; \
	if [[ $(SUNDIAL_PROCESSING) == hpc ]]; then \
		echo "Submitting job to HPC..."; \
		sbatch \
			--output=$(SUNDIAL_BASE_PATH)/data/logs/sundial.test.o \
			--error=$(SUNDIAL_BASE_PATH)/data/logs/sundial.test.e \
			--partition $(A64_PARTITION) \
			--chdir=$(SUNDIAL_BASE_PATH) \
			--export=METHOD="predict",SUNDIAL_CONFIG="$(SUNDIAL_CONFIG)",SUNDIAL_BASE_PATH="$(SUNDIAL_BASE_PATH)",SUNDIAL_ENV_NAME="$(SUNDIAL_ENV_NAME)"SUNDIAL_SAMPLE_NAME="$(SUNDIAL_SAMPLE_NAME)",SUNDIAL_EXPERIMENT_SUFFIX="$(SUNDIAL_EXPERIMENT_SUFFIX)" \
			$(SUNDIAL_BASE_PATH)/utils/run.slurm; \
	else \
		echo "Running on local machine..."; \
		if [[ -n "$(SUNDIAL_CONFIG)" ]]; then \
			python $(SUNDIAL_BASE_PATH)/src/runner.py predict --config $(SUNDIAL_CONFIG); \
		else \
			python $(SUNDIAL_BASE_PATH)/src/runner.py predict; \
		fi \
	fi \

nuke:
	echo "Deleting meta data and ZARR chips datasets."; \
	rm -rf $(SUNDIAL_BASE_PATH)/data/samples/$(SUNDIAL_SAMPLE_NAME); \
