SHELL := /bin/bash

.ONE_SHELL:
.EXPORT_ALL_VARIABLES:
.SILENT:
.PHONY:

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
	@echo "    Methods:"
	@echo "        sample:         Generates chip sample polygons using Google Earth Engine and provided shapefile."
	@echo "        download:       Downloads chip sample images from Google Earth Engine."
	@echo "        fit:            Train model using subset of data from sample and download."
	@echo "        validate:       Validate model subset of using data from sample and download."
	@echo "        test:           Test model using subset of data from sample and download."
	@echo "        predict:        Predict and image from subset of data from sample and download."
	@echo "        clean:          Removes all logs and sample data."
	@echo "        clean_logs:     Removes all logs."
	@echo "        clean_sample:   Removes all sample data."
	@echo "        clean_download: Removes all chip and anno data."
	@echo "        nuke:           Removes all data from run."
	@echo
	@echo "    Variables:"
	@echo "        SUNDIAL_BASE_PATH:           Base path for Sundial scripts. Default: 'shell pwd' of this file"
	@echo "        SUNDIAL_SAMPLE_NAME:         Sample name. Default: ''"
	@echo "        SUNDIAL_EXPERIMENT_SUFFIX:   Sundial experiment name. Default: ''"
	@echo "        SUNDIAL_ENV_NAME:            Sundial environment name. Default: 'sundial'"
	@echo "        SUNDIAL_PROCESSING:          Sundial processing method. Default: 'hpc'"
	@echo


config:
	if [[ -d $(SUNDIAL_BASE_PATH)/configs/$(SUNDIAL_EXPERIMENT_NAME) ]]; then \
		echo "Configs folder found. To restart experiment from scratch, use make nuke..."; \
		exit 1; \
	else \
		echo "Generateing Sundial config files for experiment with values from sample and download..."; \
		python $(SUNDIAL_BASE_PATH)/src/pipeline/settings.py; \
	fi; \


config_check:
	if [[ ! -d $(SUNDIAL_BASE_PATH)/configs/$(SUNDIAL_EXPERIMENT_NAME) ]]; then \
		echo "No configs found. Please run make config first..."; \
		exit 1; \
	fi; \

variable_check:
	if [[ -z "$(SUNDIAL_EXPERIMENT_NAME)" ]]; then \
		echo "Please provide sample name and experiment suffix to delete."; \
		exit 1; \
	fi; \

sample: variable_check config_check
	echo "Retreiving polygon sample from Google Earth Engine. This may take a sec..."; \
	$(eval export SUNDIAL_METHOD=sample)
	$(eval export SUNDIAL_PARTITION=$(X86_PARTITION))
	$(MAKE) -s _run

download: variable_check config_check
	echo "Downloading chip sample from Google Earth Engine. This may take a while..."; \
	$(eval export SUNDIAL_METHOD=download)
	$(eval export SUNDIAL_PARTITION=$(X86_PARTITION))
	$(MAKE) -s _run

fit: variable_check config_check
	echo "Training model... This may take a while..."; \
	$(eval export SUNDIAL_METHOD=fit)
	$(eval export SUNDIAL_PARTITION=$(A64_PARTITION))
	$(MAKE) -s _run

validate: variable_check config_check
	echo "Validating model... Go ahead and hold your breath..."; \
	$(eval export SUNDIAL_METHOD=validate)
	$(eval export SUNDIAL_PARTITION=$(A64_PARTITION))
	$(MAKE) -s _run

test: variable_check config_check
	echo "Testing model... Please check $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME)/test.e..."; \
	$(eval export SUNDIAL_METHOD=test)
	$(eval export SUNDIAL_PARTITION=$(A64_PARTITION))
	$(MAKE) -s _run

predict: variable_check config_check
	echo "Predicting image using model... Lets see if this works!"; \
	$(eval export SUNDIAL_METHOD=predict)
	$(eval export SUNDIAL_PARTITION=$(A64_PARTITION))
	$(MAKE) -s _run

_run:
	if [[ "$(SUNDIAL_PROCESSING)" == hpc ]]; then \
		echo "Running on HPC..."; \
		sbatch $(SUNDIAL_BASE_PATH)/utils/run.slurm; \
	else \
		echo "Running on local machine..."; \
		python $(SUNDIAL_BASE_PATH)/src/runner.py; \
	fi; \

package: variable_check
	echo "Compressing logs for  $(SUNDIAL_EXPERIMENT_NAME) to tar. Tar file will be saved in home directory and overwrite already existing archives."; \
	tar -czvf --overwrite $(HOME)/$(SUNDIAL_EXPERIMENT_NAME).tar.gz $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME); \ 

clean: variable_check
	echo "Cleaning up logs and sample data for $(SUNDIAL_EXPERIMENT_NAME)."; \
	rm -rf $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME); \
	rm -rf $(SUNDIAL_BASE_PATH)/data/samples/$(SUNDIAL_EXPERIMENT_NAME); \

clean_logs: variable_check
	echo "Cleaning up logs for $(SUNDIAL_EXPERIMENT_NAME)."; \
	rm -rf $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME); \

clean_sample: variable_check
	echo "Cleaning up all sample data for $(SUNDIAL_EXPERIMENT_NAME)."; \
	rm -rf $(SUNDIAL_BASE_PATH)/data/samples/$(SUNDIAL_EXPERIMENT_NAME); \

clean_download: variable_check
	echo "Cleaning up chip data and anno data for $(SUNDIAL_EXPERIMENT_NAME)."; \
	rm -rf $(SUNDIAL_BASE_PATH)/data/samples/$(SUNDIAL_EXPERIMENT_NAME)/chip_data*; \
	rm -rf $(SUNDIAL_BASE_PATH)/data/samples/$(SUNDIAL_EXPERIMENT_NAME)/anno_data*; \

clean_predict: variable_check
	echo "Cleaning up predictions generated for  $(SUNDIAL_EXPERIMENT_NAME)."; \
	rm -rf $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME)/predictions

nuke: variable_check
	echo "Deleting logs, sample data, and configs for $(SUNDIAL_EXPERIMENT_NAME)."; \
	rm -rf $(SUNDIAL_BASE_PATH)/logs/$(SUNDIAL_EXPERIMENT_NAME); \
	rm -rf $(SUNDIAL_BASE_PATH)/data/samples/$(SUNDIAL_EXPERIMENT_NAME); \
	rm -rf $(SUNDIAL_BASE_PATH)/configs/$(SUNDIAL_EXPERIMENT_NAME); \
