#!/bin/bash -l

#SBATCH --export=ALL
#SBATCH --job-name=sundial.run
#SBATCH --cpus-per-task=32

. $(HOME)/.bashrc
conda activate $(SUNDIAL_ENV_NAME)

if [[ -n ${config} ]]; then
    ./run.sh -m $(method) -c $config
else
    ./run.sh -m $(method)
fi