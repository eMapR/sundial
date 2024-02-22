#!/bin/bash -l

#SBATCH --export=ALL
#SBATCH --job-name=sundial.sample
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32

. $(HOME)/.bashrc
conda activate $(SUNDIAL_ENV_NAME)

if [[ -n ${config} ]]; then
    ./sample.sh -c $config
else
    ./sample.sh 
fi
