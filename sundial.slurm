#!/bin/bash -l

source ${HOME}/${SUNDIAL_ENV_NAME}_${ARCH}/bin/activate
srun python -u ${SUNDIAL_BASE_PATH}/sundial/runner.py
