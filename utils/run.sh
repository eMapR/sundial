#!/bin/bash

usage() {
    echo "Usage: $0 [-h|--help] [-c|--config <file>] [-d|--method]"
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -c, --config        Specify a YAML configuration file"
    echo "  -m, --method        Specify what method to run model (default: fit)"
}

# TODO: Add conda environment checks
method=fit

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            usage
            exit 0
            ;;
        -c|--config)
            if [[ ! -f "$2" ]]; then
                echo "Error: Config file '$2' does not exist."
                exit 1
            fi
            if [[ ! -s "$2" ]]; then
                echo "Error: Config file '$2' is empty."
                exit 1
            fi
            config_file="$2"
            shift
            shift
            ;;
        -m|--method)
            method="$2"
            if [[ "$method" != "fit" && "$method" != "validate" && "$method" != "test" && "$method" != "predict" ]]; then
                echo "Error: Invalid method '$method'. Must be either fit, validate, test, or predict."
                exit 1
            fi
            shift
            shift
            ;;
        *)
            echo "Unknown option: $key"
            usage
            exit 1
            ;;
    esac
done


if [[ -n "$config_file" ]]; then
    python $SUNDIAL_BASE_PATH/src/runner.py $method -c "$config_file" 
else
    python $SUNDIAL_BASE_PATH/src/runner.py $method
fi