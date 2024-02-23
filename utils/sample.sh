#!/bin/bash

usage() {
    echo "Usage: $0 [-h|--help] [-c|--config <file>] [-d|--download]"
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -c, --config        Specify a YAML configuration file"
    echo "  -d, --download      Specify whether to download or not (default: true)"
}

# TODO: Add conda environment checks
download=true

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
        -d|--download)
            if [[ "$2" != true && "$2" != false ]]; then
                echo "Error: Invalid value for download argument. Must be 'true' or 'false'."
                exit 1
            fi
            download="$2"
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
    python $SUNDIAL_BASE_PATH/src/sampler.py -c "$config_file"
    if [[ -d "$SUNDIAL_BASE_PATH/data/samples/$SUNDIAL_SAMPLE_NAME/meta_data.zarr" ]]; then
        if [[ "$download" == true ]]; then
            python $SUNDIAL_BASE_PATH/src/downloader.py -c "$config_file"
        fi
    fi
else
    python $SUNDIAL_BASE_PATH/src/sampler.py
    if [[ -d "$SUNDIAL_BASE_PATH/data/samples/$SUNDIAL_SAMPLE_NAME/meta_data.zarr" ]]; then
        if [[ "$download" == true ]]; then
            python $SUNDIAL_BASE_PATH/src/downloader.py
        fi
    fi
fi