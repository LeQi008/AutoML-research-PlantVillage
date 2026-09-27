#!/bin/bash

CONFIG=${1:-configs/baseline.yaml}

python.exe src/main.py --config "$CONFIG" "${@:2}"