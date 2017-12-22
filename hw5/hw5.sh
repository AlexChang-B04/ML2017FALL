#!/bin/sh
python3 src/predict.py --test_path $1 --model_name R1_norm2 --normalize True --output_file $2