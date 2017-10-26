#!/bin/sh
if [ ! -f "./model/xgb.model" ]; then
	python3 src/hw2_xgboost.py --train \
					--train-data-path=$3 \
					--train-label-path=$4 \
					--test-data-path=$5 \
					--output-path=$6
fi
python3 src/hw2_xgboost.py --infer \
				--train-data-path=$3 \
				--train-label-path=$4 \
				--test-data-path=$5 \
				--output-path=$6