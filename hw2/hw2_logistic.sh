#!/bin/sh
if [ ! -f "./params/w" ] || [ ! -f "./params/b" ]; then
	python3 src/hw2.py --train --logistic \
			--train-data-path=$3 \
			--train-label-path=$4 \
			--test-data-path=$5 \
			--output-path=$6
fi
python3 src/hw2.py --infer --logistic \
		--train-data-path=$3 \
		--train-label-path=$4 \
		--test-data-path=$5 \
		--output-path=$6