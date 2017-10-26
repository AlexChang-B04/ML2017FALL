#!/bin/sh
if [ ! -f "./params/C0" ] || [ ! -f "./params/C1" ] || [ ! -f "./params/mu0" ] || \
   [ ! -f "./params/mu1" ] || [ ! -f "./params/sigma" ]; then
	python3 src/hw2.py --train --generative \
					   --train-data-path=$3 \
					   --train-label-path=$4 \
					   --test-data-path=$5 \
					   --output-path=$6
fi
python3 src/hw2.py --infer --generative \
				   --train-data-path=$3 \
				   --train-label-path=$4 \
				   --test-data-path=$5 \
				   --output-path=$6