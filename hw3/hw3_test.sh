#!/bin/sh
curl https://www.csie.ntu.edu.tw/~b04902011/weights.0122-0.6969.hdf5 > model.hdf5
python3 src/test.py --test-data-path=$1 --model-path=./model.hdf5 --output-path=$2