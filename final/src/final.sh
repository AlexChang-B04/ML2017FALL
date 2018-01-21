#!/bin/sh
python3 preprocess.py $1
python3 train_word2vec.py
python3 predict.py word_model.w2v $2 $3
