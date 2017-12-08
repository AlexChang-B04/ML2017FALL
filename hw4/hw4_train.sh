#!/bin/sh
# python3 src/pretrain.py 5 model/word2vec_5 $1 $2 data/testing_data.txt
python3 src/hw4.py word2vec_double_rnn train \
		-emb_dim 100 \
		-hid_siz 256 \
		--train_data $1 \
		--semi_data $2 \
		--word2vec model/word2vec_5