#!/bin/sh
# python3 src/pretrain.py 5 model/word2vec_5 data/training_label.txt data/training_nolabel.txt $1
# python3 src/hw4.py word2vec_double_rnn test \
# 		-emb_dim 100 \
# 		--test_data $1 \
# 		--load_model word2vec_double_rnn \
# 		--output_file $2 \
# 		--word2vec model/word2vec_5
python3 src/ensemble.py \
		--test_data $1 \
		--output_file $2
