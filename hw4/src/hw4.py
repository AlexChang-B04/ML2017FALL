import sys, argparse, os
import pickle as pk
import numpy as np
import keras

from keras import regularizers
from keras.models import Model,Sequential
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from gensim.models import Word2Vec

from util import DataManager

parser = argparse.ArgumentParser(description='Sentiment classification')
parser.add_argument('model')
parser.add_argument('action', choices=['train', 'test', 'semi'])

parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--nb_epoch', default=20, type=int)
parser.add_argument('--val_ratio', default=0.1, type=float)
parser.add_argument('--vocab_size', default=20000, type=int)
parser.add_argument('--max_length', default=40, type=int)

parser.add_argument('--model_type', default='RNN', choices=['RNN', 'BOW'])
parser.add_argument('--cell', default='LSTM', choices=['LSTM', 'GRU'])
parser.add_argument('-emb_dim', '--embedding_dim', default=128, type=int)
parser.add_argument('-hid_siz', '--hidden_size', default=512, type=int)
parser.add_argument('--dropout_rate', default=0.3, type=float)
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
parser.add_argument('--threshold', default=0.1, type=float)

parser.add_argument('--train_data', default='data/training_label.txt')
parser.add_argument('--semi_data', default='data/training_nolabel.txt')
parser.add_argument('--test_data', default='data/testing_data.txt')

parser.add_argument('--load_model', default=None)
parser.add_argument('--save_dir', default='model/')
parser.add_argument('--output_file', default='prediction.csv')
parser.add_argument('--word2vec', default=None)	
args = parser.parse_args()

def simpleRNN(args, dm):
	model = Sequential()

	dropout_rate = args.dropout_rate
	
	if args.model_type == 'RNN':
		if args.word2vec is not None:
			word_index = dm.tokenizer.word_index
			print('Found %s unique tokens.' % len(word_index))

			word_model = Word2Vec.load(args.word2vec)
			embedding_index = {}
			for word in word_model.wv.index2word:
				embedding_index[word] = word_model[word]
			print('Found %s word vectors.' % len(embedding_index))

			cnt = 0
			embedding_matrix = np.zeros((args.vocab_size + 1, args.embedding_dim))
			for word, i in word_index.items():
				if i > args.vocab_size:
					continue
				embedding_vector = embedding_index.get(word)
				if embedding_vector is not None:
					cnt += 1
					# words not found in embedding index will be all-zeros.
					embedding_matrix[i] = embedding_vector
			print('initialize %d vector' % cnt)
			model.add( Embedding(args.vocab_size + 1, args.embedding_dim,
								 weights=[embedding_matrix],
								 input_length=args.max_length,
								 trainable=True) )
		else:
			model.add( Embedding(args.vocab_size, args.embedding_dim, input_length=args.max_length, trainable=True) )

		if args.cell == 'LSTM':
			model.add( LSTM(args.hidden_size, return_sequences=True, dropout=dropout_rate) )
			model.add( LSTM(args.hidden_size, return_sequences=False, dropout=dropout_rate) )
		if args.cell == 'GRU':
			model.add( GRU(args.hidden_size, return_sequences=True, dropout=dropout_rate) )
			model.add( GRU(args.hidden_size, return_sequences=False, dropout=dropout_rate) )

		model.add( Dense(args.hidden_size//2, activation='relu', kernel_regularizer=regularizers.l2(0.1)) )
		model.add( Dropout(dropout_rate) )
	
	if args.model_type == 'BOW':
		model.add( Dense(args.hidden_size//2, activation='relu', input_shape=(args.vocab_size,)) )
		model.add( Dropout(dropout_rate) )
		model.add( Dense(args.hidden_size//4, activation='relu') )
		model.add( Dropout(dropout_rate) )
		model.add( Dense(args.hidden_size//8, activation='relu') )
		model.add( Dropout(dropout_rate) )

	model.add( Dense(1, activation='sigmoid') )

	print('compile model...')
	model.compile(loss='binary_crossentropy', optimizer=Adam(lr=args.learning_rate), metrics=['accuracy',])

	return model

def main():
	save_path = os.path.join(args.save_dir, args.model)
	if not os.path.isdir(save_path):
		os.makedirs(save_path)
	if args.load_model is not None:
		load_path = os.path.join(args.save_dir, args.load_model)

	dm = DataManager()
	print('loading data...')
	if args.action == 'train' or args.action == 'semi':
		dm.add_data('train_data', args.train_data, True, False)
		dm.add_data('semi_data', args.semi_data, False, False)
	if args.action == 'test':
		dm.add_data('test_data', args.test_data, False, True)

	print('get tokenizer...')
	if args.load_model is not None:
		dm.load_tokenizer(os.path.join(load_path, 'token.pk'))
	else:
		dm.tokenize(args.vocab_size)

	if args.load_model is None or not os.path.exists(os.path.join(save_path,'token.pk')):
		dm.save_tokenizer(os.path.join(save_path, 'token.pk'))

	if args.model_type == 'RNN':
		dm.to_sequence(args.max_length)
	if args.model_type == 'BOW':
		dm.del_data('semi_data')
		dm.to_bow()

	print('initial model...')
	model = simpleRNN(args, dm)
	model.summary()

	if args.load_model is not None:
		if args.action == 'train':
			print('Warning : load a exist model and keep training')
		path = os.path.join(load_path, 'model.h5')
		if os.path.exists(path):
			print('load model from %s' % path)
			model.load_weights(path)
		else:
			raise ValueError("Can't find the file %s" % path)
	elif args.action == 'test':
		print('Warning : testing without loading any model')

	if args.action == 'train':
		(X, Y), (X_val, Y_val) = dm.split_data('train_data', args.val_ratio)

		earlystopping = EarlyStopping(monitor='val_acc', patience=3, verbose=1, mode='max')
		save_path = os.path.join(save_path, 'model.h5')
		checkpoint = ModelCheckpoint(save_path, verbose=1, save_best_only=True, save_weights_only=True, monitor='val_acc', mode='max')

		history = model.fit(X, Y,
							validation_data=(X_val, Y_val),
							epochs=args.nb_epoch,
							batch_size=args.batch_size,
							callbacks=[earlystopping, checkpoint])
	elif args.action == 'semi':
		(X, Y), (X_val, Y_val) = dm.split_data('train_data', args.val_ratio)

		[semi_all_X] = dm.get_data('semi_data')

		earlystopping = EarlyStopping(monitor='val_acc', patience=3, verbose=1, mode='max')
		save_path = os.path.join(save_path, 'model.h5')
		checkpoint = ModelCheckpoint(save_path, verbose=1, save_best_only=True, save_weights_only=True, monitor='val_acc', mode='max')

		for i in range(10):
			semi_pred = model.predict(semi_all_X, batch_size=1024, verbose=1)
			semi_X, semi_Y = dm.get_semi_data('semi_data', semi_pred, args.threshold)
			semi_X = np.concatenate((semi_X, X))
			semi_Y = np.concatenate((semi_Y, Y))
			print('-- iteration %d  semi_data size: %d' % (i+1, len(semi_X)))

			history = model.fit(semi_X, semi_Y,
								validation_data=(X_val, Y_val),
								epochs=2,
								batch_size=args.batch_size,
								callbacks=[earlystopping, checkpoint])

			if os.path.exists(save_path):
				print('load model from %s' % save_path)
				model.load_weights(save_path)
			else:
				raise ValueError("Can't find the file %s" % save_path)
	elif args.action == 'test':
		[test_X] = dm.get_data('test_data')
		test_pred = model.predict(test_X, batch_size=args.batch_size, verbose=1)
		test_pred = np.squeeze(test_pred)
		#print(test_pred)
		np.savetxt(args.output_file, np.c_[np.arange(len(test_pred)), np.round(test_pred)],
				   fmt='%d', delimiter=',', header='id,label', comments='')
	else:
		print('Error : No specified action')

if __name__ == '__main__':
	main()