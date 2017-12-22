import os
import pickle as pk
import argparse

import numpy as np
import pandas as pd

import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dropout, Dot, Add, Lambda
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint

def argParse():
	parser = argparse.ArgumentParser(description='ML hw5: Matrix Factorization.')
	parser.add_argument('--test_path', default='data/test.csv', type=str)

	parser.add_argument('--latent_dim', default=16, type=int)
	parser.add_argument('--batch_size', default=4096, type=int)
	parser.add_argument('--normalize', default=False, type=bool)

	parser.add_argument('--model_name', default='default', type=str)
	parser.add_argument('--load_dir', default='model/', type=str)
	parser.add_argument('--output_file', default='pred.csv', type=str)
	
	return parser.parse_args()

def readData(test_path, user2id_path, movie2id_path):
	test_data = pd.read_csv(test_path)
	user2id = pk.load(open(user2id_path, 'rb'))
	movie2id = pk.load(open(movie2id_path, 'rb'))

	testX = test_data[['UserID', 'MovieID']].values
	testX[:,0]=[user2id[x] for x in testX[:,0]]
	testX[:,1]=[movie2id[x] for x in testX[:,1]]

	return user2id, movie2id, testX

def buildModel(args, nb_users, nb_movies):
	input_u = Input( shape=(1,) )
	embed_u = Embedding( input_dim=nb_users, output_dim=args.latent_dim, embeddings_regularizer=l2(0.00001) )(input_u)
	embed_u = Flatten()(embed_u)
	embed_u = Dropout( 0.1 )(embed_u)
	bias_u = Embedding( input_dim=nb_users, output_dim=1, embeddings_regularizer=l2(0.00001) )(input_u)
	bias_u = Flatten()(bias_u)

	input_m = Input( shape=(1,) )
	embed_m = Embedding( input_dim=nb_movies, output_dim=args.latent_dim, embeddings_regularizer=l2(0.00001) )(input_m)
	embed_m = Flatten()(embed_m)
	embed_m = Dropout( 0.1 )(embed_m)
	bias_m = Embedding( input_dim=nb_movies, output_dim=1, embeddings_regularizer=l2(0.00001) )(input_m)
	bias_m = Flatten()(bias_m)

	dot = Dot( axes=1 )([embed_u, embed_m])
	output = Add()([dot, bias_u, bias_m])

	if args.normalize:
		output = Lambda( lambda x: x * K.constant(1.1168976611462060, dtype=K.floatx()) )(output)
		output = Lambda( lambda x: x + K.constant(3.5817120860388076, dtype=K.floatx()) )(output)

	return Model([input_u, input_m], output)

def rmse(y_true, y_pred):
	y_pred = K.clip(y_pred, 1, 5)
	return K.sqrt(K.mean(K.square(y_true - y_pred)))

def main():
	args = argParse()

	load_path = os.path.join(args.load_dir, args.model_name)
	user2id_path = os.path.join(args.load_dir, 'user2id.pkl')
	movie2id_path = os.path.join(args.load_dir, 'movie2id.pkl')
	if not os.path.isdir(load_path):
		raise ValueError('Can\'t find the directory %s' % load_path)
	model_path = os.path.join(load_path, 'model.h5')

	user2id, movie2id, testX = readData(args.test_path, user2id_path, movie2id_path)

	model = buildModel(args, len(user2id), len(movie2id))
	# model.compile(optimizer='adam', loss='mse', metrics=[rmse,])
	model.summary()
	model.load_weights(model_path)

	pred = model.predict([testX[:,0], testX[:,1]], batch_size=args.batch_size, verbose=1)
	np.savetxt(args.output_file, np.c_[np.arange(len(pred))+1, np.clip(pred, 1, 5)],
			   fmt=['%d','%f'], delimiter=',', header='TestDataID,Rating', comments='')

if __name__ == '__main__':
	main()