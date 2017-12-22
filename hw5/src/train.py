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

class rmseHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.train_loss = []
		self.valid_loss = []

	def on_epoch_end(self, epoch, logs={}):
		self.train_loss.append(logs.get('rmse'))
		self.valid_loss.append(logs.get('val_rmse'))

def argParse():
	parser = argparse.ArgumentParser(description='ML hw5: Matrix Factorization.')
	parser.add_argument('--train_path', default='data/train.csv', type=str)
	parser.add_argument('--test_path', default='data/test.csv', type=str)

	parser.add_argument('--latent_dim', default=16, type=int)
	parser.add_argument('--nb_epoch', default=100, type=int)
	parser.add_argument('--batch_size', default=1024, type=int)
	parser.add_argument('--normalize', default=False, type=bool)

	parser.add_argument('--model_name', default='default', type=str)
	parser.add_argument('--save_dir', default='model/', type=str)
	
	return parser.parse_args()

def readData(train_path, test_path):
	train_data = pd.read_csv(train_path)
	test_data = pd.read_csv(test_path)

	id2user = pd.concat([train_data, test_data])['UserID'].unique()
	user2id = {u: i for i, u in enumerate(id2user)}
	id2movie = pd.concat([train_data, test_data])['MovieID'].unique()
	movie2id = {m: i for i, m in enumerate(id2movie)}

	trainX = train_data[['UserID', 'MovieID']].values
	trainY = train_data['Rating'].values

	trainX[:,0]=[user2id[x] for x in trainX[:,0]]
	trainX[:,1]=[movie2id[x] for x in trainX[:,1]]

	pk.dump(movie2id, open('model/movie2id.pkl', 'wb'))
	pk.dump(user2id, open('model/user2id.pkl', 'wb'))

	return user2id, movie2id, trainX, trainY

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

	save_path = os.path.join(args.save_dir, args.model_name)
	if not os.path.isdir(save_path):
		os.makedirs(save_path)
	model_path = os.path.join(save_path, 'model.h5')
	log_path = os.path.join(save_path, 'log.csv')

	user2id, movie2id, trainX, trainY = readData(args.train_path, args.test_path)
	index = np.arange(len(trainX))
	np.random.shuffle(index)
	trainX, trainY = trainX[index], trainY[index]

	model = buildModel(args, len(user2id), len(movie2id))
	model.compile(optimizer='adam', loss='mse', metrics=[rmse,])
	model.summary()

	checkpoint = ModelCheckpoint(model_path, verbose=0, save_best_only=True, monitor='val_rmse', mode='min')
	history = rmseHistory()

	model.fit([trainX[:,0], trainX[:,1]], trainY,
			  batch_size=args.batch_size,
			  epochs=args.nb_epoch,
			  callbacks=[checkpoint, history],
			  validation_split=0.1)
	np.savetxt(log_path, np.c_[np.arange(len(history.train_loss))+1, history.train_loss, history.valid_loss],
			   fmt=['%d','%f','%f'], delimiter=',', header='Epoch,TrainRMSE,ValidRMSE', comments='')

if __name__ == '__main__':
	main()