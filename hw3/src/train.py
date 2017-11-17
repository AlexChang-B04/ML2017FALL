import os
import time
import numpy as np
import argparse

import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import LeakyReLU, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LambdaCallback, TensorBoard
from keras.preprocessing.image import ImageDataGenerator

def load_data(train_data_path):
	#raw_data = pd.read_csv(train_data_path).values
	raw_data = np.loadtxt(train_data_path, dtype=str, delimiter=',', skiprows=1)
	X_all = []
	for feature in raw_data[:,1]:
		X_all.append(np.fromstring(feature, dtype=float, sep=' ').reshape((48,48,1)))
	X_all = np.array(X_all)
	
	Y_all = keras.utils.to_categorical(raw_data[:,0], 7)

	return (X_all / 255, Y_all)

def shuffle_data(X, Y):
	random_order = np.arange(len(X))
	np.random.shuffle(random_order)
	return (X[random_order], Y[random_order])

def split_valid_set(X_all, Y_all, percentage):
	split_pos = int(len(X_all) * percentage)
	X_all, Y_all = shuffle_data(X_all, Y_all)

	return (X_all[split_pos:], Y_all[split_pos:], X_all[0:split_pos], Y_all[0:split_pos])

def train(X_all, Y_all, batch_size, epoch):
	X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, 0.1)

	model = Sequential()
	model.add( Conv2D( filters=64, kernel_size=(5,5), padding='same', activation='selu', kernel_initializer='glorot_normal', input_shape=(48,48,1) ) )
	#model.add( LeakyReLU( alpha=0.1 ) )
	#model.add( BatchNormalization() )
	model.add( MaxPooling2D( pool_size=(2,2), padding='same' ) )
	model.add( Dropout( 0.35 ) )
	
	model.add( Conv2D( filters=128, kernel_size=(3,3), padding='same', activation='selu', kernel_initializer='glorot_normal' ) )
	#model.add( LeakyReLU( alpha=0.1 ) )
	#model.add( BatchNormalization() )
	model.add( MaxPooling2D( pool_size=(2,2), padding='same' ) )
	model.add( Dropout( 0.4 ) )

	model.add( Conv2D( filters=256, kernel_size=(3,3), padding='same', activation='selu', kernel_initializer='glorot_normal' ) )
	#model.add( LeakyReLU( alpha=0.1 ) )
	#model.add( BatchNormalization() )
	model.add( MaxPooling2D( pool_size=(2,2), padding='same' ) )
	model.add( Dropout( 0.45 ) )
	
	model.add( Conv2D( filters=512, kernel_size=(3,3), padding='same', activation='selu', kernel_initializer='glorot_normal' ) )
	#model.add( LeakyReLU( alpha=0.1 ) )
	#model.add( BatchNormalization() )
	model.add( MaxPooling2D( pool_size=(2,2), padding='same' ) )
	model.add( Dropout( 0.5 ) )
	
	model.add( Flatten() )
	
	model.add( Dense ( units=512, activation='selu', kernel_initializer='glorot_normal' ) )
	model.add( BatchNormalization() )
	model.add( Dropout( 0.5 ) )

	model.add( Dense( units=256, activation='selu', kernel_initializer='glorot_normal' ) )
	model.add( BatchNormalization() )
	model.add( Dropout( 0.45 ) )
	'''
	model.add( Dense( units=128, activation='relu', kernel_initializer='glorot_normal' ) )
	model.add( BatchNormalization() )
	model.add( Dropout( 0.5 ) )

	model.add( Dense( units=64, activation='relu', kernel_initializer='glorot_normal' ) )
	model.add( BatchNormalization() )
	model.add( Dropout( 0.5 ) )
	'''
	model.add( Dense( units=7, activation='softmax' ) )

	model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
	model.summary()

	callbacks = []
	#callbacks.append(ModelCheckpoint('check/weights.{epoch:04d}-{val_acc:.4f}.hdf5', monitor='val_acc', save_best_only=True))
	#callbacks.append(LambdaCallback(on_epoch_end=lambda batch, logs: print('\nEpoch[%d] Train-loss=%f Train-accuracy=%f Validation-loss=%f Validation-accuracy=%f' %(batch,logs['loss'], logs['acc'],logs['val_loss'],logs['val_acc']))))
	#tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
	'''
	model.fit(X_train, Y_train,
			  batch_size=batch_size,
			  epochs=epoch,
			  verbose=1,
			  callbacks=callbacks,
			  validation_data=(X_valid, Y_valid))
	'''

	datagen = ImageDataGenerator(rotation_range=5,
								 width_shift_range=0.2,
								 height_shift_range=0.2,
								 zoom_range=[0.95,1.05],
								 horizontal_flip=True)
	model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
						steps_per_epoch=5*len(X_train)//batch_size,
						epochs=epoch,
						callbacks=callbacks,
						validation_data=(X_valid, Y_valid))

	model.save_model(args.save_path)

def main(args):
	X_all, Y_all = load_data(args.train_data_path)
	train(X_all, Y_all, args.batch_size, args.epoch)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='CNN for Multi-class Classification')
	parser.add_argument('--train-data-path', type=str, default='./data/train.csv', dest='train_data_path', help='Path to training data')
	parser.add_argument('--save-path', type=str, default='./model.hdf5', dest='save_path', help='Path to saving model')
	parser.add_argument('--batch-size', type=int, default=128, dest='batch_size', help='Specify the batch size')
	parser.add_argument('--epoch', type=int, default=128, dest='epoch', help='Specify the number of epoch')
	args = parser.parse_args()
	main(args)