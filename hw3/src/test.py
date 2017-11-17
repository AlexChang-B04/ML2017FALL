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

def load_data(test_data_path):
	#raw_data = pd.read_csv(test_data_path).values
	raw_data = np.loadtxt(test_data_path, dtype=str, delimiter=',', skiprows=1)
	X_test = []
	for feature in raw_data[:,1]:
		X_test.append(np.fromstring(feature, dtype=float, sep=' ').reshape((48,48,1)))
	X_test = np.array(X_test)

	return (X_test / 255)

def infer(X_test, model_path, output_path):
	model = load_model(model_path)
	Y_test = model.predict(X_test, verbose=1)
	#print(Y_test)
	Y_test = np.argmax(Y_test, axis=1)
	np.savetxt(output_path, np.c_[np.arange(len(Y_test)), Y_test], fmt='%d', delimiter=',', header='id,label', comments='')

def main(args):
	X_test = load_data(args.test_data_path)
	infer(X_test, args.model_path, args.output_path)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='CNN for Multi-class Classification')
	parser.add_argument('--test-data-path', type=str, default='./data/test.csv', dest='test_data_path', help='Path to testing data')
	parser.add_argument('--model-path', type=str, default='./model.hdf5', dest='model_path', help='Path to model')
	parser.add_argument('--output-path', type=str, default='./prediction.csv', dest='output_path', help='Path to outputing prediction')
	parser.add_argument('--batch-size', type=int, default=128, dest='batch_size', help='Specify the batch size')
	parser.add_argument('--epoch', type=int, default=128, dest='epoch', help='Specify the number of epoch')
	args = parser.parse_args()
	main(args)