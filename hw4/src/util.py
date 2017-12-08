import os
import pickle as pk
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class DataManager:
	def __init__(self):
		self.data = {}

	def add_data(self, name, data_path, with_label, is_test):
		print('read data from %s...' % data_path)
		X, Y = [], []
		with open(data_path, 'r') as f:
			for line in f:
				if with_label:
					lines = line.strip().split(' +++$+++ ')
					X.append(lines[1])
					Y.append(lines[0])
				else:
					if is_test:
						X.append(line.strip()[line.find(',')+1:])
					else:
						X.append(line.strip())
		if with_label:
			self.data[name] = [X, Y]
		else:
			if is_test:
				self.data[name] = [X[1:]]
			else:
				self.data[name] = [X]

	def tokenize(self, vocab_size):
		print('create new tokenizer')
		self.tokenizer = Tokenizer(num_words=vocab_size)
		for key in self.data:
			print('tokenizing %s' % key)
			texts = self.data[key][0]
			self.tokenizer.fit_on_texts(texts)

	def save_tokenizer(self, path):
		print('save tokenizer to %s' % path)
		pk.dump(self.tokenizer, open(path, 'wb'))

	def load_tokenizer(self, path):
		print('load tokenizer from %s' % path)
		self.tokenizer = pk.load(open(path, 'rb'))

	def to_sequence(self, maxlen):
		self.maxlen = maxlen
		for key in self.data:
			print('converting %s to sequences' % key)
			tmp = self.tokenizer.texts_to_sequences(self.data[key][0])
			self.data[key][0] = np.array(pad_sequences(tmp, maxlen=maxlen))

	def to_bow(self):
		for key in self.data:
			print('converting %s to count' % key)
			self.data[key][0] = np.array(self.tokenizer.texts_to_matrix(self.data[key][0], mode='count'))

	def get_semi_data(self, name, label, threshold):
		label = np.squeeze(label)
		index = (label > 1 - threshold) + (label < threshold)
		semi_X = self.data[name][0]
		semi_Y = np.greater(label, 0.5).astype(np.int32)
		return semi_X[index,:], semi_Y[index]

	def get_data(self, name):
		return self.data[name]

	def del_data(self, name):
		return self.data.pop(name)

	def split_data(self, name, ratio):
		data = self.data[name]
		X = data[0]
		Y = data[1]
		data_size = len(X)
		val_size = int(data_size * ratio)
		return (X[val_size:], Y[val_size:]), (X[:val_size], Y[:val_size])