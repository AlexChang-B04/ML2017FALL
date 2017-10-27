import os
import pandas as pd
import numpy as np
import argparse

def load_data(train_data_path, train_label_path, test_data_path):
	col_select = [i for i in range(106)]
	X_all = np.array(pd.read_csv(train_data_path, sep=',').values)[:,col_select]
	Y_all = np.array(pd.read_csv(train_label_path, sep=',').values)
	X_test = np.array(pd.read_csv(test_data_path, sep=',').values)[:,col_select]
	"""
	col_append = [(i, p/2.0) for i in [0,3,4,5] for p in range(3,19)]
	X_all = np.hstack((X_all, np.array([X_all[:,i] ** p for (i, p) in col_append]).T))
	X_test = np.hstack((X_test, np.array([X_test[:,i] ** p for (i, p) in col_append]).T))

	for col in [0,3,4,5]:
		X_all = np.hstack((X_all, (np.log(np.clip(X_all[:,col], 1, None))).reshape((32561, 1))))
		X_test = np.hstack((X_test, (np.log(np.clip(X_test[:,col], 1, None))).reshape((16281, 1))))
	"""
	return (X_all, Y_all, X_test)

def normalize(X_all, X_test):
	X = np.concatenate((X_all, X_test))
	X_mean = np.tile(np.mean(X, 0), (X.shape[0], 1))
	X_std = np.tile(np.std(X, 0), (X.shape[0], 1))
	X_normed = (X - X_mean) / X_std

	X_all_normed = X_normed[:X_all.shape[0]]
	X_test_normed = X_normed[X_all.shape[0]:]

	return (X_all_normed, X_test_normed)

def shuffle_data(X, Y):
	random_order = np.arange(len(X))
	np.random.shuffle(random_order)
	return (X[random_order], Y[random_order])

def split_valid_set(X_all, Y_all, percentage):
	split_pos = int(len(X_all) * percentage)
	X_all, Y_all = shuffle_data(X_all, Y_all)

	return (X_all[split_pos:], Y_all[split_pos:], X_all[0:split_pos], Y_all[0:split_pos])

def sigmoid(z):
	z_clip = np.clip(z, -1e2, 1e2)
	return np.clip((1 / (1.0 + np.exp(-z_clip))), 1e-8, 1-(1e-8))

def acc_generative(X, Y, C0, C1, mu0, mu1, sigma):
	sigma_inv = np.linalg.inv(sigma)
	w = np.dot((mu0 - mu1), sigma_inv)
	b = -0.5 * np.dot(np.dot(mu0, sigma_inv), mu0) + 0.5 * np.dot(np.dot(mu1, sigma_inv), mu1) + np.log(C0 / C1)
	z = np.dot(w, X.T) + b
	p = np.around(sigmoid(z))
	result = (np.squeeze(Y) == p)
	return (np.sum(result) / result.shape[0])

def train_generative(X_all, Y_all, validation, save_path):
	if validation:
		X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, 0.1)
	else:
		X_train, Y_train = (X_all, Y_all)

	C0 = 0
	C1 = 0
	mu0 = np.zeros((X_train.shape[1],))
	mu1 = np.zeros((X_train.shape[1],))
	sigma0 = np.zeros((X_train.shape[1],X_train.shape[1]))
	sigma1 = np.zeros((X_train.shape[1],X_train.shape[1]))

	for i in range(X_train.shape[0]):
		if Y_train[i] == 1:
			C0 = C0 + 1
			mu0 = mu0 + X_train[i]
		else:
			C1 = C1 + 1
			mu1 = mu1 + X_train[i]
	mu0 = mu0 / C0
	mu1 = mu1 / C1

	for i in range(X_train.shape[0]):
		if Y_train[i] == 1:
			sigma0 = sigma0 + np.dot(np.transpose([X_train[i] - mu0]), [X_train[i] - mu0])
		else:
			sigma1 = sigma1 + np.dot(np.transpose([X_train[i] - mu1]), [X_train[i] - mu1])
	sigma0 = sigma0 / C0
	sigma1 = sigma1 / C1
	sigma = (float(C0) / (C0 + C1)) * sigma0 + (float(C1) / (C0 + C1)) * sigma1

	train_acc = acc_generative(X_train, Y_train, C0, C1, mu0, mu1, sigma)
	if validation:
		valid_acc = acc_generative(X_valid, Y_valid, C0, C1, mu0, mu1, sigma)
	else:
		valid_acc = 0.0
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		np.savetxt(os.path.join(save_path, 'C0'), [C0,])
		np.savetxt(os.path.join(save_path, 'C1'), [C1,])
		np.savetxt(os.path.join(save_path, 'mu0'), mu0)
		np.savetxt(os.path.join(save_path, 'mu1'), mu1)
		np.savetxt(os.path.join(save_path, 'sigma'), sigma)

	return (train_acc, valid_acc)

def infer_generative(X_test, save_path, output_path):
	C0 = np.loadtxt(os.path.join(save_path, 'C0'))
	C1 = np.loadtxt(os.path.join(save_path, 'C1'))
	mu0 = np.loadtxt(os.path.join(save_path, 'mu0'))
	mu1 = np.loadtxt(os.path.join(save_path, 'mu1'))
	sigma = np.loadtxt(os.path.join(save_path, 'sigma'))

	sigma_inv = np.linalg.inv(sigma)
	w = np.dot((mu0 - mu1), sigma_inv)
	b = -0.5 * np.dot(np.dot(mu0, sigma_inv), mu0) + 0.5 * np.dot(np.dot(mu1, sigma_inv), mu1) + np.log(C0 / C1)
	z = np.dot(w, X_test.T) + b
	result = np.around(sigmoid(z))

	if not os.path.exists(os.path.dirname(output_path)):
		os.makedirs(os.path.dirname(output_path))
	with open(output_path, 'w') as f:
		f.write('id,label\n')
		for i in range(len(result)):
			f.write('%d,%d\n' % (i+1, result[i]))

def acc_logistic(X, Y, w, b):
	z = np.dot(w, X.T) + b
	p = np.around(sigmoid(z))
	result = (np.squeeze(Y) == p)
	return (np.sum(result) / result.shape[0])

def train_logistic(X_all, Y_all, X_test, batch_size, epoch, validation, save_path):
	if validation:
		X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, 0.1)
	else:
		X_train, Y_train = (X_all, Y_all)

	w = np.zeros((X_train.shape[1],)) + 1e-3
	b = np.zeros((1,))
	eta = 0.001
	# lamda = 0.01
	m = 0
	v = 0
	mb = 0
	vb = 0
	bm = 0.9
	bv = 0.999
	bm_p = 1
	bv_p = 1
	epsilon = 1e-8
	# acc_max_params = {'max': 0, 'pos': 0, 'w': w, 'b': b}
	# early_stopping_rounds = 300

	if batch_size == 0:
		batch_size = X_train.shape[0]
	step_num = int(X_train.shape[0] / batch_size)
	
	for e in range(1, epoch + 1):
		X_train, Y_train = shuffle_data(X_train, Y_train)
		total_loss = 0.0

		for step in range(step_num):
			X = X_train[step*batch_size:(step+1)*batch_size]
			Y = Y_train[step*batch_size:(step+1)*batch_size]

			z = np.dot(w, X.T) + b
			f = sigmoid(z)

			total_loss += -(np.dot(np.squeeze(Y), np.log(f)) + np.dot(1 - np.squeeze(Y), np.log(1 - f)))

			w_grad = -1.0 * np.dot(np.squeeze(Y) - f, X)
			m = bm * m + (1 - bm) * w_grad
			v = bv * v + (1 - bv) * (w_grad ** 2)
			bm_p = bm_p * bm
			bv_p = bv_p * bv
			m_ = m / (1 - bm_p)
			v_ = v / (1 - bv_p)
			w = w - eta * m_ / (v_ ** 0.5 + epsilon)

			b_grad = -1.0 * np.sum(np.squeeze(Y) - f)
			mb = bm * mb + (1 - bm) * b_grad
			vb = bv * vb + (1 - bv) * (b_grad ** 2)
			mb_ = mb / (1 - bm_p)
			vb_ = vb / (1 - bv_p)
			b = b - eta * mb_ / (vb_ ** 0.5 + epsilon)

		if e % 100 == 0:
			print('epoch %04d: train acc = %.10f loss = %.10f' % (e, acc_logistic(X_train, Y_train, w, b), total_loss))
		"""
		if validation:
			valid_acc = acc_logistic(X_valid, Y_valid, w, b)
			if valid_acc > acc_max_params['max']:
				acc_max_params['max'] = valid_acc
				acc_max_params['pos'] = e
				acc_max_params['w'] = w
				acc_max_params['b'] = b
			elif e - acc_max_params['pos'] > early_stopping_rounds:
				print('Early Stopping at epoch %d' % (e))
				print('Acc_max = %.10f at epoch %d' % (acc_max_params['max'], acc_max_params['pos']))
				w = acc_max_params['w']
				b = acc_max_params['b']
				break
			else:
				pass
		"""
	train_acc = acc_logistic(X_train, Y_train, w, b)
	if validation:
		valid_acc = acc_logistic(X_valid, Y_valid, w, b)
	else:
		valid_acc = 0.0
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		np.savetxt(os.path.join(save_path, 'w'), w)
		np.savetxt(os.path.join(save_path, 'b'), [b,])
	
	return (train_acc, valid_acc)

def infer_logistic(X_test, save_path, output_path):
	w = np.loadtxt(os.path.join(save_path, 'w'))
	b = np.loadtxt(os.path.join(save_path, 'b'))

	z = np.dot(w, X_test.T) + b
	result = np.around(sigmoid(z))

	if not os.path.exists(os.path.dirname(output_path)):
		os.makedirs(os.path.dirname(output_path))
	with open(output_path, 'w') as f:
		f.write('id,label\n')
		for i in range(len(result)):
			f.write('%d,%d\n' % (i+1, result[i]))

def main(args):
	# np.random.seed(1126)

	X_all, Y_all, X_test = load_data(args.train_data_path, args.train_label_path, args.test_data_path)
	X_all, X_test = normalize(X_all, X_test)

	if args.train or args.valid:
		if args.generative:
			train_acc, valid_acc = train_generative(X_all, Y_all, args.valid, args.save_path)
		elif args.logistic:
			train_acc, valid_acc = train_logistic(X_all, Y_all, X_test, args.batch_size, args.epoch, args.valid, args.save_path)
		print('train acc = %.10f valid_acc = %.10f' % (train_acc, valid_acc))
	elif args.infer:
		if args.generative:
			infer_generative(X_test, args.save_path, args.output_path)
		elif args.logistic:
			infer_logistic(X_test, args.save_path, args.output_path)
	else:
		pass
    
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Probabilistic Generative Model or Logistic Regression Model for Binary Classification')
	group1 = parser.add_mutually_exclusive_group()
	group2 = parser.add_mutually_exclusive_group()
	group1.add_argument('--train', action='store_true', default=False, dest='train', help='Input --train to Train')
	group1.add_argument('--valid', action='store_true', default=False, dest='valid', help='Input --valid to Train & Valid')
	group1.add_argument('--infer', action='store_true', default=False, dest='infer', help='Input --infer to Infer')
	group2.add_argument('--generative', action='store_true', default=False, dest='generative', help='Choose probabilistic generative model')
	group2.add_argument('--logistic', action='store_true', default=True, dest='logistic', help='Choose logistic regression model')
	parser.add_argument('--train-data-path', type=str, default='./data/X_train.dms', dest='train_data_path', help='Path to training data')
	parser.add_argument('--train-label-path', type=str, default='./data/Y_train.dms', dest='train_label_path', help='Path to training data\'s label')
	parser.add_argument('--test-data-path', type=str, default='./data/X_test.dms', dest='test_data_path', help='Path to testing data')
	parser.add_argument('--save-path', type=str, default='./params', dest='save_path', help='Path to saving parameters')
	parser.add_argument('--output-path', type=str, default='./out/prediction.csv', dest='output_path', help='Path to outputing prediction')
	parser.add_argument('--batch-size', type=int, default=32, dest='batch_size', help='Specify the batch size')
	parser.add_argument('--epoch', type=int, default=1000, dest='epoch', help='Specify the number of epoch')
	args = parser.parse_args()
	main(args)
