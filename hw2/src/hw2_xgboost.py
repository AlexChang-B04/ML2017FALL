import os
import pandas as pd
import numpy as np
import argparse
import xgboost as xgb

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
		X_all = np.hstack((X_all, np.log(np.clip(X_all[:,col], 1, None)).reshape((32561, 1))))
		X_test = np.hstack((X_test, np.log(np.clip(X_test[:,col], 1, None)).reshape((16281, 1))))
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

def train(X_all, Y_all, X_test, validation, save_path):
	if validation:
		X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, 0.1)
		xgb_valid = xgb.DMatrix(X_valid, label=Y_valid)
	else:
		X_train, Y_train = (X_all, Y_all)

	xgb_train = xgb.DMatrix(X_train, label=Y_train)
	watchlist = [(xgb_train, 'train')]
	if validation:
		xgb_valid = xgb.DMatrix(X_valid, label=Y_valid)
		watchlist.append((xgb_valid, 'valid'))
	
	params = {'objective': 'binary:logistic', 'silent': 1, 'eta': 0.03, 'lambda': 3, 'seed': 1126}
	model = xgb.train(params, xgb_train, 1000, watchlist)
	if not validation:
		if not os.path.exists(os.path.dirname(save_path)):
			os.makedirs(os.path.dirname(save_path))
		model.save_model(save_path)
	"""
	model = xgb.XGBClassifier(learning_rate=0.07, n_estimators=1300, silent=True, nthread=2)
	model.fit(X_all, np.squeeze(Y_all), eval_metric='error')
	preds = model.predict(X_test)
	"""

def infer(X_test, save_path, output_path):
	xgb_test = xgb.DMatrix(X_test)

	params = {'objective': 'binary:logistic', 'silent': 1, 'eta': 0.03, 'lambda': 3, 'seed': 1126}
	model = xgb.Booster(params)
	model.load_model(save_path)
	preds = np.round(model.predict(xgb_test))

	if not os.path.exists(os.path.dirname(output_path)):
		os.makedirs(os.path.dirname(output_path))
	np.savetxt(output_path, np.c_[range(1,len(preds)+1),preds], delimiter=',', header='id,label', comments='', fmt='%d')

def main(args):
	#np.random.seed(1126)

	X_all, Y_all, X_test = load_data(args.train_data_path, args.train_label_path, args.test_data_path)
	X_all, X_test = normalize(X_all, X_test)
	
	if args.train:
		train(X_all, Y_all, X_test, args.valid, args.save_path)
	elif args.valid:
		train(X_all, Y_all, X_test, args.valid, args.save_path)
	elif args.infer:
		infer(X_test, args.save_path, args.output_path)
	else:
		pass

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Apply XGBoost for Binary Classification')
	group1 = parser.add_mutually_exclusive_group()
	group1.add_argument('--train', action='store_true', default=False, dest='train', help='Input --train to Train')
	group1.add_argument('--valid', action='store_true', default=False, dest='valid', help='Input --valid to Train & Valid')
	group1.add_argument('--infer', action='store_true', default=False, dest='infer', help='Input --infer to Infer')
	parser.add_argument('--train-data-path', type=str, default='./data/X_train.dms', dest='train_data_path', help='Path to training data')
	parser.add_argument('--train-label-path', type=str, default='./data/Y_train.dms', dest='train_label_path', help='Path to training data\'s label')
	parser.add_argument('--test-data-path', type=str, default='./data/X_test.dms', dest='test_data_path', help='Path to testing data')
	parser.add_argument('--save-path', type=str, default='./model/xgb.model', dest='save_path', help='Path to saving model')
	parser.add_argument('--output-path', type=str, default='./out/prediction_xgboost.csv', dest='output_path', help='Path to outputing prediction')
	args = parser.parse_args()
	main(args)
