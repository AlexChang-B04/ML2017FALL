import sys, argparse, os
import pickle as pk
import numpy as np

from keras.models import load_model

from util import DataManager

parser = argparse.ArgumentParser(description='Sentiment classification: Ensemble')

parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--max_length', default=40, type=int)

parser.add_argument('--test_data', default='data/testing_data.txt')

parser.add_argument('--model_dir', default='model/')
parser.add_argument('--model_list', default='model/modellist.txt')
parser.add_argument('--output_file', default='ensemble.csv')
args = parser.parse_args()

def main():
	modellist = [line.strip() for line in open(args.model_list, 'r')]
	pred = np.zeros(200000)

	for modelname in modellist:
		load_path = os.path.join(args.model_dir, modelname)

		dm = DataManager()
		dm.add_data('test_data', args.test_data, False, True)
		dm.load_tokenizer(os.path.join(load_path, 'token.pk'))
		dm.to_sequence(args.max_length)

		print('loading model...')
		model = load_model(os.path.join(load_path, 'model_full.h5'))
		#model.summary()

		[test_X] = dm.get_data('test_data')
		pred = pred + np.squeeze(model.predict(test_X, batch_size=args.batch_size, verbose=1))
		
	pred = pred / len(modellist)
	np.savetxt(args.output_file, np.c_[np.arange(len(pred)), np.round(pred)],
		   	   fmt='%d', delimiter=',', header='id,label', comments='')

if __name__ == '__main__':
	main()