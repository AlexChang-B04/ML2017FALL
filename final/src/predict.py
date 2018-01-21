import sys
import numpy as np
import jieba
from gensim.models import word2vec
from scipy.spatial.distance import cosine

jieba.set_dictionary('dict.txt.big')
word_model = word2vec.Word2Vec.load(sys.argv[1])

f = open(sys.argv[2], 'r', encoding='utf-8')
f.readline()

index = []
questions = []
options = []

for line in f:
	lines = line.split(',')
	index.append(int(lines[0]))
	questions.append(lines[1].replace('A:', '').replace('B:', '').split('\t'))
	options.append(lines[2].replace('A:', '').replace('B:', '').split('\t'))

for i in range(len(questions)):
	segs = []
	for j in range(len(questions[i])):
		words = jieba.lcut(questions[i][j])
		for word in words:
			if word != ' ' and word != '':
				segs.append(word)
	questions[i] = segs

for i in range(len(options)):
	for j in range(len(options[i])):
		segs = []
		words = jieba.lcut(options[i][j])
		for word in words:
			if word != ' ' and word != '' and word != '\n':
				segs.append(word)
		options[i][j] = segs

threshold = 0.29
alpha = 1e-3
total_cnt = np.sum([word_model.wv.vocab[word].count for word in word_model.wv.vocab.keys()])
similarity = np.zeros((len(index),len(options[0])))

for i in range(len(index)):
	qLen = 0
	qCnt = 0
	qVec = np.zeros(word_model.vector_size)
	for j, qSeg in enumerate(questions[i]):
		try:
			qVec += word_model[qSeg] * (alpha / (alpha + word_model.wv.vocab[qSeg].count / total_cnt))
			qLen += len(qSeg)
			qCnt += 1
		except:
			pass
	if qCnt != 0:
		qVec /= qLen

	for j, option in enumerate(options[i]):
		oLen = 0
		oCnt = 0
		oVec = np.zeros(word_model.vector_size)
		for oSeg in option:
			try:
				oVec += word_model[oSeg] * (alpha / (alpha + word_model.wv.vocab[oSeg].count / total_cnt))
				oLen += len(oSeg)
				oCnt += 1
			except:
				pass
		if oCnt != 0:
			oVec /= oLen

		if qCnt != 0 and oCnt != 0:
			similarity[i][j] = 1 - cosine(qVec, oVec)

similarity /= similarity.max()
similarity *= 65

for i in range(len(index)):
	for j, option in enumerate(options[i]):
		sim_cnt = 0
		for k, qSeg in enumerate(questions[i]):
			for oSeg in option:
				try:
					sim = word_model.similarity(qSeg, oSeg)
					if sim > threshold:
						similarity[i][j] += sim
						sim_cnt += 1
				except:
					pass
	similarity[i][0] *= 1.1

choices = np.argmax(similarity, 1)
np.savetxt(sys.argv[3], np.c_[np.arange(len(choices))+1, choices],
		   fmt=['%d','%d'], delimiter=',', header='id,ans', comments='')
