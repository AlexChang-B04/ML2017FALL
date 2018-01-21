import os
import sys
import jieba

jieba.set_dictionary('dict.txt.big')

trainFiles = os.listdir(sys.argv[1])
sentences = []
for filename in trainFiles:
	if filename[0] == '.':
		continue
	
	tmp = []
	f = open(os.path.join(sys.argv[1], filename), 'r')
	for line in f:
		if line[0] == '"' and line[-2] == '"':
			continue
		if len(tmp) == 0 or (len(tmp) > 0 and line[:-1] != tmp[-1]):
			tmp.append(line[:-1])

	for j, t in enumerate(tmp):
		tmp[j] = jieba.lcut(t)

	for j in range(len(tmp)-2):
		sentences.append(tmp[j] + tmp[j+1] + tmp[j+2])

f = open('trainData_preprocessed.txt', 'w')
for sentence in sentences:
	f.write(' '.join(sentence) + '\n')
f.close()
