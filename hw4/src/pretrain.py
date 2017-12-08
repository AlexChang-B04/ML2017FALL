import sys

from gensim.utils import tokenize
from gensim.models import Word2Vec

sentences = []
for line in open(sys.argv[3], 'r'):
	sentences.append(list(tokenize(line[10:-1], deacc=True)))
	#sentences.append(line[10:-1].split(' '))
for line in open(sys.argv[4], 'r'):
	sentences.append(list(tokenize(line[:-1], deacc=True)))
	#sentences.append(line[:-1].split(' '))
for line in open(sys.argv[5], 'r'):
	sentences.append(list(tokenize(line[line.find(',')+1:-1], deacc=True)))
	#sentences.append(line[line.find(',')+1:-1].split(' '))
sentences.pop(-200001)
print('%d sentences.' % len(sentences))

word_model = Word2Vec(sentences, min_count=int(sys.argv[1]))
word_model.save(sys.argv[2])