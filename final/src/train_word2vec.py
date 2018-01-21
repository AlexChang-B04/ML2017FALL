from gensim.models import word2vec

size = 32
window = 3
min_count = 1
sg = 1
negative = 3
iteration = 35

sentences = word2vec.LineSentence('trainData_preprocessed.txt')
model = word2vec.Word2Vec(sentences,
						  size=size, window=window ,min_count=min_count, sg=sg, negative=negative, iter=iteration)
model.save('word_model.w2v')
