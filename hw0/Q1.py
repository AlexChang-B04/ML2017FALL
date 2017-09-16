import sys

f = open(sys.argv[1], 'r')
words = f.read().split()

d = {}
l = []
for word in words:
	if word in l:
		d[word] += 1
	else:
		l.append(word)
		d[word] = 1

num = 0
for word in l:
	if num == len(l)-1:
		print(word, num, d[word], end='')
	else:
		print(word, num, d[word])
	num += 1