import sys

f = open(sys.argv[1], 'r')
words = f.read().split()

d = {}
for word in words:
	if word in d:
		d[word] += 1
	else:
		d[word] = 1

cnt = 0
for word, count in d.items():
	if cnt == len(d)-1:
		print(word, cnt, count, end='')
	else:
		print(word, cnt, count)
	cnt += 1