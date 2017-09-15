import sys
from PIL import Image

im = Image.open(sys.argv[1])
pixels = im.load()
for x in range(im.size[0]):
	for y in range(im.size[1]):
		pixels[x, y] = tuple(t//2 for t in pixels[x, y])
im.save('Q2.png')