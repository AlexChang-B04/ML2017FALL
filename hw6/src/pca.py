import sys, os
import numpy as np
from skimage import io

imageFiles = [os.path.join(sys.argv[1], fname) for fname in os.listdir(sys.argv[1])]
images = np.array([io.imread(filename).flatten() for filename in imageFiles]).T
print(images.shape)

X_mean = images.mean(axis=1)
X = (images.T - X_mean).T

U, s, V = np.linalg.svd(X, full_matrices=False)
# np.save('U.npy', U)
# np.save('s.npy', s)
# np.save('V.npy', V)
# U, s, V = np.load('U.npy'), np.load('s.npy'), np.load('V.npy')

### Q1 ###
# io.imsave('meanface.jpg', X_mean.astype(np.uint8).reshape((600,600,3)))

### Q2 ###
# for i in range(4):
# 	M = U[:,i].copy()
# 	M -= M.min()
# 	M /= M.max()
# 	M = (M * 255).astype(np.uint8)
# 	M = M.reshape((600,600,3))
# 	io.imsave('eigenface_' + str(i) + '.jpg', M)

### Q3 ###
target = os.path.join(sys.argv[1], sys.argv[2])
M = io.imread(target).flatten() - X_mean
N = np.zeros(images.shape[0])
for j in range(4):
	N += M.dot(U[:,j]) * U[:,j]
N += X_mean
N -= N.min()
N /= N.max()
N = (N * 255).astype(np.uint8)
N = N.reshape((600,600,3))
io.imsave('reconstruction.jpg', N, quality=100)

### Q4 ###
# for i in range(4):
# 	print(s[i] / s.sum() * 100)