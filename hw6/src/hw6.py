import sys
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.cluster import KMeans

image = np.load(sys.argv[1]).astype('float32') / 255. - 0.5

encoder_i = Input( shape=(image.shape[1],) )
encoder_o = Dense( units=512, activation='relu' )(encoder_i)
encoder_o = Dense( units=256, activation='relu' )(encoder_o)
encoder_o = Dense( units=128, activation='relu' )(encoder_o)
encoder_o = Dense( units=64,  activation='relu' )(encoder_o)

decoder_o = Dense( units=128,  activation='relu' )(encoder_o)
decoder_o = Dense( units=256, activation='relu' )(decoder_o)
decoder_o = Dense( units=512, activation='relu' )(decoder_o)
decoder_o = Dense( units=image.shape[1], activation='tanh' )(decoder_o)

autoencoder = Model(encoder_i, decoder_o)
encoder = Model(encoder_i, encoder_o)

autoencoder.compile(optimizer=Adam(lr=1e-4), loss='mse')
autoencoder.summary()

# modelcheck = ModelCheckpoint('model/model_AE_4_64_256_128_lr.h5', monitor='loss', save_best_only=True, mode='min')
# autoencoder.fit(image, image,
# 				batch_size=256,
# 				epochs=128,
# 				callbacks=[modelcheck])

autoencoder.load_weights('model/model_AE_4_64_256_128_lr.h5')
encoder.set_weights(autoencoder.get_weights()[:len(encoder.get_weights())])
image_reduced = encoder.predict(image, verbose=1)

kmeans = KMeans(n_clusters=2, max_iter=10000).fit(image_reduced)
# np.save('model/image_label.npy', kmeans.labels_)

testCase = np.loadtxt(sys.argv[2], dtype=int, delimiter=',', skiprows=1)
answer = np.array([kmeans.labels_[a] == kmeans.labels_[b] for i, a, b in testCase])
np.savetxt(sys.argv[3], np.c_[np.arange(len(answer)), answer],
		   fmt='%d', delimiter=',', header='ID,Ans', comments='')

