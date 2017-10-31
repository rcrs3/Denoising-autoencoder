import autoencoder
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
data, target = mnist.train.images, mnist.train.labels

idx = np.random.rand(data.shape[0]) < 0.8

train_X, train_Y = data[idx], target[idx]

model = autoencoder.autoencoder(corruption_level = 0.2, batch = 15, epoch = 100, input_width = 480, input_height = 640)

model.fit(train_X)
