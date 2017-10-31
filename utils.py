from scipy import misc
import tensorflow as tf
import numpy as np


def xavier_init(n_features, n_components, const = 1):
	
	low = -const * np.sqrt(6.0 / (n_features + n_components))
	high = const * np.sqrt(6.0 / (n_features + n_components))
	
	return tf.random_uniform((n_features, n_components), minval = low, maxval = high)
