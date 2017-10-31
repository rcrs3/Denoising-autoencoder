from scipy import misc
import tensorflow as tf
import numpy as np


def xavier_init(n_features, n_components, const = 1):
	
	low = -const * np.sqrt(6.0 / (n_features + n_components))
	high = const * np.sqrt(6.0 / (n_features + n_components))
	
	return tf.random_uniform((n_features, n_components), minval = low, maxval = high)
	
def get_batch(X, X_, size):
    a = np.random.choice(len(X), size, replace=False)
    return X[a], X_[a]

