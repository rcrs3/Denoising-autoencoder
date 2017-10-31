import tensorflow as tf
import numpy as np
import utils

class autoencoder(object, n_components):
	
	def __init__(self):
		self.W_ = None
		self.bh_ = None
		self.bv_ = None
		
		self.encode = None
		self.decode = None
		
		self.n_components = n_components
		
	
	def set_variables(self, n_features):
		W_ = tf.Variable(utils.xavier_init(n_features, self.n_components))
		bh_ = tf.Variable(tf.zeros([self.n_components]))
		bv_ = tf.Variable(tf.zeros([n_features]))
		
		return W_, bh_, bv_
		
		
	def _create_encode(self):
		self.encode = tf.nn.sigmoid(tf.matmul(self.data_corr, self.W_) + self.bh_)
	
	def _create_decode(self):
		self.decode = tf.nn.sigmoid(tf.matmul(self.encode, tf.transpose(self.W_)) + self.bv_)
