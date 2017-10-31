import tensorflow as tf
import numpy as np
import utils

#input_width = 480
#input_height = 640

#n_hidden_layers = 30

#number of nodes in the input and hidden layer
#n_visible = input_width * input_height * 3
#n_hidden = n_visible

#corruption_level = 0.2

#Number of training examples
#batch = 15

#Number of foward and back pass of the training examples
#epoch = 100

class autoencoder:
	
	def __init__(self, corruption_level, batch, epoch, input_width, input_height, colors=1):
		self.corruption_level = corruption_level
		self.batch = batch
		self.epoch = epoch
		self.Weights = []
		self.Biases = []
		self.n_visible = input_width * input_height * colors
		self.n_hidden = self.n_visible
		self.n_hidden_layers = 30		
			
	def add_noise(self, X):
		X_noise = X.copy()
	
		n_samples = X.shape[0]
		n_features = X.shape[1]
		
		corr_ratio = np.round(self.corruption_level * n_features).astype(np.int)
		
		for i in range(n_samples):
			mask = np.random.randint(0, n_features, corr_ratio)
		
			for m in mask:
				X_noise[i][m] = 0
	
		return X_noise
		
	def fit(self, X):
		for i in range(self.n_hidden_layers):
			print(i)
			tmp = X.copy()
			X = self.run(self.n_visible, self.n_hidden, data_x = self.add_noise(tmp), data_x_ = X) 

	def create_encoder(self, X, W, bh):
		return tf.nn.sigmoid(tf.matmul(X, W) + bh)
		
	def create_decoder(self, Y, W_t, bv):
		return tf.nn.sigmoid(tf.matmul(Y, W_t) + bv)

	def transform(self, data):
		tf.reset_default_graph()
		sess = tf.Session()
	
		x = tf.constant(data, "float")
	
		for w, b in zip(self.Weights, self.Biases):
			weight = tf.constant(w, "float")
			bias = tf.constant(b, "float")
			x = self.create_encoder(x, weight, bias)
	
		return x.eval(session = sess)
	
	def run(self, n_visible, n_hidden, data_x, data_x_):
		tf.reset_default_graph()
		sess = tf.Session()
	
		#Node for input
		X = tf.placeholder("float", [None, n_visible], name = "X")

		#Node for corruption mask
		X_ = tf.placeholder("float", [None, n_visible], name = "X_")
	
		#Weights
		W = tf.Variable(utils.xavier_init(n_visible, n_hidden), name="W")
		W_t = tf.transpose(W)

		#Bias
		bh = tf.Variable(tf.zeros([n_hidden]), name="bh")
		bv = tf.Variable(tf.zeros([n_visible]), name="bv")
	
		#encoder
		Y = self.create_encoder(X,W,bh)
		#decoder
		Z = self.create_decoder(Y,W_t,bv)		

	
	
		loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(X_, Z))))
	
		#optimizer
		train_op = tf.train.GradientDescentOptimizer(0.002).minimize(loss)
	
		sess.run(tf.global_variables_initializer())
	
		for i in range(self.epoch):
			b_x, b_x_ = utils.get_batch(data_x, data_x_, self.batch)
			sess.run(train_op, feed_dict = {X: b_x, X_: b_x_})
		self.Weights.append(sess.run(W))
		self.Biases.append(sess.run(bh))
	
		return sess.run(Y, feed_dict= {X: data_x_})
		

