# -*- coding: utf-8 -*-

import csv
import numpy as np
import tensorflow as tf

# Reading data
reader = csv.reader(open("normalized_palio_features.csv", "rt"), delimiter = ",")
x = list(reader)
features = np.array(x).astype("float")
np.random.shuffle(features)

# Separating features (x) from prediction labels (y)
data_x = features[:,1:]
data_y = features[:,:1]

# Dividing dataet into train set and test set
l = float(features.shape[0])
train_set_size = int(l * 0.8)

x_data , x_test = data_x[:train_set_size,:], data_x[train_set_size:,:]
y_data , y_test = data_y[:train_set_size,:], data_y[train_set_size:,:]

######################################################################################
# Here the learning process effectively begins
# Using a 3-layered Neural Network, each with 3 neurons initialized with a random 
# Gaussian distribution
######################################################################################

print("Starting learning process")

# Regularization strength
Lambda = 0.01
learning_rate = 0.01

# Defining Input, Weights and Biases
with tf.name_scope('input'):
	x = tf.placeholder("float", name = "palio")
	y = tf.placeholder("float", name = "prices")

with tf.name_scope('weights'):
	w1 = tf.Variable(tf.random_normal([3,3]), name = 'w1')
	w2 = tf.Variable(tf.random_normal([3,2]), name = 'w2')
	w3 = tf.Variable(tf.random_normal([2,1]), name = 'w3')

with tf.name_scope('biases'):
	b1 = tf.Variable(tf.random_normal([1,3]), name = 'b1')
	b2 = tf.Variable(tf.random_normal([1,2]), name = 'b2')
	b3 = tf.Variable(tf.random_normal([1,1]), name = 'b3')

# Multilayered Neural Network
with tf.name_scope('layer_1'):
	layer_1 = tf.nn.tanh(tf.add(tf.matmul(x,w1), b1))
with tf.name_scope('layer_2'):
	layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1,w2), b2))
with tf.name_scope('layer_3'):
	layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_2,w3), b3))

# Regularization
with tf.name_scope('regularization'):
	regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3)

# Loss function
with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.square(layer_3 - y)) + Lambda*regularization
	loss = tf.Print(loss, [loss], "loss")

# Optimization
with tf.name_scope('train'):
	train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Lauching Model
init = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(init)

	for i in range(5000):
		accuracy = session.run(train_op, feed_dict = {x: x_data, y: y_data})

	# Testing the network
	print("Testing Data")
	print("Loss: " + str(session.run([layer_3, loss], feed_dict = {x: x_test, y: y_test})[1]))

writer = tf.summary.FileWriter('tensorboard', graph = tf.get_default_graph())
