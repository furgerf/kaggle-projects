# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import sys
import numpy as np

"""
labels = list()
images = list()

with open('train.csv', 'r') as f:
  f.readline()

  for line in f:
    vals = line.split(',')
    label = int(vals[0])
    image = [int(v) for v in vals[1:]]
    labels.append(label)
    images.append(image)

"""
import pickle

# pickle.dump((labels, images), open('mnist.p', 'wb'))

labels, images = pickle.load(open('mnist.p', 'rb'))
labels = np.array(labels)
oh_labels = np.zeros((len(labels), 10))
for i,o in enumerate(oh_labels):
  o[labels[i]] = 1

images = np.array(images)

print('done loading data')

import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.reduce_sum((y_ - y) ** 2, reduction_indices=[1]))

cross_entropy_2 = tf.nn.log_poisson_loss(y, y_)

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

init = tf.global_variables_initializer()

print('setup done')

sess = tf.Session()
sess.run(init)

train_length = int(0.7 * len(labels))
print('Training set size', train_length)
for i in range(1000):
  # print('Epoch', i)
  idx = (np.random.rand(int(train_length / 2)) * train_length).astype(np.int)
  img = images[idx, :]
  lab = oh_labels[idx, :]
  sess.run(train_step, feed_dict={x: img, y_: lab})

  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: images[train_length:], y_: oh_labels[train_length:]}))

