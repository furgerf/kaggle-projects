import os
import sys
import numpy as np
import tensorflow as tf
import pickle
from scan import Scan
from data_manager import DataManager
from tf_network import TfNetwork
import csv
from datetime import datetime

def load_training_set(file_name):
  with open(file_name) as csv_file:
    reader = csv.reader(csv_file)

    # skip header
    next(reader, None)
    return np.array([Scan(scan_id, label) for scan_id, label in reader])

"""
def load_test_set():
  with open('stage1_labels.csv') as csv_file:
    reader = csv.reader(csv_file)

    # skip header
    next(reader, None)

    training_scans = []
    return np.array([Scan(scan_id, label) for scan_id, label in reader])
"""

def train(sess, network, train_step, y_, scans, dimensionality):
  training_set_size = len(scans)
  training_batch_size = 15
  epochs = 100

  for i in range(epochs):
    print(datetime.utcnow(), 'Epoch {}/{}'.format(i+1, epochs))

    print(datetime.utcnow(), 'Selecting examples... ', end='')
    sys.stdout.flush()
    idx = (np.random.rand(training_batch_size) * training_set_size).astype(np.int)
    current_training_scans = scans[idx]

    print('loading data... ', end='')
    sys.stdout.flush()
    current_training_images = [scan.data.reshape(-1)[:dimensionality] for scan in current_training_scans]
    current_training_labels  = [scan.label for scan in current_training_scans]

    print('training... ', end='')
    sys.stdout.flush()
    sess.run(train_step, feed_dict={network.x: current_training_images, y_: np.transpose([current_training_labels])})

    print('freeing memory...', end='')
    sys.stdout.flush()

    for scan in current_training_scans:
      scan._data = None

    print('done!')

    # correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print(sess.run(accuracy, feed_dict={x: images[train_length:], y_: oh_labels[train_length:]}))

def store_session(session):
  saver = tf.train.Saver()
  saver.save(session, 'hello-world-high-dimensional-session.ckpt')

def load_session(dimensionality):
  network = TfNetwork(dimensionality)
  # saver = tf.train.Saver()
  session = tf.Session()
  # saver.restore(session, 'hello-world-high-dimensional-session.ckpt')

  init = tf.global_variables_initializer()
  session.run(init)

  new_saver = tf.train.import_meta_graph('hello-world-high-dimensional-session.ckpt.meta')
  new_saver.restore(session, tf.train.latest_checkpoint('./'))

  return session, network

def predict(sess, network, scans, dimensionality):
  for scan in scans:
    scan_image = scan.data.reshape(-1)[:dimensionality]
    scan.predicted_label = sess.run(network.y, feed_dict ={network.x: [scan_image]})
    print("Prediction for {}: {} -> {}".format(scan.scan_id, scan.predicted_label, scan.label))
    scan._data = None

start_time = datetime.utcnow()
dimensionality = 15886000
"""
print('Setting up network')
# dimensionality = 10000000
network = TfNetwork(dimensionality)
y_ = tf.placeholder(tf.float32, [None, 1])
l2_error = tf.reduce_mean(tf.reduce_sum((y_ - network.y) ** 2, reduction_indices=[1]))
# cross_entropy = tf.nn.log_poisson_loss(y, y_)
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(l2_error)

print('Setting up session')
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print('Loading training set metadata')
training_set = load_training_set()
print('Starting training')
train(sess, network, train_step, y_, training_set, dimensionality)

print('Storing session')
store_session(sess)
"""

print('Loading session')
sess, network = load_session(dimensionality)

print('Loading training set metadata')
test_set = load_training_set('last-20-percent-training-set.csv')
predict(sess, network, test_set, dimensionality)

print('DONE! Started {}, ended {}, took {}'.format(start_time, datetime.utcnow(), datetime.utcnow() - start_time))

