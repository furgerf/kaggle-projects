import tensorflow as tf

class TfNetwork(object):
  def __init__(self, dimensionality):

    self.x = tf.placeholder(tf.float32, [None, dimensionality])
    self.W = tf.Variable(tf.zeros([dimensionality, 1]))
    self.b = tf.Variable(tf.zeros([1]))
    self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)

