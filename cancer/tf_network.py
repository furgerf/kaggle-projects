import tensorflow as tf

class TfNetwork(object):
  def __init__(self, dimensionality):
    self.x = tf.placeholder(tf.float32, [None, dimensionality])
    self.W = tf.Variable(tf.zeros([dimensionality, 1]), name='W')
    self.b = tf.Variable(tf.zeros([1]), name='b')
    self.y = tf.matmul(self.x, self.W) + self.b

  def load_variables(self, session):
    self.b = None
    self.W = None
    variables = tf.trainable_variables()
    for variable in variables:
      if variable.name == 'b:0':
        self.b = variable
        print('Found b', self.b, self.b.eval(session))
      if variable.name == 'W:0':
        self.W = variable
        print('Found W', self.W, self.W.eval(session))
    if self.b == None or self.W == None:
      raise ValueError('No tensors b or W found')

