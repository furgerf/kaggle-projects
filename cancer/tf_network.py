import tensorflow as tf

class TfNetwork(object):

  def __init__(self, dimensionality):
    """
    Creates a new TfNetwork. Note that it is apparently required to define
    variables W and b even if they'll be later restored from a previous session.

    dimensionality: (int) Number of dimensions (features) for the network
    """
    self.x = tf.placeholder(tf.float32, [None, dimensionality])
    self.W = tf.Variable(tf.random_normal([dimensionality, 1]), name='W')
    self.b = tf.Variable(tf.random_normal([1]), name='b')
    self.y = tf.matmul(self.x, self.W) + self.b

  def load_variables(self, session):
    """
    Loads variables W and b from the provided session.
    """
    self.b = None
    self.W = None
    variables = tf.trainable_variables()
    for variable in variables:
      if variable.name == 'b:0':
        self.b = variable
        print('Found b', self.b, self.b.eval(session))
      if variable.name == 'W:0':
        self.W = variable
        w = self.W.eval(session)
        print('Found W', self.W, w.min(), '-', w.max(), '~', w.mean())
    if self.b == None or self.W == None:
      raise ValueError('Variable b or W was not found')
    self.y = tf.matmul(self.x, self.W) + self.b

