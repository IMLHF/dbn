import tensorflow as tf
import numpy as np
from print_ import print_
from PIL import Image


class NN(object):
  def __init__(
          self, sizes=None, learning_rate=None, batch_size=None, n_epoches=None):
    self._sizes = sizes
    self._learning_rate = learning_rate
    self._batch_size = batch_size
    self._n_epoches = n_epoches
    self.w_list = []
    self.b_list = []
    if self._sizes is None:
      return
    with tf.Session() as session:
      for i, size in enumerate(self._sizes[1:]):
        in_size = self._sizes[i]
        out_size = size
        self.w_list.append(session.run(tf.random_normal([in_size, out_size])))
        self.b_list.append(session.run(tf.random_normal([out_size])))

  def load_from_dbn_to_reconstructNN(self, dbn):
    assert len(self._sizes) == len(dbn._sizes)*2-1
    for i in range(len(dbn._sizes)):
      assert dbn._sizes[i] == self._sizes[i]
      assert self._sizes[-i-1] == dbn._sizes[i]
    for i in range(len(dbn._sizes)-1):
      self.w_list[i], _, self.b_list[i] = dbn._rbm_list[i].get_param()
    n_dbn_layers = len(dbn._sizes)
    for i in range(n_dbn_layers-1):
      w, vb, _ = dbn._rbm_list[i].get_param()
      self.w_list[-i-1] = np.transpose(w)
      self.b_list[-i-1] = vb

  # start_layer，end_layer : 网络层编号，从0开始编号。
  # 从nnet中抽取start_layer到end_layer层作为新的网络
  def load_layers_from_NN(self, nnet, start_layer, end_layer):
    assert start_layer < end_layer
    assert end_layer-start_layer+1 < len(nnet._sizes)
    self._sizes = nnet._sizes[start_layer:end_layer+1]
    self._learning_rate = nnet._learning_rate
    self._batch_size = nnet._batch_size
    self._n_epoches = nnet._n_epoches
    self.w_list = nnet.w_list[start_layer:end_layer]
    self.b_list = nnet.b_list[start_layer:end_layer]

  def _load_w_b_from_self(self):
    w_list = []
    b_list = []
    for i, size in enumerate(self._sizes[1:]):
      in_size = self._sizes[i]
      out_size = size
      w_list.append(tf.get_variable("nn_weight_"+str(i), [in_size, out_size],
                                    # initializer=tf.random_normal_initializer()))
                                    initializer=tf.constant_initializer(self.w_list[i])))
      b_list.append(tf.get_variable("nn_bias_"+str(i), [out_size],
                                    # initializer=tf.random_normal_initializer()))
                                    initializer=tf.constant_initializer(self.b_list[i])))
    return w_list, b_list

  def _save_w_b_to_self(self, session, w_list, b_list):
    self.w_list = session.run(w_list)
    self.b_list = session.run(b_list)

  def _MLP(self, x_in, w_list, b_list):
    y_out = x_in
    for i in range(len(self._sizes)-2):
      y_out = tf.nn.sigmoid(
          tf.add(tf.matmul(y_out, w_list[i]), b_list[i]))
    y_out = tf.add(
        tf.matmul(y_out, w_list[len(self._sizes)-2]), b_list[len(self._sizes)-2])# TODO fix loss fun
    return y_out

  def train(self, X, Y):
    batch_size = self._batch_size
    n_epoches = self._n_epoches
    display_epoches = 1
    x_input = tf.placeholder("float", [None, self._sizes[0]])
    y_target = tf.placeholder("float", [None, self._sizes[-1]])
    w_list, b_list = self._load_w_b_from_self()
    y_output = self._MLP(x_input, w_list, b_list)
    # TODO fix loss fun
    loss_cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=y_output, labels=y_target))
    # loss_cross_entropy = -tf.reduce_mean(y_target*y_output)
    # loss_cross_entropy = -tf.reduce_mean(y_target*tf.log(y_output))
    train_op = tf.train.GradientDescentOptimizer(
        learning_rate=self._learning_rate).minimize(loss_cross_entropy)

    init = tf.global_variables_initializer()
    with tf.Session() as session_t:
      session_t.run(init)
      for epoch in range(n_epoches):
        avg_lost = 0.0
        total_batch = len(X)//batch_size
        for i in range(total_batch):
          s_site = i*batch_size
          if(s_site+batch_size <= len(X)):
            e_site = s_site+batch_size
          else:
            e_site = len(X)
          x_batch = X[s_site:e_site]
          y_batch = Y[s_site:e_site]
          _, lost_t = session_t.run([train_op, loss_cross_entropy],
                                    feed_dict={
              x_input: x_batch,
              y_target: y_batch
          })
          self._save_w_b_to_self(session_t, w_list, b_list)

          avg_lost += float(lost_t)/total_batch
        if epoch % display_epoches == 0:
          print_("NNET Training : Epoch"+' %04d' %
                 (epoch+1)+" Lost "+str(avg_lost))
      print_("Optimizer Finished!")

  def test_linear(self, X, Y):
    x_in = tf.placeholder("float", [None, self._sizes[0]])
    y_in = tf.placeholder("float", [None, self._sizes[-1]])
    __predict = tf.nn.sigmoid(self._MLP(x_in, self.w_list, self.b_list)) # TODO fix Loss function
    error_square = tf.square(tf.subtract(__predict, y_in))
    mean_error_square = tf.reduce_mean(tf.cast(error_square, tf.float32))
    init = tf.global_variables_initializer()
    with tf.Session() as session_t:
      session_t.run(init)
      print_("Error: "+str(session_t.run(mean_error_square,
                                         feed_dict={x_in: X,
                                                    y_in: Y})))

  def test_logical(self, X, Y):
    x_in = tf.placeholder("float", [None, self._sizes[0]])
    y_in = tf.placeholder("float", [None, self._sizes[-1]])
    __predict = tf.nn.softmax(tf.nn.sigmoid(self._MLP(x_in, self.w_list, self.b_list))) # TODO fix loss fun
    __correct = tf.equal(tf.argmax(__predict, 1), tf.argmax(y_in, 1))
    __accuracy_rate = tf.reduce_mean(tf.cast(__correct, tf.float32))
    init = tf.global_variables_initializer()
    with tf.Session() as session_t:
      session_t.run(init)
      print_("Accuracy: " + str(session_t.run(__accuracy_rate,
                                              feed_dict={x_in: X,
                                                         y_in: Y})))

  def predict(self, X):
    X_t = tf.placeholder("float", [None, self._sizes[0]])
    __predict = tf.nn.sigmoid(self._MLP(X_t, self.w_list, self.b_list)) # TODO fix loss fun
    init = tf.global_variables_initializer()
    with tf.Session() as session_t:
      session_t.run(init)
      __predict = session_t.run([__predict],
                                feed_dict={X_t: X})
      return __predict
