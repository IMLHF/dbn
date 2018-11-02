import tensorflow as tf
import numpy as np
from print_ import print_


class RBM(object):
  def __init__(self, name, isize, osize, learning_rate=0.01, CDk=1):
    self._name = name
    self._input_size = isize
    self._output_size = osize
    self._weight = tf.Session().run(tf.random_normal([isize, osize]))
    self._v_bias = tf.Session().run(tf.random_normal([isize]))
    self._h_bias = tf.Session().run(tf.random_normal([osize]))
    self._learning_rate = learning_rate
    self._CDk = CDk

  def _load_w_b_from_self(self):
    weight = tf.get_variable(self._name+"_weight", [self._input_size, self._output_size],
                             # initializer=tf.random_normal_initializer()))
                             initializer=tf.constant_initializer(self._weight))
    v_bias = tf.get_variable(self._name+"_v_bias", [self._input_size],
                             initializer=tf.constant_initializer(self._v_bias))
    h_bias = tf.get_variable(self._name+"_h_bias", [self._output_size],
                             initializer=tf.constant_initializer(self._h_bias))
    return weight, v_bias, h_bias

  def _save_w_b_to_self(self,session,weight,v_bias,h_bias):
    self._weight=session.run(weight)
    self._v_bias=session.run(v_bias)
    self._h_bias=session.run(h_bias)

  def _0_1_sample_given_p(self, p):
    return tf.nn.relu(tf.sign(p - tf.random_uniform(tf.shape(p))))

  def _predict_h_given_v(self, v, weight, h_bias):
    prob = tf.nn.sigmoid(
        tf.matmul(v, weight)+h_bias)
    return prob, self._0_1_sample_given_p(prob)

  def _predict_v_given_h(self, h, weight, v_bias):
    prob = tf.nn.sigmoid(
        tf.matmul(h, tf.transpose(weight))+v_bias)
    return prob, self._0_1_sample_given_p(prob)

  def _CDk_f(self, vis,weight,v_bias,h_bias):
    v0_prob = vis
    h0_prob, h0_sample = self._predict_h_given_v(v0_prob,weight,h_bias)
    hk_sample = h0_sample
    hk_prob = h0_sample
    for i in range(self._CDk):
      # vk_prob, vk_sample = self._predict_v_given_h(hk_prob,weight,v_bias)  # 隐层使用概率
      vk_prob, vk_sample = self._predict_v_given_h(hk_sample,weight,v_bias)  # 隐层使用逻辑单元
      hk_prob, hk_sample = self._predict_h_given_v(vk_prob,weight,h_bias)   # 可视层使用概率代替

    delta_w_positive = tf.matmul(tf.transpose(v0_prob), h0_prob)
    delta_w_negative = tf.matmul(tf.transpose(vk_prob), hk_prob)

    delta_w = tf.subtract(delta_w_positive, delta_w_negative) / \
        tf.to_float(tf.shape(v0_prob)[0])
    delta_vb = tf.reduce_mean(v0_prob-vk_prob, 0)
    delta_hb = tf.reduce_mean(h0_prob-hk_prob, 0)

    return delta_w, delta_vb, delta_hb

  def _rbm_train_epoche(self, vis,weight,v_bias,h_bias):
    delta_w, delta_vb, delta_hb = self._CDk_f(vis,weight,v_bias,h_bias)
    # update rbm parameters
    update_w_op = weight.assign_add(self._learning_rate*delta_w)
    update_vb_op = v_bias.assign_add(self._learning_rate*delta_vb)
    update_hb_op = h_bias.assign_add(self._learning_rate*delta_hb)

    return [update_w_op, update_vb_op, update_hb_op]

  def reconstruct(self, vis,weight,v_bias,h_bias):
    _, h_samp = self._predict_h_given_v(vis,weight,h_bias)
    for i in range(self._CDk):
      v_recon, _ = self._predict_v_given_h(h_samp,weight,v_bias)
      _, h_samp = self._predict_h_given_v(v_recon,weight,h_bias)
    return tf.reduce_mean(tf.square(vis - v_recon))

  def rbm_train(self, data_x, batch_size=128, n_epoches=1):
    x_in = tf.placeholder(tf.float32, shape=[None, self._input_size])
    weight, v_bias, h_bias = self._load_w_b_from_self()
    rbm_pretrain = self._rbm_train_epoche(x_in,weight,v_bias,h_bias)
    x_loss = self.reconstruct(x_in,weight,v_bias,h_bias)

    n_data = np.shape(data_x)[0]
    n_batches = n_data // batch_size

    # # whether or not plot
    # if self.plot is True:
    #     plt.ion() # start the interactive mode of plot
    #     plt.figure(1)

    errs = []
    with tf.Session() as sess:
      init = tf.global_variables_initializer()
      sess.run(init)
      for epoch in range(n_epoches):
        mean_lost = []
        for i_batche in range(n_batches):
          s_site = i_batche*batch_size
          if(s_site+batch_size <= len(data_x)):
            e_site = s_site+batch_size
          else:
            e_site = len(data_x)
          batch_x = data_x[s_site:e_site]
          sess.run(rbm_pretrain, feed_dict={x_in: batch_x})
          self._save_w_b_to_self(sess,weight,v_bias,h_bias)
          recon_lost = sess.run(x_loss, feed_dict={x_in: batch_x})
          mean_lost.append(recon_lost)
        errs.append(np.mean(mean_lost))
        print_('%s Training : Epoch %04d lost %g' % (self._name, epoch, errs[-1]))
        # # plot ?
        # if plt.fignum_exists(1):
        #     plt.plot(range(epoch+1),errs,'-r')
    self.train_error = errs
    return errs

  def rbm_forward(self, vis):
    assert np.shape(vis)[1] == self._input_size
    x_up, _ = self._predict_h_given_v(vis,self._weight,self._h_bias)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      return sess.run(x_up)

  def get_param(self):
      return self._weight, self._v_bias, self._h_bias
