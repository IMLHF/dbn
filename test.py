import tensorflow as tf
import numpy as np
from rbm import RBM
from rbm_0 import RBM as RBM_0
from dbn import DBN
from nn import NN
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image, ImageDraw,ImageFont
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# sizes=['a','b','c','d','e']
# for i,size in enumerate(sizes[1:]):
#       print("RBM_%d" % i,sizes[i],size)


class Test(object):
  def __init__(self):
    pass

  def test_all(self, n):
    _dbn=DBN([784,1000,500,250,30],learning_rate=0.01,cd_k=1)
    _dbn.pretrain(mnist.train.images,128,50)

    _nnet = NN([784, 1000, 500, 250, 30, 250, 500, 1000, 784], 0.01, 128, 50)
    _nnet.load_from_dbn_to_reconstructNN(_dbn)
    _nnet.train(mnist.train.images, mnist.train.images)
    _nnet.test_linear(mnist.test.images, mnist.test.images)

    x_in = mnist.test.images[:30]
    _predict = _nnet.predict(x_in)
    _predict_img = np.concatenate(np.reshape(_predict, [-1, 28, 28]), axis=1)
    x_in = np.concatenate(np.reshape(x_in, [-1, 28, 28]), axis=1)
    img = Image.fromarray(
        (1.0-np.concatenate((_predict_img, x_in), axis=0))*255.0)
    img = img.convert('L')
    img.save(str(n)+'_.jpg')
    img2 = Image.fromarray(
        (np.concatenate((_predict_img, x_in), axis=0))*255.0)
    img2 = img2.convert('L')
    img2.save(str(n)+'.jpg')

    nnet_encoder=NN()
    nnet_encoder.load_layers_from_NN(_nnet,0,4)
    # featrue=nnet_encoder.predict(mnist.test.images)
    nnet_decoder=NN()
    nnet_decoder.load_layers_from_NN(_nnet,5,8)
    # pic=nnet_decoder.predict(featrue)

  def nn_test2(self):
    _nnet = NN([784, 1000, 500, 250, 30, 250, 500, 1000, 784], 0.01, 128, 100)
    _nnet.train(mnist.train.images, mnist.train.images)
    _nnet.test_linear(mnist.test.images, mnist.test.images)
    _nnet.predict(mnist.test.images[:10])

  def rbm_test(self):
    _rbm = RBM("RBM_test", 784, 10)
    _rbm.rbm_train(mnist.train.images, 128, 10)
    w, vb, hb = _rbm.get_param()
    print(hb)

  def rbm_0_test(self):
    _rbm_0 = RBM_0("RBM_test", 784, 10)
    _rbm_0.rbm_train(mnist.train.images, 128, 10)
    w, vb, hb = _rbm_0.get_param()
    print(hb)

  def nn_test(self):
    _nnet = NN([784, 256, 256, 10], 0.01, 128, 10)
    _nnet.train(mnist.train.images, mnist.train.labels)
    _nnet.test_logical(mnist.test.images, mnist.test.labels)

  def test_print(self):
    a = 0
    msg = '%04d' % a + "dfd"
    print(msg)

  def merge_img(self):
    white_pic_list = []
    for i in range(1, 13):
      index_img = Image.fromarray(255.0*np.ones([56, 100]))
      draw = ImageDraw.Draw(index_img)
      ft=ImageFont.truetype("VeraMoBI.ttf",20)
      draw.text((10, 10), "NO."+str(i), fill=(0),font=ft)
      msg_img=Image.open("report/"+str(i)+"_.jpg")
      tmp_pic = np.concatenate((index_img,msg_img), axis=1)
      white_pic_list.append(tmp_pic)

    white_pic_seperated_list = []
    for i in range(len(white_pic_list)):
      white_pic_seperated_list.append(
          125.0*np.ones([7, np.shape(white_pic_list[0])[1]]))
      white_pic_seperated_list.append(white_pic_list[i])

    white_pic = np.concatenate(white_pic_seperated_list, axis=0)

    imgw = Image.fromarray(white_pic)
    imgw = imgw.convert('L')
    imgw.save('report/all_white.jpg')
    imgb = Image.fromarray((255.0-white_pic))
    imgb = imgb.convert('L')
    imgb.save('report/all_black.jpg')


if __name__ == "__main__":
  test = Test()
  # test.rbm_test()
  # test.test_all(11)
  # test.nn_test2()
  # test.test_print()
  test.merge_img()
