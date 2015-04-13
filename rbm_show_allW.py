#!/usr/bin/env python
# -*- coding: utf-8 -*-

# これもバイナリ！！

"""
 Restricted Boltzmann Machine (RBM)

 References :
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
     Training of Deep Networks, Advances in Neural Information Processing Systems 19, 2007

   - DeepLearningTutorials
     https://github.com/lisa-lab/DeepLearningTutorials

   for analog: change "p(v|h) = N(mu = net, sigma = 1)".
"""

import sys
import numpy
# import Image
from PIL import Image
import gzip
import cPickle
from subprocess import Popen, PIPE

numpy.seterr(all='ignore')


def sigmoid(x):
    return 1. / (1 + numpy.exp(-x))


class RBM(object):
    def __init__(self, input=None, n_visible=2, n_hidden=3,
                 W=None, hbias=None, vbias=None, numpy_rng=None):

        self.n_visible = n_visible  # num of units in visible (input) layer
        self.n_hidden = n_hidden    # num of units in hidden layer

        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)

        if W is None:
            a = 1. / n_visible

            # initialize W uniformly
            initial_W = numpy.array(numpy_rng.uniform(
                low=-a,
                high=a,
                size=(n_visible, n_hidden)))

            W = initial_W

        if hbias is None:
            hbias = numpy.zeros(n_hidden)  # initialize h bias 0

        if vbias is None:
            vbias = numpy.zeros(n_visible)  # initialize v bias 0

        self.numpy_rng = numpy_rng
        self.input = input
        self.W = W
        self.hbias = hbias
        self.vbias = vbias

        # self.params = [self.W, self.hbias, self.vbias]

    def contrastive_divergence(self, lr=0.1, k=1, input=None):
        if input is not None:
            self.input = input

        ''' CD-k '''
        ph_mean = self.mean_h_given_v(self.input)
        ph_sample = self.numpy_rng.binomial(p=ph_mean, n=1)

        nh_samples = ph_sample

        for step in xrange(k):
            nv_mean, nh_mean = self.gibbs_hvh(nh_samples)
            nh_samples = self.numpy_rng.binomial(p=nh_mean, n=1)

        self.W += lr * (numpy.dot(self.input.T, ph_mean)
                        - numpy.dot(nv_mean.T, nh_mean))
        self.vbias += lr * numpy.mean(self.input - nv_mean, axis=0)
        self.hbias += lr * numpy.mean(ph_mean - nh_mean, axis=0)

        # cost = self.get_reconstruction_cross_entropy()
        # return cost

    def mean_h_given_v(self, v0_sample):
        u = self.get_u_given_v(v0_sample)
        mean = sigmoid(u)
        return mean

    def mean_v_given_h(self, h0_sample):
        u = self.get_u_given_h(h0_sample)
        #mean = u
        mean = sigmoid(u)
        #p = self.numpy_rng.normal(u)
        return mean

    def get_u_given_v(self, v):
        u = numpy.dot(v, self.W) + self.hbias
        return u

    def get_u_given_h(self, h):
        u = numpy.dot(h, self.W.T) + self.vbias
        return u

    def gibbs_hvh(self, h0_sample):
        mean_v = self.mean_v_given_h(h0_sample)
        mean_h = self.mean_h_given_v(mean_v)

        return [mean_v, mean_h]

    def get_reconstruction_cross_entropy(self):
        pre_sigmoid_activation_h = numpy.dot(self.input, self.W) + self.hbias
        sigmoid_activation_h = sigmoid(pre_sigmoid_activation_h)

        pre_sigmoid_activation_v = (numpy.dot(sigmoid_activation_h, self.W.T)
                                    + self.vbias)
        sigmoid_activation_v = sigmoid(pre_sigmoid_activation_v)

        cross_entropy = -numpy.mean(
            numpy.sum(self.input * numpy.log(sigmoid_activation_v)
                      + (1 - self.input) * numpy.log(1 - sigmoid_activation_v),
                      axis=1))

        return cross_entropy

    def reconstruct(self, v):
        h = sigmoid(numpy.dot(v, self.W) + self.hbias)
        reconstructed_v = sigmoid(numpy.dot(h, self.W.T) + self.vbias)
        return reconstructed_v


def test_rbm(learning_rate=0.1, k=1, training_epochs=1000):
    data = numpy.array([[1, 1, 1, 0, 0, 0],
                        [1, 0, 1, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0],
                        [0, 0, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 0]])

    rng = numpy.random.RandomState(123)

    # construct RBM
    rbm = RBM(input=data, n_visible=6, n_hidden=2, numpy_rng=rng)

    # train
    for epoch in xrange(training_epochs):
        rbm.contrastive_divergence(lr=learning_rate, k=k)
        cost = rbm.get_reconstruction_cross_entropy()
        print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost

    # test
    v = numpy.array([[0, 0, 0, 1, 1, 0],
                     [1, 1, 0, 0, 0, 0]])

    print rbm.reconstruct(v)


def show_W_useimage(rbm, index=0):
    img = Image.new('RGB', (28, 28))
    min = 255
    max = -255

    for i in range(rbm.n_hidden):
        if(rbm.W[index][i] > max):
            max = rbm.W[index][i]
        if(rbm.W[index][i] < min):
            min = rbm.W[index][i]

    normalize = 255.0 / (max - min)

    for y in range(28):
        for x in range(28):
            pix = int((rbm.W[28*y + x][index] - min) * normalize)
            img.putpixel((x, y), (pix, pix, pix))
    img.show()


def show_W(p, rbm, index=0):
    # 15 img -> 4 row
    # 16 img -> 4 row
    # 17 img -> 5 row
    borderval = 0
    n_x = int(numpy.sqrt(rbm.n_hidden - 1) + 1)
    count = 0

    p.stdin.write("plot '-' matrix with image notitle\n")

    print_oneline(p, n_x, borderval=borderval)  # 初め一行の描画

    for wy in range(((rbm.n_hidden - 1) / n_x) + 1):
        for y in range(28):
            p.stdin.write("%lf " % (borderval))  # 左端の枠線
            for wx in range(n_x):
                for x in range(28):
                    if(count + wx < rbm.n_hidden):
                        p.stdin.write("%lf " % rbm.W[y*28 + x][wy*n_x + wx])
                    else:
                        p.stdin.write("%lf " % 0.0)
                p.stdin.write("%lf " % (borderval))  # 各区切り
            p.stdin.write("\n")

        print_oneline(p, n_x, borderval=borderval)
        count += n_x

    p.stdin.write("e\n")
    p.stdin.write("e\n")


def print_oneline(p, n_x, borderval=-1):
    p.stdin.write("%d " % (borderval))
    for wx in range(n_x):  # 初め一行の枠線
        for x in range(28):
            p.stdin.write("%lf " % (borderval))
        p.stdin.write("%lf " % (borderval))
    p.stdin.write("\n")


def mnist_learn(n_visible=28*28, n_hidden=10, learning_rate=0.001, k=1,
                training_epochs=10000, n_learndata=500,
                cb_low=None, cb_high=None):

    print u"""
    *****************************************
    説明: RBMによる学習のデモ

    入力: %d 個のMNISTデータ（手書き数字データ）
    学習係数 = %lf

    表示: 全ての隠れ素子（%d個）についての重み

    RBMにて素子の重みがどのように変化するか（学習が進むか）のデモ．
    *****************************************""" % (n_learndata, learning_rate, n_hidden)


    p = Popen("gnuplot", stdin=PIPE)
    # p.stdin.write("set xrange[0:%d]\n" % (25*28 + 26))
    p.stdin.write("set yrange[] reverse\n")
    # p.stdin.write("set yrange[0:%d] reverse\n" % (20*28 + 26))
    p.stdin.write("set palette define (0 'blue', 5 'white', 10 'red')\n")
    if(cb_low is not None or cb_high is not None):
        p.stdin.write("set cbrange[")
        if(cb_low is not None):
            p.stdin.write("%lf" % cb_low)
        p.stdin.write(":")
        if(cb_high is not None):
            p.stdin.write("%lf" % cb_high)
        p.stdin.write("]\n")
    # p.stdin.write("set palette cubehelix start 0.5 cycles -1.5 saturation 1\n")

    rng = numpy.random.RandomState(29439)

    f = gzip.open('../mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    train_set_raw = train_set[0][0:n_learndata]
    train_set_ans = train_set[1][0:n_learndata]

    rbm = RBM(input=train_set_raw,
              n_visible=n_visible, n_hidden=n_hidden, numpy_rng=rng)

    show_W(p, rbm, 0)
    for epoch in xrange(training_epochs):
        rbm.contrastive_divergence(lr=learning_rate, k=k)
        cost = rbm.get_reconstruction_cross_entropy()
        print str(epoch) + "/" + str(training_epochs)
        # print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost
        show_W(p, rbm, 0)

    directory = "./pickles/"
    filename = "vis" + str(n_visible) +\
        "_hid" + str(n_hidden) +\
        "_epoch" + str(training_epochs) +\
        "_ndata" + str(n_learndata) +\
        "_lr" + str(learning_rate) +\
        ".pkl"
    cPickle.dump(rbm, open(directory + filename, "wb"))

    return rbm

if __name__ == "__main__":
    argc = len(sys.argv)
    if argc != 5:
        print u"usage: sh show_allW.sh"
        exit(-1)

    arg_n_epoch = int(sys.argv[1])
    arg_n_hidden = int(sys.argv[2])
    arg_learning_rate = float(sys.argv[3])
    arg_n_learndata = int(sys.argv[4])


    #def mnist_learn(n_visible=28*28, n_hidden=10, learning_rate=0.001, k=1,
    #                training_epochs=10000, n_learndata=500,
    #                cb_low=None, cb_high=None):

    mnist_learn(n_visible=28*28, n_hidden=arg_n_hidden,
                learning_rate=arg_learning_rate,
                cb_low=-5, cb_high=5,
                training_epochs=arg_n_epoch,
                n_learndata=arg_n_learndata, k=1)
