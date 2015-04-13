#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 Restricted Boltzmann Machine (RBM)

 References :
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

   - DeepLearningTutorials
   https://github.com/lisa-lab/DeepLearningTutorials
"""

import sys
import numpy
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
        ph_mean, ph_sample = self.sample_h_given_v(self.input)

        chain_start = ph_sample

        for step in xrange(k):
            if step == 0:
                (nv_means, nv_samples, nh_means, nh_samples) = self.gibbs_hvh(chain_start)
            else:
                (nv_means, nv_samples, nh_means, nh_samples) = self.gibbs_hvh(nh_samples)

        # chain_end = nv_samples

        self.W += lr * (numpy.dot(self.input.T, ph_sample)
                        - numpy.dot(nv_samples.T, nh_means))
        self.vbias += lr * numpy.mean(self.input - nv_samples, axis=0)
        self.hbias += lr * numpy.mean(ph_sample - nh_means, axis=0)

        # cost = self.get_reconstruction_cross_entropy()
        # return cost

    def sample_h_given_v(self, v0_sample):
        h1_mean = self.propup(v0_sample)
        h1_sample = self.numpy_rng.binomial(size=h1_mean.shape,   # discrete: binomial
                                            n=1,
                                            p=h1_mean)

        return [h1_mean, h1_sample]

    def sample_v_given_h(self, h0_sample):
        v1_mean = self.propdown(h0_sample)
        v1_sample = self.numpy_rng.binomial(size=v1_mean.shape,   # discrete: binomial
                                            n=1,
                                            p=v1_mean)

        return [v1_mean, v1_sample]

    def propup(self, v):
        pre_sigmoid_activation = numpy.dot(v, self.W) + self.hbias
        return sigmoid(pre_sigmoid_activation)

    def propdown(self, h):
        pre_sigmoid_activation = numpy.dot(h, self.W.T) + self.vbias
        return sigmoid(pre_sigmoid_activation)

    def gibbs_hvh(self, h0_sample):
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)

        return [v1_mean, v1_sample,
                h1_mean, h1_sample]

    def get_reconstruction_cross_entropy(self):
        pre_sigmoid_activation_h = numpy.dot(self.input, self.W) + self.hbias
        sigmoid_activation_h = sigmoid(pre_sigmoid_activation_h)

        pre_sigmoid_activation_v = numpy.dot(sigmoid_activation_h, self.W.T) + self.vbias
        sigmoid_activation_v = sigmoid(pre_sigmoid_activation_v)

        cross_entropy = -1*(numpy.mean(
            numpy.sum(self.input * numpy.log(sigmoid_activation_v)
                      + (1 - self.input) * numpy.log(1 - sigmoid_activation_v),
                      axis=1
                  )
        ))

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


def show_W(p, rbm, index = 0):
    p.stdin.write("plot '-' matrix with image\n")
    for y in range(28):
        for x in range(28):
            p.stdin.write("%lf " % rbm.W[28*y + x][index])
        p.stdin.write("\n")

    p.stdin.write("e\n")
    p.stdin.write("e\n")


def mnist_learn(learning_rate=0.01, k=1,
                training_epochs=1000, n_learndata=1000,
                view_index=0, cb_low=None, cb_high=None):

    print u"""
    *****************************************
    説明: RBMによる学習のデモ

    入力: %d 個のMNISTデータ（手書き数字データ）
    学習係数 = %lf

    表示: 第 %d 番目の素子の重み

    RBMにて素子の重みがどのように変化するか（学習が進むか）のデモ．
    *****************************************""" % (n_learndata, learning_rate, view_index)

    p = Popen("gnuplot", stdin=PIPE)
    p.stdin.write("set xrange[-1:%d]\n" % (28))
    p.stdin.write("set yrange[-1:%d] reverse\n" % (28))
    if(cb_low is not None or cb_high is not None):
        p.stdin.write("set cbrange[")
        if(cb_low is not None):
            p.stdin.write("%lf" % cb_low)
        p.stdin.write(":")
        if(cb_high is not None):
            p.stdin.write("%lf" % cb_high)
        p.stdin.write("]\n")

    rng = numpy.random.RandomState(123)

    f = gzip.open('../mnist.pkl.gz','rb')
    (train_set, valid_set, test_set) = cPickle.load(f)
    train_set_raw = train_set[0][0:n_learndata]
    train_set_ans = train_set[1][0:n_learndata]

    rbm = RBM(input=train_set_raw, n_visible=28*28, n_hidden=500, numpy_rng=rng)
    show_W(p, rbm, view_index)
    for epoch in xrange(training_epochs):
        rbm.contrastive_divergence(lr=learning_rate, k=k)
        cost = rbm.get_reconstruction_cross_entropy()
        print str(epoch) + "/" + str(training_epochs)
        # print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost
        show_W(p, rbm, view_index)

    f = open('./W.dat', 'w')
    for i in range(rbm.n_visible):
        for j in range(rbm.n_hidden):
            f.write(str(rbm.W[i][j]))
            f.write(" ")
        f.write("\n")
    f.close()

if __name__ == "__main__":

    argc = len(sys.argv)
    if argc != 6:
        print u"usage: sh show_oneW.sh"
        exit(-1)

    arg_n_epoch = int(sys.argv[1])
    arg_n_hidden = int(sys.argv[2])
    arg_learning_rate = float(sys.argv[3])
    arg_n_learndata = int(sys.argv[4])
    arg_view_index = int(sys.argv[5])

    #def mnist_learn(learning_rate=0.01, k=1,
    #                training_epochs=1000, n_learndata=1000,
    #                view_index=0, cb_low=None, cb_high=None):

    mnist_learn(learning_rate=arg_learning_rate, k=1,
                training_epochs=arg_n_epoch, n_learndata=arg_n_learndata,
                view_index=arg_view_index)
