#!/bin/sh

#    mnist_learn(n_visible=28*28, n_hidden=500,
#                cb_low=-5, cb_high=5,
#                training_epochs=1000,
#                n_learndata=2000, k=1)

# 

# **** パラメータ設定 ****
rbm_n_epoch=1000  # 学習回数
rbm_n_hidden=40  # 隠れ素子数
n_learning_rate=0.001  # 学習係数
n_learndata=80  # 入力データ数

# デフォルト値
# rbm_n_hidden=500
# n_learndata=2000
# ************************

python rbm_show_allW.py $rbm_n_epoch $rbm_n_hidden $n_learning_rate $n_learndata
