MNISTデータを用いたRBMの学習

# 概要
MNISTデータを用いてRBMを学習させ，その重みを表示する

# 環境
* sh

* python
  * Python Image Library
  * cPickle
  * gzip
  * subprocess
    * Popen
    * PIPE

* gnuplot

# 各プログラムについて
## 実行方法
1. （任意）show_allW.sh(もしくはshow_oneW.sh)のパラメータを設定
2. ```sh show_allW.sh```（もしくはもう片方）を実行
3. 満足したら ^C で中断

# 各モデルについて
## show_allW
全ての素子の重みを表示する．
パラメータのデフォルト値は以下のとおり．
* 可視層: 28*28=784素子 （各素子は画素値に対応）
* 隠れ層: 500素子
* 学習回数: 1000回
* 学習係数: 0.001

## show_oneW
全ての素子の重みを表示する．
パラメータのデフォルト値は以下のとおり．
* 可視層: 28*28=784素子 （各素子は画素値に対応）
* 隠れ層: 500素子
* 学習回数: 1000回
* 学習係数: 0.001
* 着目素子: 第0素子

# おすすめの設定
## どんな動きかをざっくり見たい
`show_allW.sh`を以下の設定で最後まで回す．
* 隠れ層: 20素子
* 学習係数: 0.001

```
rbm_n_hidden=500  # 隠れ素子数
n_learning_rate=0.0001  # 学習係数
n_learndata=2000  # 入力データ数
```

## じっくり見たい
`show_allW.sh`をデフォルト設定で最後まで回す．時間がかかります．

## もっと大きく見たい
`show_oneW.sh`を好みの設定で最後まで回す．
素子番号が特定できているなら，着目素子をそれにする．
（素子番号は `show_allW.sh` の表示で左上から右に 0, 1, 2...）