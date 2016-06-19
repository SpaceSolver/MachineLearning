# -*- coding: utf-8 -*-

# 第一章にあるサンプルコード

import os
import sys

sys.path.append("./neural-networks-and-deep-learning/src")
import network
import mnist_loader

# 相対パスでデータを展開しようとするのでcdしておく
# import networkより前にやるとimportが失敗し始めるが、理由はよく分からん
os.chdir("neural-networks-and-deep-learning/src")

# 圧縮ファイルの中身からMNST1のデータを展開
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
