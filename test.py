#!/usr/bin/python
# -*- coding: utf-8 -*-

from data import *

train_dir = 'data/sanguoyanyi.txt'
corpus = Corpus(train_dir)
print("三国演义：", corpus)

train_dir = 'data/weicheng.txt'
corpus = Corpus(train_dir)
print("围城：", corpus)
