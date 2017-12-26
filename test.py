#!/usr/bin/python
# -*- coding: utf-8 -*-

from preprocessing_zh import Corpus, LMDataset


train_dir = 'data/weicheng.txt'
vocab_dir = 'data/weicheng.vocab.txt'

corpus = Corpus(train_dir, vocab_dir)
print(corpus)

print(corpus.to_word(corpus.data[100:200]))

train_data = LMDataset(corpus.data, 10, 30)

print(len(train_data))

print(train_data[0][0])


print(train_data[0][1].shape)
