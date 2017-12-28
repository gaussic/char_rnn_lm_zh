#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import torch


class Dictionary(object):
    """
    词汇表，将文本中的词转换为数字id表示。
    """

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    """
    文本预处理，获取词汇表，并将字符串文本转换为数字序列。
    """

    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(path)

    def tokenize(self, path):
        """文本符号化，转换为数字id表示。"""
        assert os.path.exists(path)

        # 将新词加入到词汇表中
        with open(path, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                if len(line.strip()) == 0:  # 过滤空的行
                    continue
                words = list(line.strip()) + ['<eos>']  # 此处与原文档不同，基于字符级
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # 将字符转换为数字
        with open(path, 'r', encoding='utf-8') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                if len(line.strip()) == 0:  # 过滤空的行
                    continue
                words = list(line.strip()) + ['<eos>']  # 此处与原文档不同，基于字符级
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

    def __repr__(self):
        return "Corpus length: %d, Vocabulary size: %d" % (self.train.size(0), len(self.dictionary))


