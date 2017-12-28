#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
from collections import Counter


class Corpus(object):
    def __init__(self, train_dir, vocab_dir):
        assert os.path.exists(train_dir)

        data = list(open(train_dir, 'r', encoding='utf-8').read().replace('\n', ''))

        if not os.path.exists(vocab_dir):
            self._build_vocab(data, vocab_dir)

        self.words = open(vocab_dir, 'r', encoding='utf-8').read().strip().split('\n')
        self.word_to_id = dict(zip(self.words, range(len(self.words))))

        data = [self.word_to_id[x] for x in data if x in self.word_to_id]
        self.data = np.array(data)

    def _build_vocab(self, data, vocab_dir):
        count_pairs = Counter(data).most_common()
        words, _ = list(zip(*count_pairs))
        open(vocab_dir, 'w', encoding='utf-8').write('\n'.join(words) + '\n')

    def to_word(self, ids):
        return list(map(lambda x: self.words[x], ids))

    def __repr__(self):
        return 'Corpus length: %d, Vocabulary size: %d.' % (len(self.data), len(self.words))


class LMDataset(object):
    def __init__(self, raw_data, batch_size, seq_len):
        num_batch = len(raw_data) // (batch_size * seq_len)

        data = raw_data[:(num_batch * batch_size * seq_len)]
        data = data.reshape(num_batch, batch_size, -1).swapaxes(1, 2)

        target = raw_data[1:(num_batch * batch_size * seq_len + 1)]
        target = target.reshape(num_batch, batch_size, -1).swapaxes(1, 2).reshape(num_batch, -1)

        self.data = data
        self.target = target

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return 'Num of batches: %d, Batch Shape: %s' % (len(self.data), self.data[0].shape)
