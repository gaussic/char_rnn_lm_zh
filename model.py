#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch.nn as nn
from torch.autograd import Variable


class RNNModel(nn.Module):
    """基于RNN的语言模型，包含一个encoder，一个rnn模块，一个decoder。"""

    def __init__(self, config):
        super(RNNModel, self).__init__()

        v_size = config.vocab_size
        em_dim = config.embedding_dim

        rnn_type = config.rnn_type
        hi_dim = config.hidden_dim
        n_layers = config.num_layers

        dropout = config.dropout
        tie_weights = config.tie_weights

        self.drop = nn.Dropout(dropout)  # dropout层
        self.encoder = nn.Embedding(v_size, em_dim)  # encoder是一个embedding层

        if rnn_type in ['RNN', 'LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(em_dim, hi_dim, n_layers, dropout=dropout)
        else:
            raise ValueError("""'rnn_type' error, options are ['RNN', 'LSTM', 'GRU']""")

        self.decoder = nn.Linear(hi_dim, v_size)  # decoder将向量映射到字

        # tie_weights将encoder和decoder的参数绑定为同一参数，在以下两篇论文中得到了证明：
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # 以及
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if hi_dim != em_dim:  # 这两个维度必须相同
                raise ValueError('When using the tied flag, hi_dim must be equal to em_dim')
            self.decoder.weight = self.encoder.weight

        self.init_weights()  # 初始化权重

        self.rnn_type = rnn_type
        self.hi_dim = hi_dim
        self.n_layers = n_layers

    def forward(self, inputs, hidden):
        emb = self.drop(self.encoder(inputs))  # encoder + dropout
        output, hidden = self.rnn(emb, hidden)  # output维度：(seq_len, batch_size, hidden_dim)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))  # 展平，映射
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden  # 复原

    def init_weights(self):
        """权重初始化，如果tie_weights，则encoder和decoder权重是相同的"""
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.fill_(0)

    def init_hidden(self, bsz):
        """初始化隐藏层，与batch_size相关"""
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':  # lstm：(h0, c0)
            return (Variable(weight.new(self.n_layers, bsz, self.hi_dim).zero_()),
                    Variable(weight.new(self.n_layers, bsz, self.hi_dim).zero_()))
        else:  # gru 和 rnn：h0
            return Variable(weight.new(self.n_layers, bsz, self.hi_dim).zero_())
