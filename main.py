#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import os
import math
import time
from datetime import timedelta

import torch
import torch.nn as nn
from torch.autograd import Variable

from model import RNNModel
from data_zh import Corpus

train_dir = 'data/sanguoyanyi.txt'
filename = str(os.path.basename(train_dir).split('.')[0])

# 用于保存模型参数
save_dir = 'checkpoints/' + filename
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
model_name = filename + '_{}.pt'

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch Chinese Language Model')
parser.add_argument('--mode', type=str, default='train', help='train or gen.')
parser.add_argument('--epoch', type=int, default=3, help='the epoch of parameter to be loaded.')
args = parser.parse_args()


class Config(object):
    """RNNLM模型配置项"""
    embedding_dim = 200  # 词向量维度

    rnn_type = 'LSTM'  # 支持RNN/LSTM/GRU
    hidden_dim = 200  # 隐藏层维度
    num_layers = 2  # RNN 层数

    dropout = 0.5  # 丢弃概率
    tie_weights = True  # 是否绑定参数

    batch_size = 10  # 每一批数据量
    seq_len = 30  # 序列长度

    clip = 0.25  # 用于梯度规范化
    learning_rate = 20  # 初始学习率

    num_epochs = 50  # 迭代轮次
    log_interval = 500  # 每隔多少个批次输出一次状态
    save_interval = 3  # 每个多少个轮次保存一次参数


def batchify(data, bsz):
    """返回数据维度为(nbatch, batch_size)"""
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)  # 去除多余部分
    data = data.view(bsz, -1).t().contiguous()  # 将数据按照bsz切分
    return data


def get_batch(source, i, seq_len, evaluation=False):
    """
    获取一个batch
    data: (seq_len, batch_size)
    target: (seq_len * batch_size)
    """
    seq_len = min(seq_len, len(source) - 1 - i)
    data = Variable(source[i:(i + seq_len)], volatile=evaluation)
    target = Variable(source[(i + 1):(i + 1 + seq_len)].view(-1))  # 为训练方便，展平
    if use_cuda:
        data, target = data.cuda(), target.cuda()
    return data, target


def repackage_hidden(h):
    """用新的变量重新包装隐藏层，将它们从历史中分离。"""
    if type(h) == Variable:  # rnn/gru
        return Variable(h.data)
    else:  # lstm
        return tuple(repackage_hidden(v) for v in h)


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def generate(model, idx2word, word_len=200, temperature=1.0):
    """生成一定数量的文本，temperature结合多项式分布可增添抽样的多样性。"""
    model.eval()
    hidden = model.init_hidden(1)  # batch_size为1
    inputs = Variable(torch.rand(1, 1).mul(len(idx2word)).long(), volatile=True)  # 随机选取一个字作为开始
    if use_cuda:
        inputs = inputs.cuda()

    word_list = []
    for i in range(word_len):  # 逐字生成
        output, hidden = model(inputs, hidden)
        word_weights = output.squeeze().data.div(temperature).exp().cpu()

        # 基于词的权重，对其再进行一次抽样，增添其多样性，如果不使用此法，会导致常用字的无限循环
        word_idx = torch.multinomial(word_weights, 1)[0]
        inputs.data.fill_(word_idx)  # 将新生成的字赋给inputs
        word = idx2word[word_idx]
        word_list.append(word)
    return word_list


def train():
    # 载入数据与配置模型
    print("Loading data...")
    corpus = Corpus(train_dir)
    print(corpus)

    config = Config()
    config.vocab_size = len(corpus.dictionary)
    train_data = batchify(corpus.train, config.batch_size)
    train_len = train_data.size(0)
    seq_len = config.seq_len

    print("Configuring model...")
    model = RNNModel(config)
    if use_cuda:
        model.cuda()
    print(model)

    criterion = nn.CrossEntropyLoss()
    lr = config.learning_rate  # 初始学习率
    start_time = time.time()

    print("Training and generating...")
    for epoch in range(1, config.num_epochs + 1):  # 多轮次训练
        total_loss = 0.0
        model.train()  # 在训练模式下dropout才可用。
        hidden = model.init_hidden(config.batch_size)  # 初始化隐藏层参数

        for ibatch, i in enumerate(range(0, train_len - 1, seq_len)):
            data, targets = get_batch(train_data, i, seq_len)  # 取一个批次的数据
            # 在每批开始之前，将隐藏的状态与之前产生的结果分离。
            # 如果不这样做，模型会尝试反向传播到数据集的起点。
            hidden = repackage_hidden(hidden)
            model.zero_grad()

            output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, config.vocab_size), targets)
            loss.backward()  # 反向传播

            # `clip_grad_norm` 有助于防止RNNs/LSTMs中的梯度爆炸问题。
            torch.nn.utils.clip_grad_norm(model.parameters(), config.clip)
            for p in model.parameters():  # 梯度更新
                p.data.add_(-lr, p.grad.data)

            total_loss += loss.data  # loss累计

            if ibatch % config.log_interval == 0 and ibatch > 0:  # 每隔多少个批次输出一次状态
                cur_loss = total_loss[0] / config.log_interval
                elapsed = get_time_dif(start_time)
                print("Epoch {:3d}, {:5d}/{:5d} batches, lr {:2.3f}, loss {:5.2f}, ppl {:8.2f}, time {}".format(
                    epoch, ibatch, train_len // seq_len, lr, cur_loss, math.exp(cur_loss), elapsed))
                total_loss = 0.0
        lr /= 4.0  # 在一轮迭代完成后，尝试缩小学习率

        # 每隔多少轮次保存一次模型参数
        if epoch % config.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, model_name.format(epoch)))

        print(''.join(generate(model, corpus.dictionary.idx2word)))


def generate_flow(epoch=3):
    """读取存储的模型，生成新词"""
    corpus = Corpus(train_dir)
    config = Config()
    config.vocab_size = len(corpus.dictionary)

    model = RNNModel(config)
    model_file = os.path.join(save_dir, model_name.format(epoch))
    assert os.path.exists(model_file), 'File %s does not exist.' % model_file
    model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    word_list = generate(model, corpus.dictionary.idx2word, word_len=50)
    print(''.join(word_list))


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'gen':
        generate_flow(args.epoch)
    else:
        raise ValueError("""mode error.""")
