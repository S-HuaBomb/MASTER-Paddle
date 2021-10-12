# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 10/4/2020 14:18
# @Modified Time 1: 2021-05-18 by Novio

import numpy as np
import paddle
from paddle import nn
from paddle.nn import Sequential
import paddle.nn.functional as F

# import torch

from model.backbone import ConvEmbeddingGC
from model.transformer import Encoder, Decoder
from model.initializers import xavier_uniform_


class Generator(nn.Layer):
    """
    Define standard linear + softmax generation step.
    """

    def __init__(self, hidden_dim, vocab_size):
        """

        :param hidden_dim: dim of model
        :param vocab_size: size of vocabulary
        """
        super(Generator, self).__init__()
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        return self.fc(x)


class MultiInputSequential(Sequential):
    def forward(self, *_inputs):
        for m_module_index, m_module in enumerate(self):
            if m_module_index == 0:
                m_input = m_module(*_inputs)
            else:
                m_input = m_module(m_input)
        return m_input


class MASTER(nn.Layer):
    """
     A standard Encoder-Decoder MASTER architecture.
    """

    def __init__(self, common_kwargs, backbone_kwargs, encoder_kwargs, decoder_kwargs):

        super(MASTER, self).__init__()
        self.with_encoder = common_kwargs['with_encoder']
        self.padding_symbol = 0
        self.sos_symbol = 1
        self.eos_symbol = 2
        self.build_model(common_kwargs, backbone_kwargs, encoder_kwargs, decoder_kwargs)
        for m_parameter in self.parameters():
            if m_parameter.dim() > 1:
                xavier_uniform_(m_parameter)

    def build_model(self, common_kwargs, backbone_kwargs, encoder_kwargs, decoder_kwargs):
        target_vocabulary = common_kwargs['n_class'] + 4
        heads = common_kwargs['multiheads']
        dimensions = common_kwargs['model_size']
        self.conv_embedding_gc = ConvEmbeddingGC(
            gcb_kwargs=backbone_kwargs['gcb_kwargs'],
            in_channels=backbone_kwargs['in_channels']
        )
        # with encoder: cnn(+gc block) + transformer encoder + transformer decoder
        # without encoder: cnn(+gc block) + transformer decoder
        self.encoder = Encoder(
            _with_encoder=common_kwargs['with_encoder'],
            _multi_heads_count=heads,
            _dimensions=dimensions,
            _stacks=encoder_kwargs['stacks'],
            _dropout=encoder_kwargs['dropout'],
            _feed_forward_size=encoder_kwargs['feed_forward_size'],
            _share_parameter=encoder_kwargs.get('share_parameter', 'false'),
        )
        self.encode_stage = nn.Sequential(self.conv_embedding_gc, self.encoder)
        self.decoder = Decoder(
            _multi_heads_count=heads,
            _dimensions=dimensions,
            _stacks=decoder_kwargs['stacks'],
            _dropout=decoder_kwargs['dropout'],
            _feed_forward_size=decoder_kwargs['feed_forward_size'],
            _n_classes=target_vocabulary,
            _padding_symbol=self.padding_symbol,
            _share_parameter=decoder_kwargs.get('share_parameter', 'false')
        )
        self.generator = Generator(dimensions, target_vocabulary)
        self.decode_stage = MultiInputSequential(self.decoder, self.generator)

    def eval(self):
        self.conv_embedding_gc.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.generator.eval()

    def forward(self, _source, _target):
        encode_stage_result = self.encode_stage(_source)
        decode_stage_result = self.decode_stage(_target, encode_stage_result)
        return decode_stage_result

    def model_parameters(self):
        model_parameters = filter(lambda p: p.trainable, self.parameters())
        params = sum([np.prod(p.shape) for p in model_parameters])
        return params


def predict(_memory, _source, _decode_stage, _max_length, _sos_symbol, _eos_symbol, _padding_symbol):
    batch_size = _source.shape[0]
    device = paddle.get_device()  # _source.device
    to_return_label = \
        paddle.ones((batch_size, _max_length + 2), dtype=paddle.int64) * _padding_symbol
    probabilities = paddle.ones((batch_size, _max_length + 2), dtype=paddle.float32)
    to_return_label[:, 0] = _sos_symbol
    for i in range(_max_length + 1):
        m_label = _decode_stage(to_return_label, _memory)
        m_probability = F.softmax(m_label, axis=-1)
        # m_max_probs, m_next_word = paddle.max(m_probability, axis=-1)
        m_max_probs = paddle.max(m_probability, axis=-1)
        m_next_word = paddle.argmax(m_probability, axis=-1)
        to_return_label[:, i + 1] = m_next_word[:, i]
        probabilities[:, i + 1] = m_max_probs[:, i]
    eos_position_y, eos_position_x = paddle.nonzero(to_return_label == _eos_symbol, as_tuple=True)
    if len(eos_position_y) > 0:
        eos_position_y_index = eos_position_y[0]
        for m_position_y, m_position_x in zip(eos_position_y, eos_position_x):
            if eos_position_y_index == m_position_y:
                to_return_label[m_position_y, m_position_x + 1:] = _padding_symbol
                probabilities[m_position_y, m_position_x + 1:] = 1
                eos_position_y_index += 1

    return to_return_label, probabilities
