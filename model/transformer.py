# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Email: yuwenwen62@gmail.com
# @Created Time: 10/3/2020 1:27 PM

import copy
import math

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.nn import Dropout


def clones(_to_clone_module, _clone_times, _is_deep=True):
    """Produce N identical layers."""
    copy_method = copy.deepcopy if _is_deep else copy.copy
    return nn.ModuleList([copy_method(_to_clone_module) for _ in range(_clone_times if _is_deep else 1)])


def subsequent_mask(_target):
    batch_size = _target.size(0)
    sequence_length = _target.size(1)
    return paddle.tril(paddle.ones((batch_size, 1, sequence_length, sequence_length), dtype=paddle.bool))


class MultiHeadAttention():  # 继承自 torch.jit.ScriptModule
    def __init__(self, _multi_attention_heads, _dimensions, _dropout=0.1):
        """

        :param _multi_attention_heads: number of self attention head
        :param _dimensions: dimension of model
        :param _dropout:
        """
        super(MultiHeadAttention, self).__init__()

        assert _dimensions % _multi_attention_heads == 0
        # requires d_v = d_k, d_q = d_k = d_v = d_m / h
        self.d_k = int(_dimensions / _multi_attention_heads)
        self.h = _multi_attention_heads
        self.linears = clones(nn.Linear(_dimensions, _dimensions), 4)  # (q, k, v, last output layer)
        self.attention = None
        self.dropout = nn.Dropout(p=_dropout)

    def dot_product_attention(self, _query, _key, _value, _mask):
        """
        Compute 'Scaled Dot Product Attention

        :param _query: (N, h, seq_len, d_q), h is multi-head
        :param _key: (N, h, seq_len, d_k)
        :param _value: (N, h, seq_len, d_v)
        :param _mask: None or (N, 1, seq_len, seq_len), 0 will be replaced with -1e9
        :return:
        """

        d_k = _value.size(-1)
        score = paddle.matmul(_query, _key.transpose([-2, -1])) / math.sqrt(d_k)  # (N, h, seq_len, seq_len)
        if _mask is not None:
            scoren = score.numpy()
            maskn = (_mask == 0).numpy()
            scoren[maskn] = -1e9
            score = paddle.to_tensor(scoren)
            # score = score.masked_fill(_mask == 0, -1e9)  # score (N, h, seq_len, seq_len)
        p_attn = F.softmax(score, axis=-1)
        # (N, h, seq_len, d_v), (N, h, seq_len, seq_len)
        return paddle.matmul(p_attn, _value), p_attn

    def forward(self, _query, _key, _value, _mask):
        batch_size = _query.size(0)

        # do all the linear projections in batch from d_model => h x d_k
        # (N, seq_len, d_m) -> (N, seq_len, h, d_k) -> (N, h, seq_len, d_k)
        _query, _key, _value = \
            [l(x).reshape([batch_size, -1, self.h, self.d_k]).transpose([1, 2])
             for l, x in zip(self.linears, (_query, _key, _value))]

        # apply attention on all the projected vectors in batch.
        # (N, h, seq_len, d_v), (N, h, seq_len, seq_len)
        product_and_attention = self.dot_product_attention(_query, _key, _value, _mask=_mask)
        x = product_and_attention[0]
        # self.attention = self.dropout(product_and_attention[1])

        # "Concat" using a view and apply a final linear.
        # (N, seq_len, d_m)
        x = x.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.h * self.d_k)

        # (N, seq_len, d_m)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Layer):
    def __init__(self, _dimensions, _feed_forward_dimensions, _dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(_dimensions, _feed_forward_dimensions)
        self.w_2 = nn.Linear(_feed_forward_dimensions, _dimensions)
        self.dropout = nn.Dropout(p=_dropout)

    def forward(self, _input_tensor):
        return self.w_2(self.dropout(F.relu(self.w_1(_input_tensor))))


class PositionalEncoding():
    """Implement the PE function."""

    def __init__(self, _dimensions, _dropout=0.1, _max_len=5000):
        """

        :param _dimensions:
        :param _dropout:
        :param _max_len:
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=_dropout)

        # Compute the positional encodings once in log space.
        pe = paddle.zeros([_max_len, _dimensions])
        position = paddle.arange(0, _max_len).unsqueeze(1).astype(paddle.float32)
        div_term = paddle.exp(paddle.arange(0, _dimensions, 2).astype(paddle.float32) *
                              -(math.log(10000.0) / _dimensions))
        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, _input_tensor):
        _input_tensor = _input_tensor + self.pe[:, :_input_tensor.size(1)]  # pe 1 5000 512
        return self.dropout(_input_tensor)


class Encoder(nn.Layer):
    def __init__(self, _with_encoder, _multi_heads_count, _dimensions, _stacks, _dropout, _feed_forward_size,
                 _share_parameter=True):
        super(Encoder, self).__init__()
        self.share_parameter = _share_parameter
        self.attention = nn.ModuleList([
            MultiHeadAttention(_multi_heads_count, _dimensions, _dropout)
            for _ in range(1 if _share_parameter else _stacks)
        ])
        self.position_feed_forward = nn.ModuleList([
            PositionwiseFeedForward(_dimensions, _feed_forward_size, _dropout)
            for _ in range(1 if _share_parameter else _stacks)
        ])
        self.position = PositionalEncoding(_dimensions, _dropout)
        self.layer_norm = nn.LayerNorm(_dimensions, epsilon=1e-6)
        self.stacks = _stacks
        self.dropout = Dropout(_dropout)
        self.with_encoder = _with_encoder

    def eval(self):
        self.attention.eval()
        self.position_feed_forward.eval()
        self.position.eval()
        self.layer_norm.eval()
        self.dropout.eval()

    def _generate_mask(self, _position_encode_tensor):
        target_length = _position_encode_tensor.size(1)
        return paddle.ones((target_length, target_length))

    def forward(self, _input_tensor):
        output = self.position(_input_tensor)
        if self.with_encoder:
            source_mask = self._generate_mask(output)
            for i in range(self.stacks):
                actual_i = 0 if self.share_parameter else i
                normed_output = self.layer_norm(output)
                output = output + self.dropout(
                    self.attention[actual_i](normed_output, normed_output, normed_output, source_mask)
                )
                normed_output = self.layer_norm(output)
                output = output + self.dropout(self.position_feed_forward[actual_i](normed_output))
            output = self.layer_norm(output)
        return output


class Decoder(nn.Layer):
    def __init__(self, _multi_heads_count, _dimensions, _stacks, _dropout, _feed_forward_size, _n_classes,
                 _padding_symbol=0, _share_parameter=True):
        super(Decoder, self).__init__()
        self.share_parameter = _share_parameter
        self.attention = nn.ModuleList([
            MultiHeadAttention(_multi_heads_count, _dimensions, _dropout)
            for _ in range(1 if _share_parameter else _stacks)
        ])
        self.source_attention = nn.ModuleList([
            MultiHeadAttention(_multi_heads_count, _dimensions, _dropout)
            for _ in range(1 if _share_parameter else _stacks)
        ])
        self.position_feed_forward = nn.ModuleList([
            PositionwiseFeedForward(_dimensions, _feed_forward_size, _dropout)
            for _ in range(1 if _share_parameter else _stacks)
        ])
        self.position = PositionalEncoding(_dimensions, _dropout)
        self.stacks = _stacks
        self.dropout = Dropout(_dropout)
        self.layer_norm = nn.LayerNorm(_dimensions, epsilon=1e-6)
        self.embedding = nn.Embedding(_n_classes, _dimensions)
        self.sqrt_model_size = math.sqrt(_dimensions)
        self.padding_symbol = _padding_symbol

    def _generate_target_mask(self, _source, _target):
        target_pad_mask = (_target != self.padding_symbol).unsqueeze(1).unsqueeze(3)  # (b, 1, len_src, 1)
        target_length = _target.size(1)
        target_sub_mask = paddle.tril(
            paddle.ones((target_length, target_length), dtype=paddle.uint8)
        )
        source_mask = paddle.ones((target_length, _source.size(1)), dtype=paddle.uint8)
        target_mask = target_pad_mask & target_sub_mask.bool()
        return source_mask, target_mask

    def eval(self):
        self.attention.eval()
        self.source_attention.eval()
        self.position_feed_forward.eval()
        self.position.eval()
        self.dropout.eval()
        self.layer_norm.eval()
        self.embedding.eval()

    def forward(self, _target_result, _memory):
        target = self.embedding(_target_result) * self.sqrt_model_size
        target = self.position(target)
        source_mask, target_mask = self._generate_target_mask(_memory, _target_result)
        output = target
        for i in range(self.stacks):
            actual_i = 0 if self.share_parameter else i
            normed_output = self.layer_norm(output)
            output = output + self.dropout(
                self.attention[actual_i](normed_output, normed_output, normed_output, target_mask)
            )
            normed_output = self.layer_norm(output)
            output = output + self.dropout(
                self.source_attention[actual_i](normed_output, _memory, _memory, source_mask))
            normed_output = self.layer_norm(output)
            output = output + self.dropout(self.position_feed_forward[actual_i](normed_output))
        return self.layer_norm(output)
