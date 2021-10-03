# -*- coding: utf-8 -*-

from typing import *
import json
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

import numpy as np

import paddle


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt', encoding='utf-8') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def check_parameters(module: paddle.nn.Layer):
    invalid_params = []
    for name, p in module.named_parameters():
        is_invalid = paddle.isnan(p).any() or paddle.isinf(p).any()
        if is_invalid:
            invalid_params.append(name)
    return invalid_params


def is_invalid(tensor: paddle.Tensor):
    invalid = paddle.isnan(tensor).any() or paddle.isinf(tensor).any()
    return invalid


def binary_accuracy(output: paddle.Tensor, target: paddle.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with paddle.no_grad():
        batch_size = target.shape[0]  # .size(0)
        pred = (output >= 0.5).t().flatten()  # .view(-1)
        correct = pred.eq(target.flatten()).sum()
        correct.set_value(correct.multiply(100. / batch_size))
        # correct.mul_(100. / batch_size)
        return correct
