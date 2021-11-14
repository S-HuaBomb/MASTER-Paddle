#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import Counter

import paddle
from paddle.io import Sampler

from data_utils.datasets import TextDataset


class ImbalancedDatasetSampler(Sampler):
    def __init__(self, _data_source):
        # if indices is not provided, all elements in the dataset will be considered
        super().__init__(_data_source)
        self.num_samples = len(_data_source)

        all_labels = self._get_labels(_data_source)
        character_counter = Counter()
        total_characters = 0
        for m_label in all_labels:
            character_counter.update(m_label)
            total_characters += len(m_label)
        print('dataset character statistics')
        for m_label, m_count in sorted(character_counter.items(), key=lambda x: x[1]):
            print(f'char:{m_label},count:{m_count},ratio:{round(m_count * 100 / total_characters, 2)}%')
        weights = []
        for m_label in all_labels:
            weights.append(
                total_characters / (sum([character_counter.get(m_char) for m_char in m_label]) / len(m_label)))
        self.weights = paddle.to_tensor(weights).astype(paddle.float64)

    def _get_labels(self, _dataset):
        if isinstance(_dataset, TextDataset):
            return _dataset.get_all_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (i for i in paddle.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

# TODO: distributed imbalance sampler
