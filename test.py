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

import os
import argparse
import paddle
from tqdm import tqdm
from pathlib import Path
import json
import sys

from paddle.io import DataLoader

import model.master as master_arch_module
from data_utils.datasets import LmdbTestDataset, TextDataset, ResizeWeight, DistCollateFn, CustomImagePreprocess
from utils.label_util import LabelTransformer
from utils import decode_util

# set device
paddle.set_device('gpu' if paddle.is_compiled_with_cuda() else 'cpu')


def predict(args):
    device = paddle.get_device()
    checkpoint = paddle.load(args.checkpoint)

    config = checkpoint['config']
    state_dict = checkpoint['model_state_dict']
    monitor_best = checkpoint['monitor_best']
    print('Loading checkpoint: {} \nwith saved best metric {:.4f}'.format(args.checkpoint, monitor_best))

    # prepare model for testing
    model = config.init_obj('model_arch', master_arch_module)
    model.set_state_dict(state_dict)
    model.eval()

    # setup dataset and data_loader instances
    # img_w = config['train_dataset']['args']['img_w']
    # img_h = config['train_dataset']['args']['img_h'] txt_file=None, img_root=None
    index_txt_file = args.index_txt_file  # if None, read img from img_root, otherwise, txt_file must be set.
    in_channels = config['model_arch']['args']['backbone_kwargs']['in_channels']
    convert_to_gray = False if in_channels == 3 else True

    for subset in os.listdir(args.evaluation):
        subset_path = os.path.join(args.evaluation, subset)
        if os.path.isdir(subset_path):
            img_folder = subset_path
        else:
            continue
        # setup output path
        output_path = Path(args.output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        if args.is_lmdb_folder >= 1:
            print('is_lmdb_folder:', args.is_lmdb_folder)
            test_dataset = LmdbTestDataset(lmdb_dir_root=img_folder,
                                           label_output=output_path,
                                           transform=ResizeWeight((args.width, args.height),
                                                                  gray_format=convert_to_gray),
                                           img_w=args.width,
                                           img_h=args.height)
        else:
            test_dataset = TextDataset(img_root=img_folder, txt_file=index_txt_file,
                                       transform=ResizeWeight((args.width, args.height), gray_format=convert_to_gray),
                                       img_w=args.width,
                                       img_h=args.height,
                                       training=False,
                                       testing_with_label_file=index_txt_file is not None,
                                       convert_to_gray=convert_to_gray)

        test_data_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                      collate_fn=DistCollateFn(training=False),
                                      num_workers=args.num_workers, drop_last=False)

        print(f'subset {subset} size: {len(test_dataset)} steps: {len(test_data_loader)}')

        result_output_file = output_path.joinpath(f"{subset}_pred.json")
        pred_results = []
        # predict and save to file
        for step_idx, input_data_item in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
            batch_size = input_data_item['batch_size']
            if batch_size == 0:
                continue

            images = input_data_item['images']
            file_names = input_data_item['file_names']
            with paddle.no_grad():
                if hasattr(model, '_layers'):
                    model = model._layers
                # (bs, max_len)
                # TODO replace with memory-cache based decode
                outputs, probs = decode_util.greedy_decode_with_probability(model, images, LabelTransformer.max_length,
                                                                            LabelTransformer.SOS,
                                                                            LabelTransformer.EOS,
                                                                            _padding_symbol_index=LabelTransformer.PAD,
                                                                            _result_device=paddle.get_device(),
                                                                            _is_padding=True)

            for index, (pred, prob, img_name) in enumerate(zip(outputs[:, 1:], probs, file_names)):
                predict_text = ""
                # pred_list = []
                pred_score_list = []
                for i in range(len(pred)):  # decode one sample
                    if pred[i] == LabelTransformer.EOS:
                        pred_score_list.append(prob[i])
                        break
                    if pred[i] == LabelTransformer.UNK:
                        continue
                    decoded_char = LabelTransformer.decode(pred[i])
                    predict_text += decoded_char
                    # pred_list.append(decoded_char)
                    pred_score_list.append(prob[i])
                pred_score = sum(pred_score_list) / len(pred_score_list)
                pred_item = {"filename": Path(f"{img_name}").name,
                             "result": predict_text,
                             "pred_score": pred_score.cpu().item()}
                pred_results.append(pred_item)

        with result_output_file.open(mode='w') as f:
            f.write(json.dumps(pred_results))
        print(f'Predict results has written to {result_output_file.as_posix()}\n')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='MASTER Pytorch Test')
    args.add_argument('-ckpt', '--checkpoint', default=None, type=str, required=True,
                      help='path to load checkpoint (default: None)')
    args.add_argument('-img', '--img_folder', default=None, type=str, required=False,
                      help='image folder (default: None)')
    args.add_argument('-index_txt_file', '--index_txt_file', default=None, type=str, required=False,
                      help='path to index txt and corresponding filename, '
                           'if None, read img from img_folder, otherwise, index_txt_file must be set (default: None)')
    args.add_argument('-width', '--width', default=160, type=int, required=False,
                      help='resized image width (default: 256)')
    args.add_argument('-height', '--height', default=48, type=int, required=False,
                      help='resized image height (default: 32)')
    args.add_argument('-output', '--output_folder', default='predict_results', type=str, required=False,
                      help='output folder (default: predict_results)')
    args.add_argument('-output_file_name', '--output_file_name', default='predict_result.json', type=str,
                      required=False,
                      help='output file name (default: predict_result.json)')
    args.add_argument('-g', '--gpu', default=1, type=int, required=False,
                      help='GPU id to use. (default: -1, cpu)')
    args.add_argument('--bs', '--batch_size', default=128, type=int, required=False,
                      help='batch size (default: 1)')
    args.add_argument('--num_workers', default=4, type=int, required=False,
                      help='batch size (default: 1)')
    args.add_argument('--evaluation', default='work/data_lmdb_release/evaluation',
                      type=str, required=True,
                      help='evaluation path')
    args.add_argument('--is_lmdb_folder', default=0,
                      type=int, required=True,
                      help='weather the evaluation data is lmdb data?')

    args = args.parse_args()
    predict(args)
