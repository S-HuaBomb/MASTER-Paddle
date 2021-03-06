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

import os, sys, argparse, logging
# project_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),"..","..")
# sys.path.append(project_dir)
import distance
import json

def process_args(args):
    parser = argparse.ArgumentParser(description='Evaluate accuracy and text edit distance.')

    parser.add_argument('--predict-path', dest='predict_path',
                        type=str, required=True,
                        help=(
                            'Result json file containing '
                            '[{filename:<filename>, result:<predict_result>}, ...]'
                            'This should be set to the output file of the model.'
                        ))

    parser.add_argument('--label-path', dest='label_path',
                        type=str, required=False,
                        help=(
                            'Result json file containing '
                            '[{ImageFile:<filename>, Label:<label_gold>}, ...]. '
                        ))

    parser.add_argument('--log-path', dest="log_path",
                        type=str, default='log.txt',
                        help=('Log file path, default=log.txt'
                              ))
    parameters = parser.parse_args(args)
    return parameters


def main(args):
    parameters = process_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename=parameters.log_path)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info('Script being executed: %s' % __file__)

    for pred_json in os.listdir(parameters.predict_path):
        if not pred_json.endswith('json'):
            continue

        json_name = pred_json.split('.')[0]
        label_name = json_name.replace('pred', 'label')
        pred_json_path = os.path.join(parameters.predict_path, pred_json)
        label_path = os.path.join(parameters.label_path, f"{label_name}.txt")
        if not os.path.isfile(label_path):
            raise ValueError(f"{label_name}.txt not found in {parameters.label_path}")

        print(f"calculating metrics of {json_name}")

        total_ref = 0
        total_edit_distance = 0
        total_correct_num = 0
        total_correct_num_case_ins = 0
        predicted_res = json.load(open(pred_json_path, encoding='utf-8'))

        label_gold_dict = {}
        with open(label_path, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().rstrip("\n").split(',')
                # print('line:', line)
                # lineobj = json.loads(line)
                # text_label = lineobj['Label']
                # ImageFile = lineobj['ImageFile']
                label_gold_dict[line[0]] = line[1]

        hold_history = []

        for idx, predict_item in enumerate(predicted_res):
            if idx % 1000 == 0:
                print('current sample idx: ' + str(idx))
            filename = predict_item['filename']
            label_pred = predict_item['result']
            if not filename in label_gold_dict.keys(): continue
            label_gold = label_gold_dict[filename]
            # label_pred = label_pred.strip()
            # label_gold = label_gold.strip()

            # calculate edit distance
            ref = len(label_gold)
            edit_distance = distance.levenshtein(label_gold, label_pred)
            total_ref += ref
            total_edit_distance += edit_distance

            # calculate accuracy
            # print(f"filename: {filename} - label_gold: {label_gold} - label_pred: {label_pred}")
            if edit_distance==0:
                total_correct_num += 1
            if label_gold.lower() == label_pred.lower():
                total_correct_num_case_ins += 1

            hold_history.append(filename)

        # number of pred not equal to label
        if len(hold_history) != len(label_gold_dict.keys()):
            for file in label_gold_dict.keys():
                if file in hold_history: continue
                label_gold = label_gold_dict[file]
                # label_gold = label_gold.strip()
                total_edit_distance +=len(label_gold)
                total_ref += len(label_gold)
        # print(total_correct_num)
        logging.info('Sequence Accuracy: %f Case_ins: %f' % (float(total_correct_num) / len(label_gold_dict.keys()),
                                                            float(total_correct_num_case_ins) / len(label_gold_dict.keys())))
        logging.info('Edit Distance Accuracy: %f' % (1. - float(total_edit_distance) / total_ref))
        print("=============================================================================\n")

'''
usage:
python calculate_metrics.py --predict-path predict_result.json --label-path label.txt

predict_result.json: output of model
label.txt file format: multi-line, every line containing {ImageFile:<ImageFile>, Label:<Label>}

requirements: 
pip install distance
'''
if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Evaluation finished')