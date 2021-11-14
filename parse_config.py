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
import collections
from argparse import ArgumentParser
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime

import paddle.distributed as dist

from logger import setup_logging
from utils import read_json, write_json


class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume
        # str to bool, from modification or from default json file
        self.update_config('distributed', (self.config['distributed'] == 'true') or self.config['distributed'] is True)
        self.update_config('finetune', (self.config['finetune'] == 'true') or self.config['finetune'] is True)

        if (self.config['local_rank'] == 0 and self.config['distributed']) \
                or (not self.config['distributed']):  # only local master process create saved output dir
            # set save_dir where trained model and log will be saved.
            save_dir = Path(self.config['trainer']['save_dir'])
            log_dir = Path(self.config['trainer']['log_dir'])

            exper_name = self.config['name']
            if run_id is None:  # use timestamp as default run-id
                run_id = datetime.now().strftime(r'%m%d_%H%M%S')
            else:
                run_id = run_id + '_' + datetime.now().strftime(r'%m%d_%H%M%S')
            self._save_dir = save_dir / 'models' / exper_name / run_id
            if self.config['distributed']:
                self._log_dir = log_dir
            else:
                self._log_dir = save_dir / 'log' / exper_name / run_id

            # make directory for saving checkpoints and log.
            exist_ok = run_id == ''
            self.save_dir.mkdir(parents=False, exist_ok=True)
            self.log_dir.mkdir(parents=False, exist_ok=True)

            # save updated config file to the checkpoint dir, only local master save file
            write_json(self.config, self.save_dir / 'config.json')

            # configure logging module, only local master setup logging
            setup_logging(self.log_dir)
            self.log_levels = {
                0: logging.WARNING,
                1: logging.INFO,
                2: logging.DEBUG
            }

    @classmethod
    def from_args(cls, args: ArgumentParser, options: collections.namedtuple = ''):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        for opt in options:
            args.add_argument(*opt.flags, default=opt.default, type=opt.type, help=opt.help)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume is not None:
            resume = Path(args.resume)
            if args.config is None:
                config_file_path = resume.parent / 'config.json'
            else:
                config_file_path = args.config
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            config_file_path = Path(args.config)

        config = read_json(config_file_path)
        if args.config and resume and args.finetune == 'false':
            # update new config for resume (continue train), finetune mode will don not use previous config
            config.update(read_json(args.config))
        try:
            if args.distributed is not None:
                config['distributed'] = (args.distributed == 'true')
                if not config['distributed']:  # change to one gpu or cpu mode if not distributed setting.
                    config['local_world_size'] = 1
            if args.finetune is not None:
                config['finetune'] = (args.finetune == 'true')
        except Exception:
            pass
        # parse custom cli options into dictionary
        modification = {opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, resume, modification, config['run_id'])

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        # assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def update_config(self, key, value):
        """Set config value ike ordinary dict. """
        self.config[key] = value

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity,
                                                                                       self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

    # @property
    # def log_levels(self):
    #     return self._log_levels


# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
