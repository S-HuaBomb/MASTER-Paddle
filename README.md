# 论文复现：MASTER: Multi-Aspect Non-local Network for Scene Text Recognition

[English](README_EN.md) | [简体中文](README.md)

- **MASTER-Paddle**
  - [一、简介](#一简介)
  - [二、复现精度](#二复现精度)
  - [三、数据集](#三数据集)
  - [四、环境依赖](四环境依赖)
  - [五、快速开始](五快速开始)
    - [5.1 训练](#51-训练)
    - [5.2 预测](#52-预测)
  - [六、代码结构与详细说明](#六代码结构与详细说明)
    - [6.1 代码结构](#61-代码结构)
    - [6.2 参数说明](#62-参数说明)
    - [6.3 训练流程](#63-训练流程)
    - [6.4 测试流程](#64-测试流程)
  - [七、实验数据及复现心得](#七实验数据比较及复现心得)
  - [八、模型信息](#八模型信息)


## 一、简介

基于 RNN 的编码器-解码器结构在场景文本识别方面已经取得了巨大成功，然而，基于 RNN 的局部注意机制由于编码特征之间的高度相似性会导致注意力混淆，从而导致注意力漂移问题（attention-drift problem）。

此外，基于 RNN 的方法由于并行性差，所以训练效率较低。受 GCNet 中采用的全局上下文的有效性以及 Transformer 在 NLP 和 CV 中取得的巨大成功的启发，我们提出了 **MASTER**（**M**ulti**A**spect non-local network for irregular **S**cene **TE**xt **R**ecognition），以实现对不规则场景的高效准确的文本识别。
![0](https://img-blog.csdnimg.cn/e6a414851d61442589801de4824408f9.png)

- 本文提出了一种新的多头注意力模块，并将其融合到传统的 CNN backbone（ResNet-31）中，从而使特征提取器能够对全局上下文建模。
- 在推理阶段，本文引入了基于内存缓存的解码策略，以加快解码过程。主要方法是删除不必要的计算并缓存以前解码的一些中间结果。


论文链接：[MASTER: Multi-Aspect Non-local Network for Scene Text Recognition](https://arxiv.org/abs/1910.02562)


## 二、复现精度
参考官方开源的 pytorch 版本代码 [https://github.com/wenwenyu/MASTER-pytorch](https://github.com/wenwenyu/MASTER-pytorch)，基于 paddlepaddle 深度学习框架，对文献算法进行复现后，本项目达到的测试精度，如下表所示：

| 数据集 | IIIT5K | SVT |  IC03 | IC13 | IC15 | SVTP | CUTE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| paddle 版本精度 |   |  |  |  |  |  |  |
| 参考文献精度 | 95.0 | 90.6 | 96.4 | 95.3 | 79.4 | 84.5 | 87.5 |

超参数配置如下：
> 详见 `MASTER-Paddle/configs/config_lmdb_dist.json`

|超参数名|设置值| 说明 |
| --- | --- | --- |
| distributed | true | 强烈推荐使用多卡训练 |
| local_world_size | 4 | 4 卡刚刚好
| local_rank | 0 | 主卡 |
| n_class | 62 | 文本类数：10 个阿拉伯数字 + 52 个区分大小写的英文字母 + 4 个文档描述符（这 4 个类竟然是代码里硬生生加上 4），还好我聪明及时发现 :blush: |
| with_encoder | false |  with_encoder? false |
|model_size | 512 | 编码器的输出尺寸 |
| ratio | 0.0625 | 转圈圈 |
| dropout | 0.2 | dropout  |
| feed_forward_size | 2048 | dff 前馈模块设置为2048| 
| multiheads | 8 | 多头注意力数是 8 |
| batch_size | 128 | bs = 128 单卡占用不到 16 GB 显存|
| num_workers | 2 | dataloader workers |
| epochs | 16 | 论文说 12 个 epoch 就好了 |
| lr | 0.0004 |这是 4 卡并行情况下。单卡情况下是 0.0001，全程保持不变 |
| lr_scheduler | LinearWarmup | 最好利用 lr warm-up 来训练，我猜的 |
| img_w | 160 | 图片裁剪宽度|
| img_h | 48 | 图片裁剪宽度 |

## 三、数据集

数据集链接：百度网盘 [data_lmdb_release.zip](https://pan.baidu.com/s/1KSNLv4EY3zFWHpBYlpFCBQ)，提取码：`rryk`，下载整个 data_lmdb_release 文件夹。

本项目使用的是原文作者制作的文本识别数据集合集：data_lmdb_release.zip，其中包含以下内容：

- training datasets：[MJSynth (MJ)](http://www.robots.ox.ac.uk/~vgg/data/text/) 和 [SynthText (ST)](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)，总数 12747394 个。
- validation datasets：训练集 [IC13](http://rrc.cvc.uab.es/?ch=2)、[IC15](http://rrc.cvc.uab.es/?ch=4)、[IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html) 和 [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset) 的合集。
- evaluation datasets：基准评估数据集，包括 [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)、[SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)、[IC03](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions)、[IC13](http://rrc.cvc.uab.es/?ch=2)、[IC15](http://rrc.cvc.uab.es/?ch=4)、[SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf) 和 [CUTE](http://cs-chan.com/downloads_CUTE80_dataset.html)。

## 四、环境依赖
- 硬件：
  - x86 cpu（RAM >= 16 GB）
  - **4 × Tesla V100 GPU（VRAM per node >= 16 GB）**
    
    > 宽油劝退：单卡训练会好久 :dog:，最多是用于 debug
 
  - CUDA + cuDNN
- 框架：
  - paddlepaddle-gpu==2.1.2
- 其它依赖项：
  - pandas==1.0.5
  - numpy==1.19.2
  - tqdm==4.47
  - Distance==0.1.3
  - Pillow==7.2.0

## 五、快速开始

### 5.1 训练

解压 `data_lmdb_release/data_lmdb_reupload.zip`，然后会得到如下的数据集目录：
```
├── data_lmdb_release                 # 解压出来的根目录
	├── evaluation                    # 测试集
	    ├── CUTE80              
	    ├── IC03_860             
	    ├── IC03_867             
	    ├── IC13_857          
	    ├── IC13_1015            
	    ├── IC15_1811         
	    ├── IC15_2077          
	    ├── IIIIT5k_3000            
	    ├── SVT             
	    ├── SVTP         
	├── training                      # 训练集
	    ├── MJ           
	    	├── MJ_test            
	    	├── MJ_train              
	    	├── MJ_valid            
	    ├── ST                        # 训练集
	    	├── data.mdb             
	    	├── lock.mdb            
	├── validation                    # 验证集
	    ├── data.mdb              
	    ├── lock.mdb                
```

在 `configs/config_lmdb_dist.json` 中修改训练配置，以下是我的训练配置（4 × Tesla V100）：
```json
{
    "name": "MASTER_Default",
    "run_id":"example",

    "finetune":false,

    "distributed":true,
    "local_world_size":4,
    "local_rank":0,
    "global_rank":-1,

    "deterministic":false,
    "seed":123,

    "model_arch": {
        "type": "MASTER",
        "args": {
            "common_kwargs":{
                "n_class":62,
                "with_encoder":false,
                "model_size": 512,
                "multiheads": 8
            },
            "backbone_kwargs":{
                "in_channels": 3,
                "gcb_kwargs":{
                    "ratio": 0.0625,
                    "headers": 1,
                    "att_scale": true,
                    "fusion_type": "channel_concat",
                    "layers":[false, true, true, true]
                }
            },
            "encoder_kwargs":{
                "stacks": 3,
                "dropout": 0.2,
                "feed_forward_size": 2048
            },
            "decoder_kwargs":{
                "stacks": 3,
                "dropout": 0.2,
                "feed_forward_size": 2048
            }
        }
    },

    "train_dataset": {
        "type": "hierarchy_dataset",
        "args": {
            "lmdb_dir_root":"/root/paddlejob/workspace/train_data/datasets/data_lmdb_release/training",
            "select_data": "MJ/MJ_train-ST",
            "img_w":160,
            "img_h":48,
            "training":true
        }
    },
    "train_loader": {
        "type": "DataLoader",
        "args":{
            "batch_size": 32,
            "shuffle": true,
            "drop_last": true,
            "num_workers": 8,
            "pin_memory":false
        }
    },

    "val_dataset": {
        "type": "hierarchy_dataset",
        "args": {
            "lmdb_dir_root":"/root/paddlejob/workspace/train_data/datasets/data_lmdb_release",
            "select_data": "validation",
            "img_w":160,
            "img_h":48,
            "training":true
        }
    },
    "val_loader": {
          "type": "DataLoader",
          "args":{
              "batch_size": 32,
              "shuffle": false,
              "drop_last": false,
              "num_workers": -1,
              "pin_memory":false
          }
      },

    "optimizer": {
          "type": "Adam",
          "args":{
              "lr": 0.0004
          }
    },
    "lr_scheduler": {
        "type": "LinearWarmup",
        "args": {
            "step_size": 3,
            "gamma": 0.5
        }
    },

    "trainer": {
        "epochs": 16,
        "max_len_step":null,

        "do_validation": true,
        "validation_start_epoch": 4,
        "log_step_interval": 1,
        "val_step_interval": 12000,

        "train_batch_size": 128,
        "val_batch_size":128,
        "train_num_workers":2,
        "val_num_workers":2,

        "save_dir": "/root/paddlejob/workspace/output/",
        "log_dir": "/root/paddlejob/workspace/log/",
        "save_period": 4,
        "log_verbosity": 2,

        "monitor": "max word_acc",
        "monitor_open": true,
        "early_stop": -1,

        "anomaly_detection": false,
        "tensorboard": false,

        "sync_batch_norm":true
    }
}
```
> 源码的默认配置是跑不起来的，比如 `local_rank=-1` 在多卡训练的时候代码会对这个变量做判断，然后跳过创建输出文件夹的一段代码，导致模型没有保存任何东西。一开始 `n_class=-1` 也是不对的，这个变量要指定训练集中出现的所有字符的类数，论文指出有 66 类。`lr=0.0004` 是 4 卡训练的配置，单卡是 0.0001。`lr_scheduler` 是我自己加的学习率预热 [`paddle.optimizer.lr.LinearWarmup`](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/ReduceOnPlateau_cn.html)。`epochs=16`，论文指出 4 卡训练 12 个 epochs 即可完成。**若要单卡训练（debug），只需设置 `"distributed":false`**。
    
#### step2: 运行训练

准备好 4 张 V100，运行以下命令从零开始训练 MASTER：
```
python -m paddle.distributed.launch train.py -c configs/config_lmdb_dist.json
```
如果你使用 AI Studio 脚本训练，需要在 shell 脚本中在运行训练之前把数据集解压：
```shell
#!/bin/bash
pip install lmdb distance

SRC_DIR=/root/paddlejob/workspace/train_data/datasets/data111037/data_lmdb_release.zip
TO_DIR=/root/paddlejob/workspace/train_data/datasets/

unzip -d $TO_DIR $SRC_DIR

python -m paddle.distributed.launch --gpus '0,1,2,3' MASTER/train.py -c MASTER/configs/config_lmdb_dist.json
```

如果训练中断，并且保存了模型，重启训练时可以加上 `--resume path/to/checkpoint`，训练会从指定的模型重启训练。

### 5.2 预测

#### Testing

*现在只训练完成了 6 个 epoch*

**训练好的模型可到百度网盘自取：[ckpts/checkpoint-epochX.pdparams](https://pan.baidu.com/s/1nUwv6Q49nM2DT7PxWNsZWw)**，提取码：5qyu。

指定模型的路径和需要用于识别文本的图片文件夹：
```shell
python test.py --checkpoint path/to/checkpoint --img_folder path/to/img_folder \
               --output_folder path/to/output_folder
```
代码运行完成的结果会输出到 `predict_result.json`，保存在 output_folder 指定的文件夹下，其结果如下所示：
```json
[{"filename": "001.jpg", "result": "BEACH", "pred_score": 0.9915353655815125}, {"filename": "002.jpg", "result": "RONALDON", "pred_score": 0.8017494082450867}]
```


#### Evaluation
*训练没有完成，还未测试过此处的代码*

我们需要运行上面的 Testing 得到预测结果之后才能运行 Evaluation 来计算准确率。

运行：
```
python utils/calculate_metrics.py --predict-path predict_result.json --label-path label.txt
```

## 六、代码结构与详细说明
### 6.1 代码结构
```
├── assets                    # 图片
├── configs                   # 配置文件
├── data                      # 一些 examples
├── data_utils                # 加载数据集的工具代码
├── logger                    # 日志 logger 工具代码
├── logs                      # 日志
├── models                     # 网络结构定义
│   ├── backbone.py              # 骨干 CNN 网络 ResNet-31
│   ├── context_block.py          # context_block
│   ├── initializers.py       # 抄 pytorch 的参数初始化函数
│   ├── master.py               # MASTER 主网络
│   ├── transformer.py              # transformer 模块
├── tests           # 测试代码
├── trainer            # 训练代码
├── utils            # 工具代码
├── LICENSE                    # 开源协议
├── README.md                  # 主页 readme
├── requirements.txt           # 项目的其它依赖
├── test.py                    # 启动预测
├── train.py                     # 启动训练
```

### 6.2 参数说明
见 [二、复现精度](#二复现精度)

### 6.3 训练流程
见 [五、快速开始](#五快速开始)

执行训练开始后，将得到类似如下的输出：
```
[2021-10-12 18:46:00,739 - train - INFO] - Distributed GPU training model start...
[2021-10-12 18:46:00,739 - train - WARNING] - You have chosen to benchmark training. This will turn on the CUDNN benchmark settingwhich can speed up your training considerably! You may see unexpected behavior when restarting from checkpoints due to RandomizedMultiLinearMap need deterministic turn on.
[2021-10-12 18:46:00,739 - train - INFO] - [Process 2578] world_size = 4, rank = 0
[2021-10-12 18:46:27,278 - train - INFO] - Dataloader instances have finished. Train datasets: 12747394 Val datasets: 6992 Train_batch_size/gpu: 128 Val_batch_size/gpu: 128.
[2021-10-12 18:46:28,274 - train - INFO] - Model created, trainable parameters: 54653557.
[2021-10-12 18:46:28,275 - train - INFO] - Optimizer and lr_scheduler created.
[2021-10-12 18:46:28,275 - train - INFO] - Max_epochs: 16 Log_step_interval: 1 Validation_step_interval: 12000.
[2021-10-12 18:46:28,275 - train - INFO] - Training start...
[2021-10-12 18:46:28,275 - trainer - INFO] - [Process 2578] world_size = 4, rank = 0, n_gpu/process = 1, device_ids = [0]
[2021-10-12 18:46:40,308 - trainer - INFO] - Train Epoch:[1/16] Step:[1/24898] Loss: 5.889343 Loss_avg: 5.889343 LR: 0.00040000
[2021-10-12 18:46:41,351 - trainer - INFO] - Train Epoch:[1/16] Step:[2/24898] Loss: 6.429784 Loss_avg: 6.659563 LR: 0.00040000
[2021-10-12 18:46:42,409 - trainer - INFO] - Train Epoch:[1/16] Step:[3/24898] Loss: 5.376784 Loss_avg: 5.565304 LR: 0.00040000
[2021-10-12 18:46:43,452 - trainer - INFO] - Train Epoch:[1/16] Step:[4/24898] Loss: 5.122001 Loss_avg: 5.404478 LR: 0.00040000
[2021-10-12 18:46:44,505 - trainer - INFO] - Train Epoch:[1/16] Step:[5/24898] Loss: 4.696676 Loss_avg: 4.262918 LR: 0.00040000
```

### 6.4 测试流程
见 [五、快速开始](#五快速开始)

## 七、实验数据比较及复现心得
![1](https://img-blog.csdnimg.cn/518674491fec48cab57ce2c0db79f446.png)

#### 训练耗时

原文的实验细节说明作者使用 4 卡（Tesla V100）并行，优化器选择 Adam 从零开始训练，总共训练 12 个 epochs，每个 epoch 大概费时 3 个小时。**batch_size 为 128 × 4，learning_rate 全程保持为 0.0004**，我使用了同样的设置，在百度 AI Studio 平台提交 4 个 V100 GPU 并行训练脚本。但是每个 epoch 大概耗时 6 个小时，训练到了中期启动 valid_epoch（训练过程验证） 之后耗时更长。


#### 多卡训练的 batch_size & learning_rate

单卡时 learning_rate 设为 0.0001，原文提到 learning_rate 的大小应该与 GPU 数量相关联，这也是以往做过的研究。多卡时得益于 [`paddle.DataParallel`](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/DataParallel_cn.html#dataparallel) 的多进程并行，在保持单卡 bs = 128 的情况下，4 个 Trainer 进程的 `paddle.io.DataLoader` 分别加载 bs = 128 的 samples，等价于 bs = 128 × 4；所以，根据 [linear scale rule](https://www.cnblogs.com/leebxo/p/10976653.html)，lr 需要等价倍增。**但是直接从头开始使用 4 倍 lr 似乎不是最佳选择，或许需要使用 [lr warm-up](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/LinearWarmup_cn.html)，在开头的几个 epochs 把 lr 从 0.0001 预热到 0.0004，后程再保持 0.0004 或许是更好的选择。**
>  [`torch.nn.DataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html?highlight=dataparallel#torch.nn.DataParallel) 是一个进程多个线程并行，受制于 Python 的 GIL 锁，所以 PyTorch 官方已经不推荐使用，而是主推更好更快的多进程并行：[`torch.nn.parallel.DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel)。所以 paddle.DataParallel 应该是对标的 PyTorch 的 DDP :smile:

#### 要不要理 TorchScript？
我把 `model/master.py` 中所有的 `torch.script.jit.` 的代码全部去掉了，网络模型全部换成继承 `nn.Layer`。因为 paddle 似乎没有 TorchScript 的替代品，所以咱们不需要？ :smile: 。

## 八、模型信息
| 信息 | 说明 |
| --- | --- |
| 发布者 | 石华榜 |
| 时间 | 2021.10 |
| 框架版本 | paddlepaddle==2.1.2 |
| 应用场景 | 多场景文本识别 |
| 支持硬件 | GPU × 4 |
| data_lmdb_release 数据集下载 | 百度网盘 [data_lmdb_release.zip](https://pan.baidu.com/s/1KSNLv4EY3zFWHpBYlpFCBQ)，提取码：`rryk` |
| AI Studio 地址 | [https://aistudio.baidu.com/aistudio/projectdetail/2351963](https://aistudio.baidu.com/aistudio/projectdetail/2351963) |

## Citations

```
@article{Lu2021MASTER,
  title={{MASTER}: Multi-Aspect Non-local Network for Scene Text Recognition},
  author={Ning Lu and Wenwen Yu and Xianbiao Qi and Yihao Chen and Ping Gong and Rong Xiao and Xiang Bai},
  journal={Pattern Recognition},
  year={2021}
}
```

## License
```
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
```
