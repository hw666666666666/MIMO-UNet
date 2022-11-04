# Copyright 2020-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""evaluate BiT model on CIFAR-10"""

import os

import mindspore as ms
import mindspore.nn as nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.dataset import create_dataset
from src.utils import do_keep_cell_fp16, context_device_init, count_params
from src.metrics import DistPSNR, PSNRCell
from src.model_utils.moxing_adapter import moxing_wrapper, modelarts_process
from src.model_utils.device_adapter import get_device_id
from src.model_utils.config import config
from src.MIMOUNet import build_net
from src.loss import ContentLoss


def process_checkpoint(net, ckpt):
    prefix = "ema."
    len_prefix = len(prefix)
    if config.enable_ema:
        ema_ckpt = {}
        for name, param in ckpt.items():
            if name.startswith(prefix):
                ema_ckpt[name[len_prefix:]] = ms.Parameter(default_input=param.data, name=param.name[len_prefix:])
        ckpt = ema_ckpt

    net_param_dict = net.parameters_dict()
    ckpt = {k:v for k, v in ckpt.items() if k in net_param_dict}

    return ckpt


@moxing_wrapper(pre_process=modelarts_process)
def eval():
    config.batch_size = 1
    config.pretrain_ckpt = config.load_path
    config.eval_dataset_path = os.path.join(config.dataset_path, 'test')
    
    if not config.device_id:
        config.device_id = get_device_id()
    context_device_init(config)
    print('\nconfig: {} \n'.format(config))

    # define network
    model_name = config.model_name
    net = build_net(model_name)

    ckpt = load_checkpoint(config.pretrain_ckpt)

    ckpt = process_checkpoint(net, ckpt)

    load_param_into_net(net, ckpt)

    # mixed precision training
    net.to_float(ms.dtype.float32)
    # do_keep_cell_fp16(net, cell_types=(nn.Conv2d, nn.Conv2dTranspose))
    
    eval_network = PSNRCell(net, config.run_distribute)

    net.set_train(False)

    metrics = {'psnr': DistPSNR(batch_size=config.batch_size, device_num=1)}
    model = ms.Model(net, eval_network=eval_network, metrics=metrics)
    
    dataset = create_dataset(dataset_path=config.eval_dataset_path, do_train=False, config=config,
                             drop_remainder=False)
    step_size = dataset.get_dataset_size()
    if step_size == 0:
        raise ValueError("The step_size of dataset is zero. Check if the images count of eval dataset is more \
            than batch_size in config.py")
    print("step_size = ", step_size)
    
    res = model.eval(dataset, dataset_sink_mode=False)
    print("result:{}\npretrain_ckpt={}".format(res, config.pretrain_ckpt))


if __name__ == '__main__':
    eval()
