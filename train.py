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
"""Train MIMO-UNet on GOPRO_Large dataset"""


import os
import time
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

from src.dataset import create_dataset
from src.utils import get_lr, do_keep_cell_fp16, build_params_groups, context_device_init, count_params
# from src.metrics import DistPSNR, PSNRCell
from src.callbacks import Monitor
from src.cell_wrapper import CustomTrainOneStepWithLossScaleCell
from src.model_utils.moxing_adapter import moxing_wrapper, modelarts_process
from src.model_utils.device_adapter import get_device_id
from src.model_utils.config import config
from src.MIMOUNet import build_net
from src.loss import ContentLoss


@moxing_wrapper(pre_process=modelarts_process)
def train():
    config.train_dataset_path = os.path.join(config.dataset_path, 'train')
    config.eval_dataset_path = os.path.join(config.dataset_path, 'test')

    if not config.device_id:
        config.device_id = get_device_id()
    start = time.time()
    # set context and device init
    context_device_init(config)
    print('\nconfig: {} \n'.format(config))

    dataset = create_dataset(dataset_path=config.train_dataset_path, do_train=True, config=config,
                             drop_remainder=True)
    step_size = dataset.get_dataset_size()

    if step_size == 0:
        raise ValueError("The step_size of dataset is zero. Check if the images' count of train dataset is more \
            than batch_size in config.py")

    # # get learning rate
    # lr = ms.Tensor(get_lr(global_step=0,
    #                       lr_max=config.lr_max,
    #                       lr_min=config.lr_min,
    #                       lr_warmup_epochs=config.lr_warmup_epochs,
    #                       total_epochs=config.num_epochs,
    #                       steps_per_epoch=step_size))

    # get learning rate
    lr = ms.Tensor(get_lr(global_step=0,
                          lr_init=config.lr_init,
                          lr_decay=config.lr_decay,
                          lr_warmup_epochs=config.lr_warmup_epochs,
                          lr_num_epochs_per_decay=config.lr_num_epochs_per_decay,
                          total_epochs=config.num_epochs,
                          steps_per_epoch=step_size))

    # define network
    model_name = config.model_name
    net = build_net(model_name)

    if config.rank_id == 0:
        print("Total number of parameters: {}".format(count_params(net)))

    # mixed precision training
    net.to_float(ms.dtype.float32)
    # do_keep_cell_fp16(net, cell_types=(nn.Conv2d, nn.Conv2dTranspose))

    metrics = None
    dist_eval_network = None
    eval_dataset = None
    # if config.run_eval:
    #     metrics = {'psnr': DistPSNR(batch_size=config.run_eval_batch_size, device_num=config.rank_size)}
    #     dist_eval_network = PSNRCell(net, config.run_distribute)
    #     eval_dataset = create_dataset(dataset_path=config.eval_dataset_path, do_train=False, config=config,
    #                                   drop_remainder=True)

    # define loss
    # label smoothing and mixup are done with dataset pipeline
    loss = ContentLoss()
    # mixed precision training loss
    net_with_loss = ms.amp._add_loss_network(net, loss, ms.dtype.float16)

    group_params = build_params_groups(net, config.weight_decay)
    # opt = nn.Lamb(params=group_params, learning_rate=lr, eps=config.epsilon)
    opt = nn.AdamWeightDecay(params=group_params, learning_rate=lr, eps=config.epsilon)
    # opt = nn.Momentum(params=group_params, learning_rate=lr, momentum=config.momentum)
    # opt = nn.RMSProp(params=group_params, learning_rate=lr, decay=config.decay,
    #                  momentum=config.momentum, epsilon=1.0)

    if config.device_target in ["Ascend", "GPU"]:
        scale_sense = nn.wrap.loss_scale.DynamicLossScaleUpdateCell(loss_scale_value=2 ** 24,
                                                                    scale_factor=2, scale_window=1000)
        train_net = CustomTrainOneStepWithLossScaleCell(net_with_loss, opt, scale_sense,
                                                        config.enable_ema, config.ema_decay,
                                                        config.enable_clip_norm, config.gradient_norm)
    else:
        raise ValueError
    model = ms.Model(train_net, metrics=metrics, eval_network=dist_eval_network)

    # add callbacks
    cb = [Monitor(lr_init=lr.asnumpy(), model=model, eval_dataset=eval_dataset)]

    ckpt_prefix = model_name
    ckpt_save_dir = os.path.join(config.save_checkpoint_path, "ckpt_" + str(config.rank_id))
    if config.save_checkpoint and config.rank_id == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max,
                                     async_save=True)
        ckpt_cb = ModelCheckpoint(prefix=ckpt_prefix, directory=ckpt_save_dir, config=config_ck)
        cb += [ckpt_cb]

    print("============== Starting Training ==============")
    model.train(config.num_epochs, dataset, callbacks=cb, dataset_sink_mode=True)
    print("============== End Training ==============")


if __name__ == '__main__':
    ms.set_seed(1)
    train()
