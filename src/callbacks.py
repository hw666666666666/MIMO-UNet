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


import os
import time
import numpy as np
import mindspore as ms
from mindspore.train.callback import Callback
from mindspore.train.serialization import save_checkpoint
from .model_utils.config import config


class Monitor(Callback):
    """
    Monitor loss and time.

    Args:
        lr_init (numpy array): train lr

    Returns:
        None

    Examples:
        >>> Monitor(100,lr_init=ms.Tensor([0.05]*100).asnumpy())
    """

    def __init__(self, lr_init=None, model=None, eval_dataset=None, save_checkpoint_path=None):
        super(Monitor, self).__init__()
        self.lr_init = lr_init
        self.lr_init_len = len(lr_init)
        self.model = model
        self.eval_dataset = eval_dataset
        self.save_checkpoint_path = save_checkpoint_path
        self.best_psnr = 0.

    def epoch_begin(self, run_context):
        self.losses = []
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        epoch_mseconds = (time.time() - self.epoch_time) * 1000

        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        batch_num = cb_params.batch_num
        device_number = cb_params.device_number
        per_step_mseconds = epoch_mseconds / batch_num

        eval_psnr = None
        if self.model is not None and self.eval_dataset is not None:
            eval_psnr = self.model.eval(self.eval_dataset)["psnr"]
        log = "epoch time: {:5.3f}, per step time: {:5.3f}, avg loss: {:5.3f}".format(epoch_mseconds,
                                                                                      per_step_mseconds,
                                                                                      np.mean(self.losses))
        if eval_psnr is not None:
            if eval_psnr > self.best_psnr:
                self.best_psnr = eval_psnr
            log += ", eval_psnr: {:.6f}, best_psnr: {:.6f}".format(eval_psnr, self.best_psnr)
        print(log, flush=True)

    def step_begin(self, run_context):
        self.step_time = time.time()

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        step_mseconds = (time.time() - self.step_time) * 1000
        step_loss = cb_params.net_outputs

        if isinstance(step_loss, (tuple, list)) and isinstance(step_loss[0], ms.Tensor):
            step_loss = step_loss[0]
        if isinstance(step_loss, ms.Tensor):
            step_loss = np.mean(step_loss.asnumpy())

        self.losses.append(step_loss)
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num

        print("epoch: [{:3d}/{:3d}], step:[{:5d}/{:5d}], loss:[{:5.3f}/{:5.3f}], time:[{:5.3f}], lr:[{:5.6f}]".format(
            cb_params.cur_epoch_num -
            1, cb_params.epoch_num, cur_step_in_epoch, cb_params.batch_num, step_loss,
            np.mean(self.losses), step_mseconds, self.lr_init[cb_params.cur_step_num - 1]))
