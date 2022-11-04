# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""evaluation metric."""


import mindspore as ms
import mindspore.communication as comm
import mindspore.ops as ops
import mindspore.nn as nn


class PSNRCell(nn.Cell):
    r"""
    PSNRCell
    """

    def __init__(self, network, run_distribute):
        super(PSNRCell, self).__init__(auto_prefix=False)
        self._network = network
        # self.argmax = ops.Argmax()
        # self.equal = ops.Equal()
        # self.cast = ops.Cast()
        self.reduce_sum = ops.ReduceSum()
        self.psnr = ms.nn.PSNR(max_val=1.0)
        self.run_distribute = run_distribute
        if run_distribute:
            self.allreduce = ops.AllReduce(ops.ReduceOp.SUM, comm.GlobalComm.WORLD_COMM_GROUP)

    def construct(self, data, label):
        outputs = self._network(data)
        # y_pred = self.argmax(outputs)
        # y_pred = self.cast(y_pred, ms.int32)
        # y_correct = self.equal(y_pred, label)
        # y_correct = self.cast(y_correct, ms.float32)
        # y_correct = self.reduce_sum(y_correct)
        y_pred = outputs[2]
        y_psnr = self.psnr(y_pred, label)
        y_psnr = self.reduce_sum(y_psnr)
        if self.run_distribute:
            # y_correct = self.allreduce(y_correct)
        # return (y_correct,)
            y_psnr = self.allreduce(y_psnr)
        return (y_psnr,)


class DistPSNR(nn.Metric):
    r"""
    DistPSNR
    """

    def __init__(self, batch_size, device_num):
        super(DistPSNR, self).__init__()
        self.clear()
        self.batch_size = batch_size
        self.device_num = device_num

    def clear(self):
        """Clears the internal evaluation result."""
        self._correct_num = 0
        self._total_num = 0

    def update(self, *inputs):
        """
        Updates the internal evaluation result :math:`y_{pred}` and :math:`y`.

        Args:
            inputs: Input `y_correct`. `y_correct` is a `scalar Tensor`.
                `y_correct` is the right prediction count that gathered from all devices
                it's a scalar in float type

        Raises:
            ValueError: If the number of the input is not 1.
        """

        if len(inputs) != 1:
            raise ValueError('Distribute accuracy needs 1 input (y_correct), but got {}'.format(len(inputs)))
        y_correct = self._convert_data(inputs[0])
        self._correct_num += y_correct
        self._total_num += self.batch_size * self.device_num

    def eval(self):
        """
        Computes the accuracy.

        Returns:
            Float, the computed result.

        Raises:
            RuntimeError: If the sample size is 0.
        """

        if self._total_num == 0:
            raise RuntimeError('Accuracy can not be calculated, because the number of samples is 0.')
        return self._correct_num / self._total_num
