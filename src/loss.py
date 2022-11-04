# Copyright 2022 Huawei Technologies Co., Ltd
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
"""
Loss function
"""

import mindspore as ms
import mindspore.nn as nn


class SmoothL1Loss(nn.Cell):
    """SmoothL1Loss"""
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
    
    def construct(self, prediction, label):
        diff = prediction - label
        abs_diff = ms.ops.Abs()(diff)
        smoothL1_sign = ms.ops.stop_gradient(
            ms.ops.Cast()(
                ms.ops.Less()(abs_diff, 1.0), 
                ms.dtype.float32
            )
        )
        loss = smoothL1_sign * 0.5 * abs_diff ** 2 / self.beta + (1.0 - smoothL1_sign) * (abs_diff - 0.5 * self.beta)
        return ms.ops.ReduceMean()(loss)


class ContentLoss(nn.Cell):
    """ContentLoss"""
    def __init__(self):
        super().__init__()
        self.criterion1 = nn.L1Loss()
        # self.criterion1 = nn.MSELoss()
        # self.criterion1 = SmoothL1Loss()
        self.nn_interpolate = nn.ResizeBilinear()

    def interpolate(self, x, scale_factor):
        """interpolate"""
        x_shape = ms.ops.Shape()(x)
        h = int(x_shape[2] * scale_factor)
        w = int(x_shape[3] * scale_factor)
        return ms.ops.ResizeBilinear((h, w))(x)

    def construct(self, pred_img, label_img):
        """construct ContentLoss"""
        label_img2 = self.interpolate(label_img, scale_factor=0.5)
        label_img4 = self.interpolate(label_img, scale_factor=0.25)
        l1 = self.criterion1(pred_img[0], label_img4) # + self.criterion2(pred_img[0], label_img4)
        l2 = self.criterion1(pred_img[1], label_img2) # + self.criterion2(pred_img[1], label_img2)
        l3 = self.criterion1(pred_img[2], label_img) # + self.criterion2(pred_img[2], label_img)
        return l1+l2+l3
