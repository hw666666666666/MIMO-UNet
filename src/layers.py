import mindspore as ms
import mindspore.nn as nn

from mindspore.common.initializer import VarianceScaling

class BasicConv(nn.Cell):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False,
                 weight_init=VarianceScaling(scale=0.2, mode='fan_out', distribution='truncated_normal'), bias_init="zeros"):
        super(BasicConv, self).__init__()
        has_bias = True
        if bias and norm:
            has_bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(nn.Conv2dTranspose(in_channel, out_channel, kernel_size, pad_mode="pad", padding=padding, stride=stride, has_bias=has_bias, weight_init=weight_init, bias_init=bias_init))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, pad_mode="pad", padding=padding, stride=stride, has_bias=has_bias, weight_init=weight_init, bias_init=bias_init))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU())
        self.main = nn.SequentialCell(layers)

    def construct(self, x):
        return self.main(x)


class ResBlock(nn.Cell):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.SequentialCell(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False, weight_init="zeros", bias_init="zeros")
        )

    def construct(self, x):
        return self.main(x) + x
