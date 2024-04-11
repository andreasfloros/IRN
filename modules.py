import torch as th
from typing import Any, Dict, List, Tuple


class DenseBlock(th.nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 num_convs: int,
                 growth_rate: int,
                 bias: bool) -> None:
        super().__init__()
        self.blocks = th.nn.ModuleList(th.nn.Conv2d(in_channels=in_channels + i * growth_rate,
                                                    out_channels=growth_rate,
                                                    kernel_size=kernel_size,
                                                    padding="same",
                                                    bias=bias)
                                       for i in range(num_convs - 1))
        self.conv_out = th.nn.Conv2d(in_channels=in_channels + (num_convs - 1) * growth_rate,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     padding="same",
                                     bias=bias)

        for block in self.blocks:
            th.nn.init.xavier_normal_(block.weight)
            if bias:
                th.nn.init.zeros_(block.bias)

        th.nn.init.zeros_(self.conv_out.weight)
        if bias:
            th.nn.init.zeros_(self.conv_out.bias)

    def forward(self, inpt: th.Tensor) -> th.Tensor:
        for block in self.blocks:
            inpt = th.cat((inpt, th.nn.functional.leaky_relu(block(inpt), 0.2)), dim=1)
        return self.conv_out(inpt)


class InvBlock(th.nn.Module):

    def __init__(self,
                 c_channels: int,
                 d_channels: int,
                 num_convs: int,
                 kernel_size: int,
                 growth_rate: int,
                 bias: bool) -> None:
        super().__init__()
        self.phi = DenseBlock(in_channels=d_channels,
                              out_channels=c_channels,
                              kernel_size=kernel_size,
                              num_convs=num_convs,
                              growth_rate=growth_rate,
                              bias=bias)
        self.rho = DenseBlock(in_channels=c_channels,
                              out_channels=d_channels,
                              kernel_size=kernel_size,
                              num_convs=num_convs,
                              growth_rate=growth_rate,
                              bias=bias)
        self.eta = DenseBlock(in_channels=c_channels,
                              out_channels=d_channels,
                              kernel_size=kernel_size,
                              num_convs=num_convs,
                              growth_rate=growth_rate,
                              bias=bias)

    def forward(self,
                coarse: th.Tensor,
                details: th.Tensor,
                forward: bool = True) -> Tuple[th.Tensor, th.Tensor]:
        if forward:
            coarse = coarse + self.phi(details)
            exp_tanh_rho = (th.tanh(self.rho(coarse) / 2)).exp()
            details = exp_tanh_rho * details + self.eta(coarse)
        else:
            exp_tanh_rho = (th.tanh(self.rho(coarse) / 2)).exp()
            details = (details - self.eta(coarse)) / exp_tanh_rho
            coarse = coarse - self.phi(details)
        return coarse, details


class Transform(th.nn.Module):

    def __init__(self,
                 c_channels: int,
                 d_channels: int,
                 num_inv: int,
                 num_convs: int,
                 growth_rate: int,
                 kernel_size: int,
                 bias: bool) -> None:
        """
        (N, c_channels, H, W) + (N, d_channels, H, W) <-> (N, c_channels, H, W) + (N, d_channels, H, W)
        """

        super().__init__()
        self.blocks = th.nn.ModuleList(InvBlock(c_channels=c_channels,
                                                d_channels=d_channels,
                                                num_convs=num_convs,
                                                kernel_size=kernel_size,
                                                growth_rate=growth_rate,
                                                bias=bias)
                                       for _ in range(num_inv))

    def forward(self,
                coarse: th.Tensor,
                details: th.Tensor,
                forward: bool = True) -> Tuple[th.Tensor, th.Tensor]:
        for block in (self.blocks if forward else reversed(self.blocks)):
            coarse, details = block(coarse, details, forward)
        return coarse, details


class HaarTransform(th.nn.Module):

    def __init__(self, in_channels: int, num_channels: int):
        """
        (N, in_channels, H, W) <-> (N, num_channels, H // 2, W // 2) + (N, -1, H // 2, W // 2)
        """

        super().__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels

        haar_weights = th.ones(4, 1, 2, 2)

        haar_weights[1, 0, 0, 1] = -1
        haar_weights[1, 0, 1, 1] = -1

        haar_weights[2, 0, 1, 0] = -1
        haar_weights[2, 0, 1, 1] = -1

        haar_weights[3, 0, 1, 0] = -1
        haar_weights[3, 0, 0, 1] = -1

        haar_weights = th.cat([haar_weights] * self.in_channels, 0)
        self.register_buffer("haar_weights", haar_weights)

    def forward(self, inpt: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        (N, in_channels, H, W) -> (N, num_channels, H // 2, W // 2) + (N, -1, H // 2, W // 2)
        """

        out = th.nn.functional.conv2d(inpt, self.haar_weights, bias=None, stride=2, groups=self.in_channels) / 4.
        out = out.reshape([inpt.shape[0], self.in_channels, 4, inpt.shape[2] // 2, inpt.shape[3] // 2])
        out = th.transpose(out, 1, 2)
        out = out.reshape([inpt.shape[0], 4 * self.in_channels, inpt.shape[2] // 2, inpt.shape[3] // 2])
        return out[:, :self.num_channels, ...], out[:, self.num_channels:, ...]

    def inverse(self, coarse: th.Tensor, details: th.Tensor) -> th.Tensor:
        """
        (N, num_channels, H // 2, W // 2) + (N, -1, H // 2, W // 2) -> (N, in_channels, H, W)
        """

        inpt = th.cat([coarse, details], 1)
        out = inpt.reshape([inpt.shape[0], 4, self.in_channels, inpt.shape[2], inpt.shape[3]])
        out = th.transpose(out, 1, 2)
        out = out.reshape([inpt.shape[0], 4 * self.in_channels, inpt.shape[2], inpt.shape[3]])
        return th.nn.functional.conv_transpose2d(out,
                                                 self.haar_weights,
                                                 bias=None,
                                                 stride=2,
                                                 groups=self.in_channels)


class IRN(th.nn.Module):

    def __init__(self,
                 num_channels: int,
                 transform_cfgs: List[Dict[str, Any]]) -> None:
        """
        (N, C, H, W) <-> (N, C, H / 2 ** s, W / 2 ** s) + (N, -1, H / 2 ** s, W / 2 ** s)
        """

        super().__init__()
        log2_scale = len(transform_cfgs)
        self.num_channels = num_channels
        self.transforms = th.nn.ModuleList(Transform(c_channels=num_channels,
                                                     d_channels=num_channels * (4 ** s - 1),
                                                     **transform_cfgs[s - 1])
                                           for s in range(1, log2_scale + 1))
        self.haars = th.nn.ModuleList(HaarTransform(in_channels=num_channels * 4 ** s,
                                                    num_channels=num_channels) for s in range(log2_scale))

    def forward(self, inpt: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        (N, C, H, W) -> (N, C, H / 2 ** s, W / 2 ** s) + (N, -1, H / 2 ** s, W / 2 ** s)
        """

        for haar, transform in zip(self.haars, self.transforms):
            coarse, details = haar(inpt)
            inpt = th.cat(transform(coarse, details), dim=1)
        return inpt[:, :self.num_channels, ...], inpt[:, self.num_channels:, ...]

    def inverse(self, coarse: th.Tensor, details: th.Tensor) -> th.Tensor:
        """
        (N, C, H / 2 ** s, W / 2 ** s) + (N, -1, H / 2 ** log2_scale, W / 2 ** s) -> (N, C, H, W)
        """

        for haar, transform in zip(reversed(self.haars), reversed(self.transforms)):
            coarse, details = transform(coarse, details, False)
            out = haar.inverse(coarse, details)
            coarse, details = out[:, :self.num_channels, ...], out[:, self.num_channels:, ...]
        return out
