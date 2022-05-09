from __future__ import annotations


import torch

import convnet_phi4.utils as utils

PositiveInt = int
Tensor = torch.Tensor
Module = torch.nn.Module


class CouplingLayer(torch.nn.Module):
    def __init__(self, transform, net_spec: dict):
        super().__init__()
        self.transform = transform
        self.net_a = self.build_convnet(**net_spec)
        self.net_b = self.build_convnet(**net_spec)

    def build_convnet(
        self,
        hidden_shape: tuple[PositiveInt],
        activation: Module = torch.nn.Tanh(),
        final_activation: Module = torch.nn.Identity(),
        kernel_size: PositiveInt = 3,
        use_bias: bool = True,
    ):
        net_shape = [1, *hidden_shape, self.transform.params_dof]
        activations = [activation for _ in hidden_shape] + [final_activation]

        net = []
        for in_channels, out_channels, activation in zip(
            net_shape[:-1], net_shape[1:], activations
        ):
            convolution = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=1,
                padding_mode="circular",
                stride=1,
                bias=use_bias,
            )
            net.append(convolution)
            net.append(activation)

        return torch.nn.Sequential(*net)

    def forward(self, x_full: Tensor, log_det_jacob: Tensor) -> tuple[Tensor]:
        n_batch, n_channels, l1, l2 = x_full.shape
        if n_channels > 1:
            x, h = torch.tensor_split(x_full, [1], dim=1)  # take first channel
        else:
            x = x_full
        mask = (
            utils.make_checkerboard((l1, l2)).view(1, 1, l1, l2).to(x.device)
        )

        x_a = x.mul(mask)
        x_b = x.mul(~mask)
        y_a, log_det_jacob_a = self.transform(x_a, self.net_b(x_b))
        y_a = y_a.mul(mask)
        y_b, log_det_jacob_b = self.transform(x_b, self.net_a(y_a))
        y_b = y_b.mul(~mask)
        y = y_a.add(y_b)

        if n_channels > 1:
            y_full = torch.cat([y, h], dim=1)
        else:
            y_full = y

        log_det_jacob.add_(log_det_jacob_a)
        log_det_jacob.add_(log_det_jacob_b)

        return y_full, log_det_jacob


class GlobalRescalingLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.log_scale = torch.nn.Parameter(torch.tensor([0.0]))

    def forward(self, x: Tensor, log_det_jacob: Tensor) -> tuple[Tensor]:
        x.mul_(self.log_scale.exp())
        numel = utils.prod(x.shape[1:])
        log_det_jacob.add_(self.log_scale.mul(numel))
        return x, log_det_jacob

    def inverse(self, y: Tensor, log_det_jacob: Tensor) -> tuple[Tensor]:
        y.mul_(self.log_scale.neg().exp())
        numel = utils.prod(y.shape[1:])
        log_det_jacob.add_(self.log_scale.neg().mul(numel))
        return y, log_det_jacob
