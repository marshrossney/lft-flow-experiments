from dataclasses import dataclass, field, asdict
from typing import Union

from jsonargparse.typing import PositiveInt
import torch

import torchlft.flows.base
import torchlft.flows.coupling
import torchlft.flows.unconditional
import torchlft.transforms
import torchlft.utils


@dataclass
class NetSpec:
    hidden_shape: list[PositiveInt]
    activation: torch.nn.Module = torch.nn.Tanh()
    final_activation: torch.nn.Module = torch.nn.Identity()
    use_bias: bool = True
    kernel_size: PositiveInt = 3


class CouplingLayerConvnet(torchlft.flows.coupling.CouplingLayer):
    def __init__(self, transform, transform_mask, net_spec):
        super().__init__(transform, transform_mask)
        self.net = self._build_convnet(**net_spec)

    def _build_convnet(
        self,
        hidden_shape,
        activation,
        final_activation,
        kernel_size,
        use_bias,
    ):
        channels = [1, *hidden_shape, self._transform.n_params]
        activations = [activation for _ in hidden_shape] + [final_activation]

        net = []
        for in_channels, out_channels, activation in zip(
            channels[:-1], channels[1:], activations
        ):
            convolution = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                padding_mode="circular",
                stride=1,
                bias=use_bias,
            )
            net.append(convolution)
            net.append(activation)

        return torch.nn.Sequential(*net)

    def net_forward(self, x_masked: torch.Tensor) -> torch.Tensor:
        return self.net(x_masked).unsqueeze(dim=2)


class CouplingLayerDensenet(torchlft.flows.coupling.CouplingLayer):
    def __init__(self, transform, transform_mask, net_spec):
        super().__init__(transform, transform_mask)
        self.net = self._build_densenet(**net_spec)

    def _build_densenet(
        self,
        hidden_shape,
        activation,
        final_activation,
        use_bias,
        kernel_size=None,
    ):
        nodes = [50, *hidden_shape, 50 * self._transform.n_params]
        activations = [activation for _ in hidden_shape] + [final_activation]

        net = []
        for in_nodes, out_nodes, activation in zip(
            nodes[:-1], nodes[1:], activations
        ):
            linear = torch.nn.Linear(
                in_features=in_nodes,
                out_features=out_nodes,
                bias=use_bias,
            )
            net.append(linear)
            net.append(activation)

        return torch.nn.Sequential(*net)

    def net_forward(self, x_masked: torch.Tensor) -> torch.Tensor:
        v_in = x_masked[..., self._condition_mask]
        v_out = self.net(v_in)
        params = torch.zeros(
            x_masked.shape[0],  # batch size
            self._transform.n_params,  # parameters
            *x_masked.shape[1:],  # configuration shape
        )
        params.masked_scatter_(self._transform_mask, v_out)
        return params


class RealNVP(torchlft.flows.base.Flow):
    def __init__(
        self,
        lattice_shape: list[PositiveInt],
        n_blocks: PositiveInt,
        conv: bool,
        net_spec: NetSpec,
    ):
        checker = torchlft.utils.make_checkerboard(lattice_shape).unsqueeze(
            dim=0
        )
        transform = torchlft.transforms.AffineTransform()
        Layer = CouplingLayerConvnet if conv else CouplingLayerDensenet

        layers = []
        for _ in range(n_blocks):
            layers.append(Layer(transform, checker, asdict(net_spec)))
            layers.append(Layer(transform, ~checker, asdict(net_spec)))
        layers.append(torchlft.flows.unconditional.GlobalRescalingLayer())
        """layers.append(
            torchlft.flows.unconditional.UnconditionalLayer(
                torchlft.transforms.Translation()
            )
        )"""

        super().__init__(*layers)
