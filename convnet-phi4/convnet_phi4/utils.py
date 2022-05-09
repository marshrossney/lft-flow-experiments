from __future__ import annotations

import math
from random import random

import torch

Tensor = torch.Tensor
BoolTensor = torch.BoolTensor


class Prior(torch.utils.data.IterableDataset):
    """Wraps around torch.distributions.Distribution to make it iterable."""

    def __init__(self, distribution, sample_shape: list[int]):
        super().__init__()
        self.distribution = distribution
        self.sample_shape = sample_shape

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Tensor]:
        sample = self.sample()
        return sample, self.log_prob(sample)

    def sample(self) -> Tensor:
        return self.distribution.sample(self.sample_shape)

    def log_prob(self, sample: Tensor) -> Tensor:
        return (
            self.distribution.log_prob(sample).flatten(start_dim=1).sum(dim=1)
        )


class Composition(torch.nn.Sequential):
    """Compose multiple layers."""

    def forward(
        self, x: Tensor, log_det_jacob: Tensor, *args
    ) -> tuple[Tensor]:
        for layer in self:
            x, log_det_jacob = layer(x, log_det_jacob, *args)
        return x, log_det_jacob

    def inverse(
        self, y: Tensor, log_det_jacob: Tensor, *args
    ) -> tuple[Tensor]:
        for layer in reversed(self):
            y, log_det_jacob = layer.inverse(y, log_det_jacob, *args)
        return y, log_det_jacob


class Flow(Composition):
    """Wraps around Composition, starting with zero log det Jacobian."""

    def forward(self, x: Tensor, *args) -> tuple[Tensor]:
        return super().forward(x, torch.zeros(x.shape[0]).to(x.device), *args)

    def inverse(self, y: Tensor, *args) -> tuple[Tensor]:
        return super().inverse(y, torch.zeros(y.shape[0]).to(y.device), *args)


def make_checkerboard(lattice_shape: list[int]) -> BoolTensor:
    """Return a boolean mask that selects 'even' lattice sites."""
    assert all(
        [n % 2 == 0 for n in lattice_shape]
    ), "each lattice dimension should be even"
    checkerboard = torch.full(lattice_shape, False)
    if len(lattice_shape) == 1:
        checkerboard[::2] = True
    elif len(lattice_shape) == 2:
        checkerboard[::2, ::2] = True
        checkerboard[1::2, 1::2] = True
    else:
        raise NotImplementedError("d > 2 currently not supported")
    return checkerboard


def prod(iterable):
    """Return product of elements of iterable."""
    out = 1
    for el in iterable:
        out *= el
    return out


def metropolis_acceptance(weights: Tensor) -> float:
    """Returns the fraction of configs that pass the Metropolis test."""
    weights = weights.tolist()
    curr_weight = weights.pop(0)
    history = []

    for prop_weight in weights:
        if random() < min(1, math.exp(curr_weight - prop_weight)):
            curr_weight = prop_weight
            history.append(1)
        else:
            history.append(0)

    return sum(history) / len(history)
