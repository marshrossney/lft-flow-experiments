import math

import torch
import matplotlib.pyplot as plt
import pandas as pd


def _imshow_subplot(L):
    fig, ax = plt.subplots()
    n = min(L, 10)
    step = max(1, L // n)
    ticks = list(range(0, L, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks)
    return fig, ax

def _covariance(phi):
    assert phi.shape[1] == phi.shape[2]
    L = phi.shape[1]
    covariance = torch.cov(phi.flatten(start_dim=1).T)
    assert covariance.shape == torch.Size([L ** 2, L ** 2])
    return covariance

def _correlator(phi):
    covariance = _covariance(phi)
    correlator = covariance.div(covariance.max())  # normalise
    L = phi.shape[1]

    # Restore geometry, exploiting translational invariance
    correlator = torch.stack(
        [
            row.view(L, L).roll((-(i // L), -(i % L)), dims=(0, 1))
            for i, row in enumerate(correlator.split(1, dim=0))
        ],
        dim=0,
    ).mean(dim=0)
    return correlator


def heatmap_covariance(phi):
    fig, ax = _imshow_subplot(phi.shape[1] * phi.shape[2])
    ax.set_title("Covariance matrix")
    im = ax.imshow(_covariance(phi))
    fig.colorbar(im)
    return fig

def heatmap_field_density(phi):
    fig, ax = _imshow_subplot(phi.shape[1])
    ax.set_title(r"$\langle \phi(x) \rangle$")
    im = ax.imshow(phi.mean(dim=0))
    fig.colorbar(im)
    return fig


def heatmap_field_density_deviation(phi):
    fig, ax = _imshow_subplot(phi.shape[1])
    ax.set_title(r"$\langle \phi(x) \rangle / \sigma_{\phi(x)}$")
    im = ax.imshow(phi.mean(dim=0).div(phi.std(dim=0)))
    fig.colorbar(im)
    return fig


def heatmap_correlator(phi):
    fig, ax = _imshow_subplot(phi.shape[1])
    ax.set_title("Two-point correlation")
    im = ax.imshow(_correlator(phi))
    fig.colorbar(im)
    return fig


def heatmap_ising_energy(phi):
    energy = torch.stack(
        [
            phi.roll(1, 1).mul(phi),
            phi.roll(-1, 1).mul(phi),
            phi.roll(1, 2).mul(phi),
            phi.roll(-1, 2).mul(phi),
        ],
        dim=0,
    ).sum(dim=0)

    fig, ax = _imshow_subplot(phi.shape[1])
    ax.set_title("Ising energy")
    im = ax.imshow(energy)
    fig.colorbar(im)
    return fig


def histogram_field_density(phi):
    mag = phi.sum(dim=(1, 2))
    phi_pos = phi[mag > 0]
    phi_neg = phi[mag < 0]
    fig, ax = plt.subplots()
    ax.hist(
        phi_pos.flatten(),
        bins=25,
        density=True,
        histtype="step",
        label=f"$M(\phi) > 0",
    )
    ax.hist(
        phi_neg.flatten(),
        bins=25,
        density=True,
        histtype="step",
        label=f"$M(\phi) < 0",
    )
    ax.legend()
    return fig


def plot_zero_momentum_correlator(phi):
    correlator = _correlator(phi)
    correlator_1d = (correlator.sum(dim=0) + correlator.sum(dim=1)) / 2
    fig, ax = plt.subplots()
    ax.set_title("Zero momentum correlator")
    ax.set_xticks(list(range(L)))
    ax.set_xticklabels(list(range(L)))
    ax.plot(correlator_1d, "o-")
    ax.set_yscale("log")

    # Correlation length computation
    g_tilde_00 = float(covariance.sum())
    g_tilde_10 = float(
        covariance.mul(
            torch.cos(2 * PI / L * torch.arange(L)).view(L, 1)
        ).sum()
    )
    xi_sq = (g_tilde_00 / g_tilde_10 - 1) / (4 * math.sin(PI / L) ** 2)
    try:
        xi = math.sqrt(xi_sq)
    except ValueError:
        print("Unable to compute correlation length")
    else:
        ax.annotate(
            f"Correlation length: {xi:.2g}",
            xy=(0.2, 0.9),
            xycoords="axes fraction",
        )

    return fig

def plot_zero_momentum_correlator_separate_dims(phi):
    correlator = _correlator(phi)
    fig, ax = plt.subplots()
    ax.set_title("Zero momentum correlator")
    ax.set_xticks(list(range(L)))
    ax.set_xticklabels(list(range(L)))
    ax.plot(correlator.sum(dim=0), "o-", label="First dim")
    ax.plot(correlator.sum(dim=1), "o-", label="Second dim")
    ax.set_yscale("log")
    ax.legend()
    return fig
