from jsonargparse import ArgumentParser, ActionConfigFile
from jsonargparse.typing import PositiveInt
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.utilities.model_summary import summarize
import torch
import yaml

from torchlft.distributions import (
    Prior,
    FreeScalarDistribution,
    PhiFourDistributionStandard,
    PhiFourDistributionIsing,
)

from convnet_phi4.model import Model
from convnet_phi4.flow import RealNVP, NetSpec

parser = ArgumentParser("my app")

parser.add_argument("--lattice_shape", type=PositiveInt, nargs=2)
parser.add_argument("--batch_size", type=PositiveInt)
parser.add_argument("--n_blocks", type=PositiveInt)
parser.add_argument("--conv", type=bool)
parser.add_argument("--net_spec", type=NetSpec)
parser.add_argument("--target", type=FreeScalarDistribution)
parser.add_argument("--trainer", type=pl.Trainer)

parser.add_argument("-c", "--config", action=ActionConfigFile)


def _train_loop(prior, target, flow, n_steps=300):
    optimizer = torch.optim.Adam(flow.parameters(), lr=0.05)
    for _ in range(n_steps):
        z, log_prob_z = next(prior)
        print(float(log_prob_z.mean()))
        phi, log_prob_phi = flow(z, log_prob_z)
        log_prob_target = target.log_prob(phi)
        print(
            float(log_prob_phi.mean()),
            float(log_prob_target.mean()),
        )
        print(float(phi.mean()), float(phi.std()))
        print("\n")
        log_weights = log_prob_phi - target.log_prob(phi)
        loss = log_weights.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(float(loss))

    return float(loss)


def main():
    config = parser.parse_args()
    config = parser.instantiate_classes(config)

    flow = RealNVP(
        config.lattice_shape,
        config.n_blocks,
        config.conv,
        config.net_spec,
    )
    prior = Prior(
        distribution=torch.distributions.Normal(
            loc=torch.zeros(config.lattice_shape).unsqueeze(0),
            scale=torch.ones(config.lattice_shape).unsqueeze(0),
        ),
        batch_size=config.batch_size,
    )
    """prior = Prior(
            distribution=FreeScalarDistribution(config.lattice_shape[0], m_sq=0.44),
            batch_size=config.batch_size,
    )"""

    # loss = _train_loop(prior, target, flow)

    model = Model(
        flow,
        prior,
        config.target,
    )
    summary = summarize(model)
    with open("summary.txt", "w") as file:
        file.write(str(summary))

    config.trainer.fit(model)
    (metrics,) = config.trainer.validate(model)
    with open("metrics.yaml", "w") as file:
        yaml.safe_dump(metrics, file)


if __name__ == "__main__":
    main()
