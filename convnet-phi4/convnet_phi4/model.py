from __future__ import annotations


import torch
import pytorch_lightning as pl

import flows.phi_four
import flows.transforms

from convnet_phi4.layers import CouplingLayer, GlobalRescalingLayer
import convnet_phi4.utils as utils

NonNegativeFloat = float
IterableDataset = torch.utils.data.IterableDataset
PositiveInt = int


class Model(pl.LightningModule):
    def __init__(
        self,
        flow=None,
        target=None,
    ):
        super().__init__()

        layers = [
            CouplingLayer(
                flows.transforms.PointwiseAffineTransform(),
                {
                    "hidden_shape": [2, 2],
                    "activation": torch.nn.Tanh(),
                    "final_activation": torch.nn.Identity(),
                    "use_bias": False,
                },
            )
            for _ in range(2)
        ]
        layers.append(GlobalRescalingLayer())

        self.flow = utils.Flow(*layers)
        self.action = flows.phi_four.PhiFourAction.from_deldebbio2021(
            beta=0.5, lam=0.5
        )

        self.curr_iter = 0

    def log_state(self, phi):
        self.logger.experiment.add_histogram(
            "phi", phi.flatten(), self.curr_iter
        )
        self.logger.experiment.add_histogram(
            "action", self.action(phi).flatten(), self.curr_iter
        )

    def forward(self, batch):
        z, log_prob_z = batch
        phi, log_det_jacob = self.flow(z)
        log_prob_phi = -self.action(phi)
        weights = log_prob_z - log_det_jacob - log_prob_phi

        self.curr_iter += 1
        if self.curr_iter % 200 == 0:
            self.log_state(phi)

        return phi, weights

    def training_step(self, batch, batch_idx):
        _, weights = self.forward(batch)
        loss = weights.mean()
        self.log("loss", loss, logger=True)
        self.lr_schedulers().step()
        return loss

    def validation_step(self, batch, batch_idx):
        phi, weights = self.forward(batch)
        loss = weights.mean()
        acceptance = utils.metropolis_acceptance(weights)
        metrics = dict(loss=loss, acceptance=acceptance)
        self.log_dict(metrics, prog_bar=False, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_steps
        )
        return [optimizer], [scheduler]

    @torch.no_grad()
    def sample(self, prior: IterableDataset, n_iter: PositiveInt = 1):
        phi, weights = self.forward(next(prior))
        for _ in range(n_iter - 1):
            _phi, _weights = self.forward(next(prior))
            phi = torch.cat((phi, _phi), dim=0)
            weights = torch.cat((weights, _weights), dim=0)
        return phi, weights
