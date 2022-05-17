from __future__ import annotations


import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI, MODEL_REGISTRY

from torchlft.distributions import Prior

# @MODEL_REGISTRY
class Model(pl.LightningModule):
    def __init__(
        self,
        arg: int = 1,
    ):
        """Description of model

        arg:
            Description of arg
        """
        super().__init__()
        self.param = torch.nn.Parameter(torch.Tensor([arg]))

    def training_step(self, batch, batch_idx):
        x, log_prob = batch
        loss = (x - self.param).pow(2).mean()
        return loss

    def train_dataloader(self):
        return Prior(
            torch.distributions.Normal(
                torch.full((10, 10), fill_value=3.5),
                torch.ones(10, 10),
            ),
            10,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


if __name__ == "__main__":
    cli = LightningCLI(Model, parser_kwargs={"error_handler": None})
