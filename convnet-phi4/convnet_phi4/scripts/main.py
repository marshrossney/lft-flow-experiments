from __future__ import annotations

import pytorch_lightning as pl
import torch

from convnet_phi4.config import parser
from convnet_phi4.model import Model
from convnet_phi4.utils import Prior

config = parser.parse_args()


def main():
    model = Model(config.flow, config.target)

    trainer = pl.Trainer(
        max_steps=config.n_train,
        val_check_interval=config.val_interval,
        limit_val_batches=1,
        enable_checkpointing=False,
    )

    dist = torch.distributions.Normal(
        loc=torch.zeros(config.lattice_shape),
        scale=torch.ones(config.lattice_shape),
    )
    train_dataloader = Prior(dist, sample_shape=[config.n_batch, 1])
    val_dataloader = Prior(dist, sample_shape=[config.n_batch_val, 1])

    trainer.validate(model, val_dataloader)
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
