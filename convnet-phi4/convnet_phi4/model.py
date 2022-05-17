from jsonargparse.typing import PositiveInt
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI

from torchlft.distributions import Prior, PhiFourDistributionIsing
import torchlft.metrics

from convnet_phi4.flow import RealNVP


class Model(pl.LightningModule):
    def __init__(
        self,
        flow: RealNVP,
        prior: Prior,
        target: PhiFourDistributionIsing,
    ):
        super().__init__()
        self.flow = flow
        self.prior = prior
        self.target = target

    def log_state(self, phi):
        self.logger.experiment.add_histogram(
            "phi", phi.flatten(), self.curr_iter
        )
        self.logger.experiment.add_histogram(
            "action", self.target(phi).flatten(), self.curr_iter
        )

    def forward(self, batch):
        z, log_prob_prior = batch
        phi, log_prob_model = self.flow(z, log_prob_prior)
        log_weights = log_prob_model - self.target.log_prob(phi)
        return phi, log_weights

    def train_dataloader(self) -> Prior:
        return self.prior

    def training_step(self, batch, batch_idx):
        _, log_weights = self.forward(batch)
        loss = log_weights.mean()
        self.log("loss", loss, logger=True)
        self.lr_schedulers().step()
        return loss

    def val_dataloader(self) -> Prior:
        return Prior(self.prior.distribution, batch_size=10000)

    def validation_step(self, batch, batch_idx):
        _, log_weights = self.forward(batch)
        metrics = torchlft.metrics.LogWeightMetrics(log_weights)
        metrics_dict = dict(
            kl=metrics.kl_divergence,
            acc=metrics.acceptance,
            r_max=metrics.longest_rejection_run,
            tau=metrics.integrated_autocorrelation,
            ess=metrics.effective_sample_size,
        )
        return metrics_dict

    def validation_epoch_end(self, metrics):
        metrics = metrics[0]
        self.log_dict(metrics, prog_bar=False, logger=True)

    def predict_dataloader(self) -> Prior:
        return Prior(self.prior.distribution, batch_size=10000)

    def predict_step(self, batch, batch_idx):
        phi, log_weights = self.forward(batch)
        metrics = torchlft.metrics.LogWeightMetrics(log_weights)
        metrics_dict = dict(
            kl=metrics.kl_divergence,
            acc=metrics.acceptance,
            r_max=metrics.longest_rejection_run,
            tau=metrics.integrated_autocorrelation,
            ess=metrics.effective_sample_size,
        )
        return metrics_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=0.005)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_steps
        )
        return [optimizer], [scheduler]

    @torch.no_grad()
    def sample(self, n_batches: PositiveInt = 1):
        phi, log_weights = self.forward(self.train_dataloader())
        for _ in range(n_batches - 1):
            _phi, _log_weights = self.forward(next(self.prior))
            phi = torch.cat((phi, _phi), dim=0)
            log_weights = torch.cat((log_weights, _log_weights), dim=0)
        return phi, log_weights
