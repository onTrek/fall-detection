import torch
import torch.nn as nn
import pytorch_lightning as pl


class TemporalBranch(pl.LightningModule):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, lr=0.001):
        super().__init__()
        self.save_hyperparameters()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )


    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)