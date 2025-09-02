import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

# Coarse branch (1 conv + 1 maxpool)
class CoarseBranch(pl.LightningModule):
    def __init__(self, lr=0.001):
        super().__init__()
        self.save_hyperparameters()

        self.conv = nn.Conv2d(1, 32, kernel_size=(3 ,3), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(1 ,2))

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
