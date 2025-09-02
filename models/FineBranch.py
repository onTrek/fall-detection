import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F


class FineBranch(pl.LightningModule):
    def __init__(self, lr=0.001, dropout=0.2):
        super().__init__()
        self.save_hyperparameters()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3 ,3), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(1 ,2))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(1 ,3), padding=(0 ,1))
        self.pool2 = nn.MaxPool2d(kernel_size=(1 ,2))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.pool2(x)
        x = torch.flatten(x, start_dim=1)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)