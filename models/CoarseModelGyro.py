import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import torch.nn.functional as F

from models.CoarseBranch import CoarseBranch
from models.FineBranch import FineBranch
from models.GyroBranch import GyroBranch
from models.TemporalBranch import TemporalBranch


# Ensemble model
class CoarseFineGRUGyro(pl.LightningModule):
    def __init__(self, num_classes=2, lr=0.001, dropout=0.2, weight=[1.0, 1.0], input_shape=(6, 140)):
        super().__init__()
        self.save_hyperparameters()

        self.coarse = CoarseBranch()
        self.fine = FineBranch(dropout=dropout)
        self.temporal = TemporalBranch(input_size=input_shape[0]//2)
        self.gyro = GyroBranch(input_size=input_shape[0]//2)


        self.dropout = nn.Dropout(dropout)

        # Calcolo automatico della dimensione dopo i branch CNN
        dummy = torch.zeros(1, 1, input_shape[0]//2, input_shape[1])  # (batch=1, channels=1, 3, 140)
        coarse_dim = self.coarse(dummy).shape[1]
        fine_dim   = self.fine(dummy).shape[1]
        temporal_dim = 64  # hidden size GRU (ultimo stato)
        gyro_dim = 64 # hidden size GRU (ultimo stato)

        total_dim = coarse_dim + fine_dim + temporal_dim + gyro_dim

        self.fc1 = nn.Linear(total_dim, 64)
        self.fc2 = nn.Linear(64, 2)

        class_weights = torch.tensor([weight[1] / (weight[0] + weight[1]),
                                      weight[0] / (weight[0] + weight[1])], dtype=torch.float32)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, x):
        # separazione accel e gyro
        accel = x[:, :3, :]  # [batch, 3, 140]
        gyro = x[:, 3:, :]  # [batch, 3, 140]

        x_cnn = accel.unsqueeze(1)  # [batch, 1, 3, 140]

        # feature extraction
        f_coarse = self.coarse(x_cnn)
        f_fine = self.fine(x_cnn)
        f_temp = self.temporal(accel)
        f_gyro = self.gyro(gyro)

        # concatenazione
        concat = torch.cat((f_coarse, f_fine, f_temp, f_gyro), dim=1)

        # passaggio fully connected
        x = F.relu(self.fc1(concat))

        x = self.dropout(x)
        pred_fusion = self.fc2(x)

        return pred_fusion

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if 'bias' in name or 'bn' in name.lower():
                no_decay.append(param)
            else:
                decay.append(param)

        optimizer = torch.optim.Adam([
            {'params': decay, 'weight_decay': 0.001},
            {'params': no_decay, 'weight_decay': 0.0}
        ], lr=self.hparams.lr)

        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('train_loss_step', loss, on_step=True, on_epoch=False)
        self.log('train_acc_step', acc, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss_step', loss, on_step=True, on_epoch=False)
        self.log('val_acc_step', acc, on_step=True, on_epoch=False)

    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics

        epoch_metrics = {
            'train_loss': metrics.get('train_loss_step').mean().item(),
            'train_acc': metrics.get('train_acc_step').mean().item(),
        }

        self.logger.experiment.log({
            'train_loss_epoch': epoch_metrics['train_loss'],
            'train_acc_epoch': epoch_metrics['train_acc'],
            'epoch': self.current_epoch
        })

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics

        epoch_metrics = {
            'val_loss': metrics.get('train_loss_step').mean().item(),
            'val_acc': metrics.get('train_acc_step').mean().item(),
        }

        self.logger.experiment.log({
            'val_loss_epoch': epoch_metrics['val_loss'],
            'val_acc_epoch': epoch_metrics['val_acc'],
            'epoch': self.current_epoch
        })

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)

        self.log('test_loss', loss, on_step=False, on_epoch=True)

        self.test_step_outputs.append({
            'logits': logits,
            'labels': y
        })

        return {'loss': loss, 'logits': logits, 'labels': y}

    def on_test_epoch_start(self):
        self.test_step_outputs = []

    def on_test_epoch_end(self):
        all_logits = torch.cat([output['logits'] for output in self.test_step_outputs]).cpu()
        all_labels = torch.cat([output['labels'] for output in self.test_step_outputs]).cpu()

        all_preds = torch.argmax(all_logits, dim=1).numpy()
        all_probs = torch.softmax(all_logits, dim=1).numpy()
        all_labels = all_labels.numpy()

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')


        self.log('test_accuracy', accuracy)
        self.log('test_precision', precision)
        self.log('test_recall', recall)
        self.log('test_f1', f1)


        cm = confusion_matrix(all_labels, all_preds)
        fig_cm, ax_cm = plt.subplots(figsize=(8, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('True')
        ax_cm.set_title('Confusion Matrix')

        fig_roc, ax_roc = plt.subplots(figsize=(8, 8))

        if self.hparams.num_classes == 2:
            fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
            roc_auc = auc(fpr, tpr)

            ax_roc.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        else:
            for i in range(self.hparams.num_classes):
                y_true_bin = (all_labels == i).astype(int)
                y_score = all_probs[:, i]

                fpr, tpr, _ = roc_curve(y_true_bin, y_score)
                roc_auc = auc(fpr, tpr)

                ax_roc.plot(fpr, tpr, lw=2, label=f'Classe {i} (AUC = {roc_auc:.2f})')

        ax_roc.plot([0, 1], [0, 1], 'k--', lw=2)
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('Curva ROC')
        ax_roc.legend(loc="lower right")

        self.logger.experiment.log({
            "confusion_matrix": wandb.Image(fig_cm),
            "roc_curve": wandb.Image(fig_roc),
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1
        })

        plt.close(fig_cm)
        plt.close(fig_roc)




