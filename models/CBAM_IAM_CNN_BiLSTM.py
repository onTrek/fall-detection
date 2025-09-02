import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from models.cbam_block import ImprovedCBAM2D


class CBAM_IAM_CNN_BiLSTM(pl.LightningModule):
    def __init__(self, input_len=100, num_classes=2, lr=0.001, dropout=0.2, weights=[1.0, 1.0]):
        super().__init__()
        self.save_hyperparameters()


        self.conv2d_1 = nn.Conv2d(1, 32, kernel_size=(5, 5), padding=2)
        self.cbam1 = ImprovedCBAM2D(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=(5, 5), padding=2)
        self.cbam2 = ImprovedCBAM2D(64)

        self.pool2 = nn.AdaptiveMaxPool2d((1, input_len//4))

        self.bilstm = nn.LSTM(input_size=3, hidden_size=400, num_layers=2,
                              batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(dropout)

        self._dummy_input_len = input_len
        self._conv_out_size = None
        self._lstm_out_size = None
        self._compute_output_sizes()

        self.fc1 = nn.Linear(self._conv_out_size + self._lstm_out_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

        class_weights = torch.tensor([weights[1] / (weights[0] + weights[1]), weights[0] / (weights[0] + weights[1])], dtype=torch.float32)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        acc = x[:, :, :3]
        gyro = x[:, :, 3:]

        acc_input = acc.permute(0, 2, 1).unsqueeze(1)

        x_cnn = self.pool1(torch.relu(self.cbam1(self.conv2d_1(acc_input))))
        x_cnn = self.pool2(torch.relu(self.cbam2(self.conv2d_2(x_cnn))))
        x_cnn = x_cnn.flatten(start_dim=1)

        x_lstm, _ = self.bilstm(gyro)
        x_lstm = self.dropout(x_lstm)
        x_lstm = x_lstm.flatten(start_dim=1)

        x = torch.cat([x_cnn, x_lstm], dim=1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)

    def _compute_output_sizes(self):
        with torch.no_grad():
            dummy = torch.zeros(1, self._dummy_input_len, 6)
            acc = dummy[:, :, :3]
            gyro = dummy[:, :, 3:]

            acc_input = acc.permute(0, 2, 1).unsqueeze(1)  # (B, 1, 3, T)

            x_cnn = self.pool1(torch.relu(self.cbam1(self.conv2d_1(acc_input))))

            x_cnn = self.pool2(torch.relu(self.cbam2(self.conv2d_2(x_cnn))))

            self._conv_out_size = x_cnn.flatten(start_dim=1).shape[1]

            x_lstm, _ = self.bilstm(gyro)
            self._lstm_out_size = x_lstm.flatten(start_dim=1).shape[1]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
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
        all_probs = torch.softmax(all_logits, dim=1).numpy()  # Per la curva ROC
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
