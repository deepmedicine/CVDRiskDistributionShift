import os
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torchmetrics.functional import average_precision, auroc


class ValidAurocAuprc(pl.Callback):
    def __init__(self, **kwargs):
        super().__init__()
        self.sig = nn.Sigmoid()

    def on_validation_epoch_start(self, trainer, pl_module):
        self.reset_buffer()

    def to_device(self, batch, device):
        for key, value in batch.items():
            batch[key] = value.to(device)
        return batch

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        batch = self.to_device(batch, pl_module.device)

        loss, y, label = pl_module.shared_step(batch, batch_idx)

        self.pred_list.append(self.sig(y).cpu())
        self.target_list.append(label.cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        label = torch.cat(self.target_list, dim=0).view(-1)
        pred = torch.cat(self.pred_list, dim=0).view(-1)

        auprc_score = average_precision(pred, target=label.type(torch.LongTensor))
        auroc_score = auroc(pred, label.type(torch.LongTensor))

        print('AUROC: {} AUPRC: {}'.format( auroc_score, auprc_score))

        pl_module.log('auprc', auprc_score)
        pl_module.log('auroc', auroc_score)
        self.reset_buffer()

    def reset_buffer(self):
        self.pred_list = []
        self.target_list = []

