import torch
import numpy as np
from tqdm import tqdm
from .trainer import Trainer
from ..utils.constants import Constants as c
import torch.optim.swa_utils as swa_utils


class SelfSLTrainer(Trainer):
    def __init__(self, model, device, logger, method, num_batches):
        super().__init__(model, device, logger, None)
        self.method = method
        self.method.set_model(model)
        self.num_batches = num_batches

    def on_start_train(self, train_data):
        self.method.on_start_train(train_data)

    def on_start_epoch(self, epoch):
        self.method.on_start_epoch(epoch)

    def on_end_train(self, train_data):
        self.method.on_end_train(train_data)

    def on_end_epoch(self, epoch):
        self.method.on_end_epoch(epoch)

    def _compute_self_supervised_loss(self, idx, unlabeled):
        outputs, targets, loss = self.method.compute_loss(idx, unlabeled)
        return outputs, targets, loss

    def _selfsl_batch_iteration(self, is_train, num_batches, unlabeled_dataloader, optimizer, metrics, total_loss, loss_counter, total_metrics, description):
        unlabeled_dataloader_iter = iter(unlabeled_dataloader)

        for idx in tqdm(range(num_batches), desc=description):
            unlabeled = next(unlabeled_dataloader_iter)

            unlabeled = unlabeled.to(self.device)

            if is_train:
                optimizer.zero_grad()

            outputs, targets, loss = self._compute_self_supervised_loss(idx, unlabeled)

            if is_train:
                loss['total'].backward()
                optimizer.step()
                if self.ema_model:
                    self.ema_model.update_parameters(self.model)

            for loss_term in loss:
                total_loss[loss_term] = total_loss.get(loss_term, 0.0) + loss[loss_term].item()
                loss_counter[loss_term] = loss_counter.get(loss_term, 0) + 1

            for metric in metrics:
                total_metrics[metric] += metrics[metric](outputs, targets).item()

    def _epoch_iteration(self, dataloader, is_train=True, optimizer=None, metrics={}, description="Train"):
        if is_train:
            assert optimizer is not None, "optimizer must be provided for training"

        self.model.train() if is_train else self.model.eval()

        total_loss = {}  # A dictionary is used to support multiple loss terms
        loss_counter = {}
        total_metrics = {metric: 0.0 for metric in metrics}

        num_batches = min(self.num_batches, len(dataloader)) if is_train else len(dataloader)

        with torch.set_grad_enabled(is_train):
            self._selfsl_batch_iteration(is_train, num_batches, dataloader, optimizer, metrics, total_loss, loss_counter, total_metrics, description)

        for loss_term in total_loss:
            total_loss[loss_term] /= num_batches

        for metric in total_metrics:
            total_metrics[metric] /= num_batches

        return total_loss, total_metrics
