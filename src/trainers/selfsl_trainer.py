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

    def on_optimize_step(self):
        self.method.on_optimize_step()

    def on_end_train(self, train_data):
        self.method.on_end_train(train_data)

    def on_end_epoch(self, epoch):
        self.method.on_end_epoch(epoch)

    def _compute_self_supervised_loss(self, idx, unlabeled):
        outputs, targets, loss = self.method.compute_loss(idx, unlabeled)
        return outputs, targets, loss

    def _selfsl_batch_iteration(self, num_batches, unlabeled_dataloader, optimizer, metrics, total_loss, loss_counter, total_metrics, description):
        unlabeled_dataloader_iter = iter(unlabeled_dataloader)

        for idx in tqdm(range(num_batches), desc=description):
            unlabeled = next(unlabeled_dataloader_iter)

            unlabeled = unlabeled.to(self.device)

            optimizer.zero_grad()

            outputs, targets, loss = self._compute_self_supervised_loss(idx, unlabeled)

            loss['total'].backward()
            optimizer.step()

            self.on_optimize_step()
            if self.ema_model:
                self.ema_model.update_parameters(self.model)

            for loss_term in loss:
                total_loss[loss_term] = total_loss.get(loss_term, 0.0) + loss[loss_term].item()
                loss_counter[loss_term] = loss_counter.get(loss_term, 0) + 1

            for metric in metrics:
                total_metrics[metric] += metrics[metric](outputs, targets).item()

    def _epoch_iteration(self, dataloader, is_train=True, optimizer=None, metrics={}, description="Train"):
        assert is_train, "SelfSLTrainer is only for training"
        assert optimizer is not None, "optimizer must be provided for training"

        self.model.train()

        total_loss = {}  # A dictionary is used to support multiple loss terms
        loss_counter = {}
        total_metrics = {metric: 0.0 for metric in metrics}

        num_batches = min(self.num_batches, len(dataloader))

        with torch.set_grad_enabled(is_train):
            self._selfsl_batch_iteration(
                num_batches,
                dataloader,
                optimizer,
                metrics,
                total_loss,
                loss_counter,
                total_metrics,
                description
            )

        for loss_term in total_loss:
            total_loss[loss_term] /= num_batches

        for metric in total_metrics:
            total_metrics[metric] /= num_batches

        return total_loss, total_metrics

    # train() method to prevent validation
    def train(
        self,
        train_dataloader,
        max_num_batches,
        num_epochs,
        optimizer,
        scheduler=None,
        stop_condition=None,
        metrics={},
        ema_decay=0.0,
        start_epoch=1,
        save_freq=1,
    ):
        self.logger.info(f"Training for {num_epochs - start_epoch + 1} epochs")

        train_history = {"loss": {}, "metrics": {metric: [] for metric in metrics}}
        best_train_loss = np.inf

        if self.train_data:
            self.on_start_train(self.train_data)

        self.ema_model = swa_utils.AveragedModel(
            self.model,
            multi_avg_fn=swa_utils.get_ema_multi_avg_fn(ema_decay)
        ) if ema_decay is not None and ema_decay > 0.0 else None

        for epoch in range(start_epoch, num_epochs + 1):
            self.logger.info(f"Epoch {epoch}/{num_epochs}")

            self.on_start_epoch(epoch)

            train_loss, train_metrics = self._epoch_iteration(
                train_dataloader,
                is_train=True,
                optimizer=optimizer,
                metrics=metrics,
                description="Train",
            )

            self.on_end_epoch(epoch)

            self._log_epoch_stats(train_loss, train_metrics, "Train")

            if train_loss['total'] < best_train_loss:
                best_train_loss = train_loss['total']
                self._save_checkpoint(c.Trainer.Checkpoints.BEST_CHECKPOINT_FILENAME, optimizer, epoch)

            if save_freq and (epoch % save_freq == 0 or epoch == num_epochs):
                self._save_checkpoint(c.Trainer.Checkpoints.LATEST_CHECKPOINT_FILENAME, optimizer, epoch)

            for loss_term in train_loss:
                train_history["loss"].setdefault(loss_term, []).append(train_loss[loss_term])

            for metric in metrics:
                train_history["metrics"][metric].append(train_metrics[metric])

            self._save_stats(train_history["loss"], train_history["metrics"], "train")

            if stop_condition and stop_condition(train_loss['total'], None):
                self.logger.warning("Stopping due to stop condition")
                self.logger.add_log_entry("early_stopped", epoch)
                break

            if scheduler:
                scheduler.step()

        if self.ema_model:
            dt = train_dataloader if isinstance(train_dataloader, torch.utils.data.DataLoader) else train_dataloader[0]
            swa_utils.update_bn(dt, self.ema_model, device=self.device)

        if self.train_data:
            self.on_end_train(self.train_data)

        self.logger.info(f"Best train loss: {best_train_loss}")
