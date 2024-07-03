import torch
from tqdm import tqdm
from .trainer import Trainer


class SemiSLTrainer(Trainer):
    def __init__(self, model, device, logger, val_loss_fn, method):
        super().__init__(model, device, logger, val_loss_fn)
        self.method = method
        self.method.set_model(model)

    def on_start_train(self, train_data):
        self.method.on_start_train(train_data)

    def on_start_epoch(self, epoch):
        self.method.on_start_epoch(epoch)

    def on_end_train(self, train_data):
        self.method.on_end_train(train_data)

    def on_end_epoch(self, epoch):
        self.method.on_end_epoch(epoch)

    def _compute_semi_supervised_loss(self, idx, labeled, targets, unlabeled):
        labeled_outputs, targets, loss = self.method.compute_loss(idx, labeled, targets, unlabeled)
        return labeled_outputs, targets, loss

    def _semisl_batch_iteration(self, num_batches, labeled_dataloader, unlabeled_dataloader, optimizer, metrics, total_loss, loss_counter, total_metrics, description):
        labeled_dataloader_iter = iter(labeled_dataloader)
        unlabeled_dataloader_iter = iter(unlabeled_dataloader)

        for idx in tqdm(range(num_batches), desc=description):
            labeled, targets = next(labeled_dataloader_iter)
            unlabeled = next(unlabeled_dataloader_iter)

            labeled, targets = labeled.to(self.device), targets.to(self.device)
            unlabeled = unlabeled.to(self.device)

            optimizer.zero_grad()

            labeled_outputs, targets, loss = self._compute_semi_supervised_loss(idx, labeled, targets, unlabeled)

            loss['total'].backward()
            optimizer.step()
            if self.ema_model:
                self.ema_model.update_parameters(self.model)

            for loss_term in loss:
                total_loss[loss_term] = total_loss.get(loss_term, 0.0) + loss[loss_term].item()
                loss_counter[loss_term] = loss_counter.get(loss_term, 0) + 1

            for metric in metrics:
                total_metrics[metric] += metrics[metric](labeled_outputs, targets).item()

    def _epoch_iteration(self, dataloader, is_train=True, optimizer=None, metrics={}, description="Train"):
        if is_train:
            assert optimizer is not None, "optimizer must be provided for training"

            labeled_dataloader, unlabeled_dataloader = dataloader

            num_labeled_batches = len(labeled_dataloader)
            num_unlabeled_batches = len(unlabeled_dataloader)

            num_batches = min(num_labeled_batches, num_unlabeled_batches)

            self.model.train()

            total_loss = {}  # A dictionary is used to support multiple loss terms
            loss_counter = {}
            total_metrics = {metric: 0.0 for metric in metrics}

            with torch.set_grad_enabled(True):
                self._semisl_batch_iteration(
                    num_batches,
                    labeled_dataloader,
                    unlabeled_dataloader,
                    optimizer,
                    metrics,
                    total_loss,
                    loss_counter,
                    total_metrics,
                    description
                )

            for loss_term in total_loss:
                total_loss[loss_term] /= loss_counter.get(loss_term, 1)

            for metric in total_metrics:
                total_metrics[metric] /= num_labeled_batches

            return total_loss, total_metrics
        else:
            return super()._epoch_iteration(dataloader, is_train, optimizer, metrics, description)
