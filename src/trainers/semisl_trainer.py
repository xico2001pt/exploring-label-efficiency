import torch
from tqdm import tqdm
from .trainer import Trainer


class SemiSLTrainer(Trainer):
    def __init__(self, model, device, logger, val_loss_fn, method):
        super().__init__(model, device, logger, val_loss_fn)
        self.method = method
        self.method.set_model(model)

    def on_change_epoch(self, epoch, num_epochs):
        self.method.on_change_epoch(epoch, num_epochs)

    def get_num_batches(self, num_labeled_batches, num_unlabeled_batches):
        if self.method.truncate_batches():
            return min(num_labeled_batches, num_unlabeled_batches)
        else:
            return max(num_labeled_batches, num_unlabeled_batches)

    def _compute_semi_supervised_loss(self, labeled, targets, unlabeled):
        labeled_outputs, loss = self.method.compute_loss(labeled, targets, unlabeled)
        return labeled_outputs, loss

    def _semisl_batch_iteration(self, num_batches, labeled_dataloader, unlabeled_dataloader, optimizer, metrics, total_loss, loss_counter, total_metrics, description):
        labeled_dataloader_iter = iter(labeled_dataloader)
        unlabeled_dataloader_iter = iter(unlabeled_dataloader)

        for _ in tqdm(range(num_batches), desc=description):
            try:
                labeled, targets = next(labeled_dataloader_iter)
            except StopIteration:
                labeled, targets = None, None

            try:
                unlabeled = next(unlabeled_dataloader_iter)
            except StopIteration:
                unlabeled = None

            if labeled is not None:
                labeled, targets = labeled.to(self.device), targets.to(self.device)
            if unlabeled is not None:
                unlabeled = unlabeled.to(self.device)

            optimizer.zero_grad()

            labeled_outputs, loss = self._compute_semi_supervised_loss(labeled, targets, unlabeled)

            loss['total'].backward()
            optimizer.step()

            for loss_term in loss:
                total_loss[loss_term] = total_loss.get(loss_term, 0.0) + loss[loss_term].item()
                loss_counter[loss_term] = loss_counter.get(loss_term, 0) + 1

            if labeled_outputs is not None:
                for metric in metrics:
                    total_metrics[metric] += metrics[metric](labeled_outputs, targets).item()

    def _epoch_iteration(self, dataloader, is_train=True, optimizer=None, metrics={}, description="Train"):
        if is_train:
            assert optimizer is not None, "optimizer must be provided for training"

            labeled_dataloader, unlabeled_dataloader = dataloader

            num_labeled_batches = len(labeled_dataloader)
            num_unlabeled_batches = len(unlabeled_dataloader)

            num_batches = self.get_num_batches(num_labeled_batches, num_unlabeled_batches)

            self.model.train()

            total_loss = {}  # A dictionary is used to support multiple loss terms
            loss_counter = {}
            total_metrics = {metric: 0.0 for metric in metrics}

            with torch.set_grad_enabled(True):
                self._semisl_batch_iteration(num_batches, labeled_dataloader, unlabeled_dataloader, optimizer, metrics, total_loss, loss_counter, total_metrics, description)

            for loss_term in total_loss:
                total_loss[loss_term] /= loss_counter.get(loss_term, 1)

            for metric in total_metrics:
                total_metrics[metric] /= num_labeled_batches

            return total_loss, total_metrics
        else:
            return super()._epoch_iteration(dataloader, is_train, optimizer, metrics, description)
