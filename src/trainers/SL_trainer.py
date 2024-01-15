from .trainer import Trainer
from tqdm import tqdm


class SLTrainer(Trainer):
    def __init__(self, model, loss_fn, device, logger):
        super().__init__(model, device, logger)
        self.loss_fn = loss_fn

    def _compute_loss(self, inputs, targets):
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        return outputs, loss

    def _batch_iteration(self, dataloader, is_train, optimizer, metrics, total_loss, total_metrics, description):
        for inputs, targets in tqdm(dataloader, desc=description):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            if is_train:
                optimizer.zero_grad()

            outputs, loss = self._compute_loss(inputs, targets)

            if is_train:
                loss['total'].backward()
                optimizer.step()

            for loss_term in loss:
                total_loss[loss_term] = total_loss.get(loss_term, 0.0) + loss[loss_term].item()

            for metric in metrics:
                total_metrics[metric] += metrics[metric](outputs, targets).item()

