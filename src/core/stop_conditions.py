class StopPatience:
    def __init__(self, patience, delta):
        self.patience = patience
        self.delta = delta
        self.best = float("inf")
        self.counter = 0

    def __call__(self, train_loss, validation_loss):
        if validation_loss < self.best - self.delta:
            self.best = validation_loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            return True
        return False
