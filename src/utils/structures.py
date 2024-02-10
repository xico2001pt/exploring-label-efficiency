class EMA:  # Exponential Moving Average
    def __init__(self, decay, initial):
        self.decay = decay
        self.value = initial

    def update(self, value):
        self.value = self.decay * self.value + (1 - self.decay) * value

    def update_partial(self, value, start_index, size):
        self.value[start_index:start_index + size] = self.decay * self.value[start_index:start_index + size] + (1 - self.decay) * value

    def get_value(self):
        return self.value
