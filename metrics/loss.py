class LossMetric:
    def __init__(self):
        self.loss = 0.
        self.count = 0

    def update(self, val_batch, val_batch_mask, val_batch_cont, output):
        self.loss += output.loss.item() * len(val_batch)
        self.count += len(val_batch)
    
    def get_value(self) -> float:
        return self.loss / self.count

    def reset(self):
        self.loss = 0.
        self.count = 0