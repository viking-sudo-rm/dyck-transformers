class EntropyMetric:
    def __init__(self):
        self.entropy = 0.
        self.count = 0

    def update(self, val_batch, val_batch_mask, val_batch_cont, output):
        probs = output.logits.softmax(dim=-1)
        cond_entropies = -(probs * probs.log()).sum(dim=2) * val_batch_mask
        entropy = cond_entropies.sum(dim=1) / val_batch_mask.sum()
        self.entropy += entropy.sum().item()
        self.count += len(val_batch)

    def get_value(self) -> float:
        return self.entropy / self.count

    def reset(self):
        self.entropy = 0.
        self.count = 0