import torch

class ValidityMetric:
    def __init__(self, close_tokens):
        self.close_tokens = close_tokens
        self.validity = 0.
        self.count = 0

    def update(self, val_batch, val_batch_mask, val_batch_cont, output):
        probs = output.logits.softmax(dim=-1)
        preds = probs.argmax(dim=-1)
        invalid = torch.zeros_like(preds)
        for close_token in self.close_tokens:
            invalid |= (preds == close_token) & (preds != val_batch_cont)
        n_valid = ((1 - invalid) * val_batch_mask).sum()
        n_total = val_batch_mask.sum()
        self.validity += n_valid.item() / n_total.item()
        self.count += len(val_batch)

    def get_value(self) -> float:
        return self.validity / self.count

    def reset(self):
        self.validity = 0.
        self.count = 0