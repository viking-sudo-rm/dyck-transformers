import torch
from torch.nn.utils.rnn import pad_sequence

class TokenIndexer:

    """Basic token indexer that just maps each token to a new ID"""

    UNK = "<UNK>"
    PAD = "<PAD>"
    BOS = "<BOS>"

    def __init__(self, bos=True):
        self.tokens_to_ids = {}
        self.ids_to_tokens = {}
        self.bos = bos

        self.get_index(self.UNK)
        self.get_index(self.PAD)
        self.get_index(self.BOS)

    def get_vocab_size(self):
        return len(self.tokens_to_ids)   
    
    def get_index(self, token, add=True):
        if add and token not in self.tokens_to_ids:
            idx = len(self.tokens_to_ids)
            self.tokens_to_ids[token] = idx
            self.ids_to_tokens[idx] = token
        
        if token in self.tokens_to_ids:
            return self.tokens_to_ids[token]
        else:
            return 

    def get_token(self, idx):
        return self.ids_to_tokens[idx]

    def get_indices(self, tokens, add=True):
        indices = [self.get_index(tok, add) for tok in tokens]
        if self.bos:
            bos = self.get_index(self.BOS)
            indices.insert(0, bos)
        return indices

    def to_padded_tensors(self, sequences):
        pad = self.tokens_to_ids[self.PAD]
        tensors = [torch.tensor(self.get_indices(seq)) for seq in sequences]
        tokens = pad_sequence(tensors, batch_first=True, padding_value=pad)
        mask = (tokens == pad)
        return tokens, mask
