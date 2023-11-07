from argparse import ArgumentParser
from collections import defaultdict
import json
import random
import os
from tqdm import trange
import torch
import logging
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, ConstantLR
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator
from transformers import GPT2LMHeadModel, GPT2Config

from decay_partition import get_decay_partition
from dyck import Dyck2Generator, get_valid_continuations
from token_indexer import TokenIndexer
from metrics import EntropyMetric, LossMetric, ValidityMetric

log = logging.getLogger("main")
logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--n_train", type=int, default=1000000)
    parser.add_argument("--n_eval", type=int, default=5000)
    # === Data generation parameters ===
    parser.add_argument("-p", type=float, default=.5)
    parser.add_argument("-q", type=float, default=.25)
    # === Non-default arguments ===
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--wd", type=float, default=.1)
    parser.add_argument("--beta1", type=float, default=.9)
    parser.add_argument("--beta2", type=float, default=.95)
    # Batch size should be: 500000 tokens / (1024 tokens/example) = 488 examples, which is ~ 30 * 16.
    parser.add_argument("--step_threshold", type=int, default=30)
    parser.add_argument("--save_threshold", type=int, default=5000)
    parser.add_argument("--eval_threshold", type=int, default=1000)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    return parser.parse_args()

def word_in_length_range(dyck_gen, min_length, max_length):
    """Function to generate data instances in a certain length range via rejection sampling"""
    while True:
        word = dyck_gen.generate()
        if min_length <= len(word) <= max_length:
            return word

class Trainer:

    """Class to train models"""

    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = {}

    def train(self, train_dataloader, val_dataloader, args):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        accelerator = Accelerator()
        self.model, self.optimizer, train_dataloader = accelerator.prepare(self.model, self.optimizer, train_dataloader)
        steps = []
        metrics = defaultdict(list)

        last_step = 0
        for epoch in range(args.n_epochs):
            log.info(f"Starting epoch {epoch}...")
            pbar = trange(len(train_dataloader))
            for step, (batch, batch_mask) in enumerate(iter(train_dataloader)):
                self.model.train()
                batch = batch.to(device)
                batch_mask = batch_mask.to(device)
                output = self.model(batch, attention_mask=batch_mask, labels=batch)
                accelerator.backward(output.loss)

                if (step + 1) % args.step_threshold == 0 or step + 1 == len(train_dataloader):
                    self.scheduler.step()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if (step + 1) % args.save_threshold == 0 or step + 1 == len(train_dataloader):
                    if not args.no_save:
                        pbar.set_description("SAVE")
                        accelerator.save_state(output_dir=os.path.join(args.output_dir, str(last_step + step)))

                if (step + 1) % args.eval_threshold == 0 or step + 1 == len(train_dataloader):
                    pbar.set_description("EVAL")
                    with torch.no_grad():
                        self.model.eval()
                        for metric in self.metrics.values():
                            metric.reset()
                        for val_batch, val_batch_mask, val_batch_cont in iter(val_dataloader):
                            val_batch = val_batch.to(device)
                            val_batch_mask = val_batch_mask.to(device)
                            val_batch_cont = val_batch_cont.to(device)
                            val_output = self.model(val_batch, attention_mask=val_batch_mask, labels=val_batch)
                            for metric in self.metrics.values():
                                metric.update(val_batch, val_batch_mask, val_batch_cont, val_output)
                    steps.append(last_step + step)
                    for name, metric in self.metrics.items():
                        metrics[name].append(metric.get_value())
                
                if steps:
                    pbar.set_description(f"train: {output.loss.item():.2f}, " + ", ".join(f"{key}: {values[-1]:.2f}" for key, values in metrics.items()))
                else:
                    pbar.set_description(f"train: {output.loss.item():.2f}")
                pbar.update()
            pbar.close()
            last_step += len(train_dataloader)
            steps_per_epoch = len(train_dataloader)

        results = {
            "step": steps,
            "steps_per_epoch": steps_per_epoch,
        }
        results.update(**metrics)
        return results

def main(args):
    random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dyck_gen = Dyck2Generator(args.p, args.q, max_depth=100)
    indexer = TokenIndexer()

    log.info("Generating train...")
    train = indexer.to_padded_tensors(word_in_length_range(dyck_gen, 2, 50) for _ in trange(args.n_train))
    train_dataloader = DataLoader(TensorDataset(*train), batch_size=args.batch_size, shuffle=True)

    log.info("Generating val data...")
    val_tokens = [word_in_length_range(dyck_gen, 52, 1000) for _ in trange(args.n_eval)]
    val = indexer.to_padded_tensors(val_tokens)
    val_continuations, _ = indexer.to_padded_tensors([get_valid_continuations(tokens) for tokens in val_tokens])
    val_dataloader = DataLoader(TensorDataset(*val, val_continuations), batch_size=args.eval_batch_size)

    log.info("Randomly initializing GPT-2 model...")
    model = GPT2LMHeadModel(GPT2Config(vocab_size=indexer.get_vocab_size()))
    model.to(device)
    all_params, decay, no_decay = get_decay_partition(model.transformer)  # Sidestep weight tying in LM head.
    optim_groups = [
        {"params": [all_params[pn] for pn in sorted(list(decay))], "weight_decay": args.wd},
        {"params": [all_params[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(args.beta1, args.beta2))
    # Linear warmup for first 10000 steps.
    scheduler = ChainedScheduler([
        LinearLR(optimizer, start_factor=1e-3, total_iters=10000 / args.batch_size),
        ConstantLR(optimizer, factor=1., total_iters=1e20),
    ])

    # Train the model, saving intermediate checkpoints and final output.
    trainer = Trainer(model, optimizer, scheduler)
    trainer.metrics["loss"] = LossMetric()
    trainer.metrics["entropy"] = EntropyMetric()
    trainer.metrics["validity"] = ValidityMetric([indexer.get_index(")"), indexer.get_index("]")])

    log.info("Training model!")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as fh:
        json.dump(vars(args), fh)
    blob = trainer.train(train_dataloader, val_dataloader, args)
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as fh:
        json.dump(blob, fh)

if __name__ == "__main__":
    main(parse_args())
