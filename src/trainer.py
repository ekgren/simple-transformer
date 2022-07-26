"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from src.utils import CfgNode as CN


def collater(batch):
    xs = []
    ys = []
    sample_ids = []
    pos_ids = []
    for sample_id, data in enumerate(batch, 1):
        x, y = data
        xs += x
        ys += y
        sample_ids += [sample_id] * len(x)  # repeat sample_id for each x
        pos_ids += [range(len(x))]          # repeat pos_id for each x

    # Flatten batch
    input_data = torch.stack([torch.tensor(xs).reshape(-1),
                              torch.tensor(sample_ids).reshape(-1),
                              torch.tensor(pos_ids).reshape(-1)], dim=0)
    output_data = torch.tensor(ys).view(-1)
    return input_data, output_data


class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.grad_accum_steps = 1
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.loss = None
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            collate_fn=collater,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:
            # for _ in range(config.grad_accum_steps):
                # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch
            for _ in range(config.grad_accum_steps):
                # forward the model
                logits, loss = model(x, y)
                loss = loss / config.grad_accum_steps
                self.loss = loss if self.loss is None else self.loss + loss
                # backprop and update the parameters
                model.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                with torch.no_grad():
                    probs = F.softmax(logits, dim=-1)
                    idx = torch.multinomial(probs, num_samples=1).view(-1)[:-1]
                    x[0, 1:] = idx

            optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow
            self.loss = None

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
