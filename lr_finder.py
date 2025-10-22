# lr_finder.py
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import copy

class LRFinder:
    """
    Learning Rate Range Finder.

    This class helps you find an optimal learning rate for your model by
    incrementally increasing the learning rate and recording the loss at each step.
    """
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.history = {"lr": [], "loss": []}
        self.best_loss = float('inf')

        # Save the initial state of the model and optimizer
        self.initial_state = {
            'model': copy.deepcopy(self.model.state_dict()),
            'optimizer': copy.deepcopy(self.optimizer.state_dict())
        }

    def range_test(self, dataloader, start_lr=1e-7, end_lr=1, num_iter=100, is_pretrain=False):
        """
        Performs the learning rate range test.

        Args:
            dataloader: The data loader to use for the test.
            start_lr: The starting learning rate.
            end_lr: The ending learning rate.
            num_iter: The number of iterations to perform.
            is_pretrain: A flag to indicate if this is for the pre-training phase.
        """
        lr_scheduler = np.geomspace(start_lr, end_lr, num_iter)
        self.model.train()
        data_iter = iter(dataloader)

        progress_bar = tqdm(range(num_iter), desc="LR Range Test")
        for i in progress_bar:
            try:
                if is_pretrain:
                    batch, _, _ = next(data_iter)
                else:
                    batch, target_batch, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                if is_pretrain:
                    batch, _, _ = next(data_iter)
                else:
                    batch, target_batch, _ = next(data_iter)


            lr = lr_scheduler[i]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            batch = batch.to(self.device).float()
            if is_pretrain:
                # For pre-training, we use a dummy target
                target = torch.randn_like(self.model(batch))
            else:
                target = target_batch.to(self.device).float().unsqueeze(1)


            # Forward pass
            preds = self.model(batch)
            loss = self.loss_fn(preds, target)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.history["lr"].append(lr)
            self.history["loss"].append(loss.item())
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.6f}")

        # Restore the initial state
        self.model.load_state_dict(self.initial_state['model'])
        self.optimizer.load_state_dict(self.initial_state['optimizer'])
        print("Model and optimizer states have been restored to their initial values.")

    def plot(self):
        """
        Plots the learning rate vs. loss.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.history["lr"], self.history["loss"])
        plt.xscale("log")
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("Learning Rate Range Test")
        plt.grid(True)
        plt.show()