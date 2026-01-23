"""Learning rate schedulers for transformer training.

This module provides custom learning rate schedulers including warmup strategies
that are important for stable transformer training.
"""

import numpy as np
import torch


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warm-up followed by cosine-shaped decay.
    
    This scheduler is particularly useful for transformer models, which are sensitive
    to large gradient updates early in training. The warmup phase starts with a very
    low learning rate and gradually increases it, helping to stabilize training.
    After warmup, the learning rate decays following a cosine curve.
    
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer whose learning rate will be scheduled.
    warmup_iters : int
        The number of iterations (steps) for the linear warm-up of the learning rate.
        During this phase, the learning rate increases linearly from 0 to the base LR.
    cosine_schedule_period_iters : int
        The number of iterations for the cosine half period of the learning rate.
        After warmup, the LR decays following a cosine curve over this many iterations.
    last_epoch : int, optional
        The index of the last epoch/iteration. Default: -1 (starts from beginning).
        
    Example
    -------
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    >>> scheduler = CosineWarmupScheduler(optimizer, warmup_iters=1000, 
    ...                                    cosine_schedule_period_iters=10000)
    >>> for epoch in range(num_epochs):
    ...     for batch in dataloader:
    ...         optimizer.zero_grad()
    ...         loss.backward()
    ...         optimizer.step()
    ...         scheduler.step()  # Step after each batch, not epoch!
    
    Notes
    -----
    - This scheduler should be stepped after each batch/iteration, not after each epoch.
    - The learning rate at iteration t is: base_lr * lr_factor(t)
    - During warmup (t <= warmup_iters): lr_factor = t / warmup_iters
    - After warmup: lr_factor = 0.5 * (1 + cos(π * t / cosine_schedule_period_iters))
    - Both phases are multiplied together during warmup for a smooth transition.
    
    References
    ----------
    This implementation is based on the Casanovo scheduler:
    https://github.com/Noble-Lab/casanovo
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int,
        cosine_schedule_period_iters: int,
        last_epoch: int = -1,
    ):
        """Initialize the scheduler."""
        self.warmup_iters = warmup_iters
        self.cosine_schedule_period_iters = cosine_schedule_period_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute the learning rate for the current iteration.
        
        Returns
        -------
        list of float
            Learning rates for each parameter group in the optimizer.
        """
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        """
        Calculate the learning rate multiplier for the given iteration.
        
        Parameters
        ----------
        epoch : int
            Current iteration number (despite the name, this is actually steps/iterations).
            
        Returns
        -------
        float
            Multiplicative factor to apply to the base learning rate.
        """
        # Cosine decay component
        lr_factor = 0.5 * (
            1 + np.cos(np.pi * epoch / self.cosine_schedule_period_iters)
        )
        
        # Linear warmup component (only applies during warmup phase)
        if epoch <= self.warmup_iters:
            lr_factor *= epoch / self.warmup_iters
            
        return lr_factor
