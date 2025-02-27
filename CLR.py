# CLR.py
import tensorflow as tf
from keras.callbacks import Callback
import numpy as np

class CyclicLR(Callback):
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular'):
        super(CyclicLR, self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.clr_iterations = 0
        self.trn_iterations = 0
        self.history = {}

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.mode == 'triangular':
            return self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))
        return self.base_lr  # Default to base_lr if mode not recognized

    def on_train_begin(self, logs=None):
        logs = logs or {}
        self.clr_iterations = 0
        self.model.optimizer.learning_rate.assign(self.base_lr)

    def on_batch_begin(self, batch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        lr = self.clr()
        self.model.optimizer.learning_rate.assign(lr)
        self.history.setdefault('lr', []).append(lr)