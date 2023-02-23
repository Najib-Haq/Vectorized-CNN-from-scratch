import numpy as np

class ReduceLROnPlateau:
    def __init__(self, factor=0.1, patience=10, verbose=0):
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.best_metric = 0
        self.counter = 0

    def step(self, metric, optimizer):
        if metric > self.best_metric:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                optimizer.lr *= self.factor
                self.counter = 0
                print(f"Reduced learning rate to {optimizer.lr}")
                            
