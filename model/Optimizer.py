import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, model):
        for layer in model.layers:
            layer.update_weights(self.lr)
