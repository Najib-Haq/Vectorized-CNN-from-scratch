import numpy as np

from model.nn.Base import Base

class Flatten(Base):
    def __init__(self):
        super().__init__()
        self.name = "Flatten"

    def forward(self, X):
        if self.trainable: self.cache['X_shape'] = X.shape
        output = X.reshape(X.shape[0], -1)
        return output

    def backward(self, dL_dy):
        dL_dX = dL_dy.reshape(self.cache['X_shape'])
        return dL_dX
