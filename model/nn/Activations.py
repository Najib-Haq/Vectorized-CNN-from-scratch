import numpy as np

from model.nn.Base import Base

class ReLU(Base):
    def __init__(self):
        super().__init__()
        self.name = 'ReLU'

    def forward(self, X):
        self.cache['X_relu_IDX'] = X>0
        return X*self.cache['X_relu_IDX']

    def backward(self, dL_dy):
        return dL_dy*self.cache['X_relu_IDX']


class Softmax(Base):
    def __init__(self):
        super().__init__()
        self.name = "Softmax"

    def forward(self, X):
        '''
        X is a 2D array of shape (N, #classes)
        '''
        # https://stackoverflow.com/questions/54880369/implementation-of-softmax-function-returns-nan-for-high-inputs
        exp = np.exp(X - np.max(X))
        return exp / np.sum(exp, axis=1, keepdims=True)   

    def backward(self, dL_dy):
        '''
        Here derivation has come from cross_entroy; already has the softmax considered
        '''
        return dL_dy