import numpy as np

class CrossEntropyLoss():
    def __init__(self):
        self.name = "CrossEntropy"

    def __call__(self, y_pred, y_true):
        # cross entropy loss for y_pred of shape (N, #classes)
        # y_true is a one-hot encoded vector of shape (N, #classes)
        return np.sum(-np.sum(y_true * np.log(y_pred), axis=1)) / y_pred.shape[0]

    def get_grad_wrt_softmax(self, y_pred, y_true):
        grad = y_pred - y_true
        return grad/y_pred.shape[0]