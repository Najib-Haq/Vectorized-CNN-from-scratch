import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from model.nn.Activations import ReLU, Softmax
from model.Loss import CrossEntropyLoss
np.random.seed(42)

def compare_relu(shape):
    X = np.random.rand(*shape)
    X_torch = torch.from_numpy(X.copy()); X_torch.requires_grad = True
    y = np.random.rand(*shape)

    # get the layers
    relu = ReLU()
    relu_torch = nn.ReLU()

    # forward
    y_pred = relu.forward(X)
    y_pred_torch = relu_torch(X_torch)
    assert y_pred.shape == y_pred_torch.shape, "Forward Shapes don't match"
    forward_check = np.allclose(y_pred, y_pred_torch.detach().numpy())
    
    # backward
    dL_dy = y_pred - y.copy()
    dL_dX = relu.backward(dL_dy)
    dL_dy_torch = y_pred_torch - torch.from_numpy(y.copy())
    y_pred_torch.backward(dL_dy_torch)

    dL_dX_torch = X_torch.grad
    assert dL_dX.shape == dL_dX_torch.shape, "Backward dX Shapes don't match"
    dx_compare = np.allclose(dL_dX, dL_dX_torch.detach().numpy())

    # return AND of all checks
    assert forward_check, "Forward check failed"
    assert dx_compare, "Backward dX check failed"
    return forward_check and dx_compare


def compare_softmax(shape):
    num_class = shape[1]
    X = np.random.rand(*shape)
    X_torch = torch.from_numpy(X.copy()); X_torch.requires_grad = True
    y = np.random.randint(0, num_class, size=shape[0])
    
    # get the layers
    sm = Softmax()
    sm_torch = nn.Softmax(dim=1)

    # forward
    y_pred = sm.forward(X)
    y_pred_torch = sm_torch(X_torch)
    assert y_pred.shape == y_pred_torch.shape, "Forward Shapes don't match"
    forward_check = np.allclose(y_pred, y_pred_torch.detach().numpy())
    
    # backward
    label = np.zeros((shape[0], num_class))
    label[np.arange(shape[0]), y] = 1
    print(y, label)
    loss = CrossEntropyLoss()
    dL_dy = loss.get_grad_wrt_softmax(y_pred, label)
    dL_dX = sm.backward(dL_dy)
    dL_dy_torch = nn.CrossEntropyLoss()(X_torch, torch.from_numpy(y.copy()).long())
    dL_dy_torch.backward()

    print(loss(y_pred, label), dL_dy_torch)

    dL_dX_torch = X_torch.grad
    assert dL_dX.shape == dL_dX_torch.shape, "Backward dX Shapes don't match"
    dx_compare = np.allclose(dL_dX, dL_dX_torch.detach().numpy())

    # return AND of all checks
    print(dL_dX, dL_dX_torch)

    assert forward_check, "Forward check failed"
    assert dx_compare, "Backward dX check failed"
    return forward_check and dx_compare


def test_relu():
    out = compare_relu(shape=(2, 3, 5, 5)) == True
    assert out, "Basic ReLU (2, 3, 5, 5) failed"

# # TODO: this has problems -> whether the dL_dX and the torch version is correct?
# def test_softmax():
#     out = compare_softmax(shape=(3, 5)) == True
#     assert out, "Basic ReLU (2,10) failed"