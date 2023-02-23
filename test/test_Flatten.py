import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from model.nn.Flatten import Flatten
np.random.seed(42)

def compare_flatten(shape):
    X = np.random.rand(*shape)
    X_torch = torch.from_numpy(X.copy()); X_torch.requires_grad = True
    y = np.random.rand(*(X.reshape(shape[0], -1)).shape)

    # get the layers
    flat = Flatten()
    flat_torch = nn.Flatten()

    # forward
    y_pred = flat.forward(X)
    y_pred_torch = flat_torch(X_torch)
    assert y_pred.shape == y_pred_torch.shape, "Forward Shapes don't match"
    forward_check = np.allclose(y_pred, y_pred_torch.detach().numpy())
    
    # backward
    dL_dy = y_pred - y.copy()
    dL_dX = flat.backward(dL_dy)
    dL_dy_torch = y_pred_torch - torch.from_numpy(y.copy())
    y_pred_torch.backward(dL_dy_torch)

    dL_dX_torch = X_torch.grad
    assert dL_dX.shape == dL_dX_torch.shape, "Backward dX Shapes don't match"
    dx_compare = np.allclose(dL_dX, dL_dX_torch.detach().numpy())

    # return AND of all checks
    assert forward_check, "Forward check failed"
    assert dx_compare, "Backward dX check failed"
    return forward_check and dx_compare

def test_flat1():
    out = compare_flatten(shape=(1, 3, 5, 5)) == True
    assert out, "Basic Flatten (1, 3, 5, 5) failed"

def test_flat2():
    out = compare_flatten(shape=(1, 10, 2, 2)) == True
    assert out, "Basic Flatten (1, 10, 2, 2) failed"

def test_flat3():
    out = compare_flatten(shape=(1, 10, 20)) == True
    assert out, "Basic Flatten (1, 10, 20) failed"
