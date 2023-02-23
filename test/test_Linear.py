import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from model.nn.Linear import Linear
np.random.seed(42)

def compare_linear(in_features, out_features):
    X = np.random.rand(3, in_features)
    X_torch = torch.from_numpy(X.copy()); X_torch.requires_grad = True
    y = np.random.rand(3, out_features)

    # get the layers
    dense = Linear(in_features, out_features, bias=True)
    dense_torch = nn.Linear(in_features, out_features, bias=True)
    dense_torch.weight = nn.Parameter(torch.from_numpy(dense.state_dict["weight"].copy()))
    dense_torch.bias = nn.Parameter(torch.from_numpy(dense.state_dict["bias"].copy()))

    # forward
    y_pred = dense.forward(X)
    y_pred_torch = dense_torch(X_torch)
    assert y_pred.shape == y_pred_torch.shape, "Forward Shapes don't match"
    forward_check = np.allclose(y_pred, y_pred_torch.detach().numpy())

    # backward
    dL_dy = y_pred - y.copy()
    dL_dX = dense.backward(dL_dy)
    dL_dW = dense.grads["weight"]
    dL_db = dense.grads["bias"]

    dL_dy_torch = y_pred_torch - torch.from_numpy(y.copy())
    y_pred_torch.backward(dL_dy_torch)
    dL_dX_torch = X_torch.grad
    dL_dW_torch = dense_torch.weight.grad
    dL_db_torch = dense_torch.bias.grad
    
    print(dL_dW, dL_dW_torch.detach().numpy())
    print("DB: ", dL_db, dL_db_torch.detach().numpy())
    print("DX: ", dL_dX, dL_dX_torch.detach().numpy())

    assert dL_dX.shape == dL_dX_torch.shape, "Backward dX Shapes don't match"
    assert dL_dW.shape == dL_dW_torch.shape, "Backward dW Shapes don't match"
    assert dL_db.shape == dL_db_torch.shape, "Backward db Shapes don't match"

    dx_compare = np.allclose(dL_dX, dL_dX_torch.detach().numpy())
    dw_compare = np.allclose(dL_dW, dL_dW_torch.detach().numpy())
    db_compare = np.allclose(dL_db, dL_db_torch.detach().numpy())

   # return AND of all checks
    assert forward_check, "Forward check failed"
    assert dw_compare, "Backward dW check failed"
    assert db_compare, "Backward db check failed"
    assert dx_compare, "Backward dX check failed"
    return forward_check and dx_compare and dw_compare and db_compare

def test_lin_same():
    out = compare_linear(in_features=128, out_features=128) == True
    assert out, "Basic Dense i=128, o=128 failed"

def test_lin_more():
    out = compare_linear(in_features=64, out_features=128) == True
    assert out, "Basic Dense i=64, o=128 failed"

def test_lin_less():
    out = compare_linear(in_features=128, out_features=64) == True
    assert out, "Basic Dense i=128, o=64 failed"



