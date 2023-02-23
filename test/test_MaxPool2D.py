import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from model.nn.MaxPool2d import MaxPool2D
np.random.seed(42)

def compare_maxpool(channels, kernel_size, stride, h=6, w=6):
    X = np.random.rand(2, channels, h, w)
    print("Input: ", X)
    X_torch = torch.from_numpy(X.copy()); X_torch.requires_grad = True
    h_out = (h - kernel_size) // stride + 1
    w_out = (w - kernel_size) // stride + 1
    y = np.random.rand(2, channels, h_out, w_out)

    # get the layers
    mp = MaxPool2D(kernel_size=kernel_size, stride=stride)
    mp_torch = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    # forward
    y_pred = mp.forward(X)
    y_pred_torch = mp_torch(X_torch)
    assert y_pred.shape == y_pred_torch.shape, "Forward Shapes don't match"
    forward_check = np.allclose(y_pred, y_pred_torch.detach().numpy())
    
    # backward
    dL_dy = np.ones_like(y_pred) #y_pred - y.copy()
    dL_dX = mp.backward(dL_dy)
    dL_dy_torch = torch.ones_like(y_pred_torch) #y_pred_torch - torch.from_numpy(y.copy())
    # print("dL/dy: ", dL_dy)
    y_pred_torch.backward(dL_dy_torch)

    dL_dX_torch = X_torch.grad
    assert dL_dX.shape == dL_dX_torch.shape, "Backward dX Shapes don't match"
    dx_compare = np.allclose(dL_dX, dL_dX_torch.detach().numpy())

    # return AND of all checks
    print(dL_dX)
    print("TORCH: ", dL_dX_torch.detach().numpy())
    assert forward_check, "Forward check failed"
    assert dx_compare, "Backward dX check failed"
    return forward_check and dx_compare

def test_mp_k2s2():
    out = compare_maxpool(channels=3, kernel_size=2, stride=2) == True
    assert out, "Basic MaxPool k=2, s=2 failed"

def test_mp_k2s1():
    out = compare_maxpool(channels=3, kernel_size=2, stride=1) == True
    assert out, "Basic MaxPool k=2, s=1 failed"
    
def test_mp_k3s1():
    out = compare_maxpool(channels=3, kernel_size=3, stride=1) == True
    assert out, "Basic MaxPool k=3, s=1 failed"
    
def test_mp_k3s2():
    out = compare_maxpool(channels=3, kernel_size=3, stride=2) == True
    assert out, "Basic MaxPool k=3, s=2 failed"

def test_mp_k3s2_h7():
    out = compare_maxpool(channels=3, kernel_size=3, stride=2, h=7, w=7) == True
    assert out, "Basic MaxPool k=3, s=2 failed"

def test_mp_k3s2_h8():
    out = compare_maxpool(channels=3, kernel_size=3, stride=2, h=8, w=8) == True
    assert out, "Basic MaxPool k=3, s=2 failed"

def test_mp_k3s2_h9():
    out = compare_maxpool(channels=3, kernel_size=3, stride=2, h=9, w=9) == True
    assert out, "Basic MaxPool k=3, s=2 failed"

def test_mp_k3s3_h9():
    out = compare_maxpool(channels=3, kernel_size=3, stride=3, h=9, w=9) == True
    assert out, "Basic MaxPool k=3, s=3 failed"

def test_mp_k3s3_h10():
    out = compare_maxpool(channels=3, kernel_size=3, stride=3, h=10, w=10) == True
    assert out, "Basic MaxPool k=3, s=3 failed"

def test_mp_k3s3_h11():
    out = compare_maxpool(channels=3, kernel_size=3, stride=3, h=11, w=11) == True
    assert out, "Basic MaxPool k=3, s=3 failed"

def test_mp_k3s3_h12():
    out = compare_maxpool(channels=3, kernel_size=3, stride=3, h=12, w=12) == True
    assert out, "Basic MaxPool k=3, s=3 failed"
