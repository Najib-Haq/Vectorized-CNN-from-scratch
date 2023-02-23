import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from model.nn.Conv2d import Conv2D
np.random.seed(42)


def compare_conv2d(input_shape, channel_out, kernel_size, stride, padding):
    bs, channel_in, h, w = input_shape
    X = np.random.rand(bs, channel_in, h, w)
    X_torch = torch.from_numpy(X.copy()); X_torch.requires_grad = True
    h_out = (h + 2 * padding - kernel_size) // stride + 1
    w_out = (w + 2 * padding - kernel_size) // stride + 1
    y = np.random.rand(bs, channel_out, h_out, w_out)

    # get the layers
    conv = Conv2D(in_channels=channel_in, out_channels=channel_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
    conv_torch = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
    conv_torch.weight = nn.Parameter(torch.from_numpy(conv.state_dict["kernels"].copy()))
    conv_torch.bias = nn.Parameter(torch.from_numpy(conv.state_dict["bias"].copy()))

    # forward
    y_pred = conv.forward(X)
    y_pred_torch = conv_torch(X_torch)
    forward_check = np.allclose(y_pred, y_pred_torch.detach().numpy())

    assert y_pred.shape == y_pred_torch.shape, "Forward Shapes don't match"
    
    # backward
    dL_dy = y_pred - y.copy()
    print("dl_dy: ",dL_dy)
    dL_dX = conv.backward(dL_dy)
    dL_dW = conv.grads["kernels"]
    dL_db = conv.grads["bias"]
    
    dL_dy_torch = y_pred_torch - torch.from_numpy(y.copy())
    y_pred_torch.backward(dL_dy_torch)
    dL_dX_torch = X_torch.grad
    dL_dW_torch = conv_torch.weight.grad
    dL_db_torch = conv_torch.bias.grad


    print(dL_dX, dL_dX_torch.detach().numpy())
    # print(dL_dy_2[0, 0, 0, :], dL_dy[0, 0, 0, :], dL_dy_torch[0, 0, 0, :].detach().numpy())
    
    # print("Bias: ", conv2.bias, conv_torch.bias)
    # assert np.allclose(dL_dX_2, dL_dX), "2 -> Backward data don't match"

    assert dL_dX.shape == dL_dX_torch.shape, "Backward dX Shapes don't match"
    assert dL_dW.shape == dL_dW_torch.shape, "Backward dW Shapes don't match"
    assert dL_db.shape == dL_db_torch.shape, "Backward db Shapes don't match"

    # print(dL_dX[1, 0, 0, 0], dL_dX_torch[1, 0, 0, 0].detach().numpy())
    dx_compare = np.allclose(dL_dX, dL_dX_torch.detach().numpy())
    dw_compare = np.allclose(dL_dW, dL_dW_torch.detach().numpy())
    db_compare = np.allclose(dL_db, dL_db_torch.detach().numpy())

    # return AND of all checks
    assert forward_check, "Forward check failed"
    assert dw_compare, "Backward dW check failed"
    assert db_compare, "Backward db check failed"
    assert dx_compare, "Backward dX check failed"
    return forward_check and dx_compare and dw_compare and db_compare


def test_conv1():
    out = compare_conv2d(input_shape=(2, 3, 6, 6), channel_out=6, kernel_size=3, stride=1, padding=0) == True
    assert out, "Basic Conv k=2, s=1, p=0 failed"
    
def test_conv2():
    out = compare_conv2d(input_shape=(2, 3, 6, 6), channel_out=6, kernel_size=3, stride=1, padding=2) == True
    assert out, "Basic Conv k=3, s=1, p=2 failed"

# # TODO: CHECK THESE CASES
def test_conv3():
    out = compare_conv2d(input_shape=(2, 3, 4, 4), channel_out=1, kernel_size=3, stride=2, padding=2) == True
    assert out, "Basic Conv k=3, s=2, p=2 failed"

def test_conv4():
    out = compare_conv2d(input_shape=(1, 1, 4, 4), channel_out=1, kernel_size=3, stride=3, padding=2) == True
    assert out, "Basic Conv k=3, s=3, p=2 failed"


def test_conv5():
    out = compare_conv2d(input_shape=(2, 3, 9, 9), channel_out=5, kernel_size=3, stride=3, padding=2) == True
    assert out, "Basic Conv k=3, s=3, p=2 failed"



def test_conv6():
    out = compare_conv2d(input_shape=(2, 3, 8, 8), channel_out=5, kernel_size=3, stride=3, padding=2) == True
    assert out, "Basic Conv k=3, s=3, p=2 failed"


def test_conv7():
    out = compare_conv2d(input_shape=(2, 3, 10, 10), channel_out=6, kernel_size=3, stride=3, padding=1) == True
    assert out, "Basic Conv k=3, s=3, p=1 failed"

def test_conv8():
    out = compare_conv2d(input_shape=(2, 3, 10, 10), channel_out=6, kernel_size=3, stride=3, padding=2) == True
    assert out, "Basic Conv k=3, s=3, p=2 failed"


def test_conv8():
    out = compare_conv2d(input_shape=(2, 3, 12, 12), channel_out=6, kernel_size=5, stride=3, padding=3) == True
    assert out, "Basic Conv k=3, s=3, p=3 failed"

