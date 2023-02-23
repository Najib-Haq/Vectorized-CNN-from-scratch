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
from model.nn.MaxPool2d import MaxPool2D
from model.nn.Flatten import Flatten
from model.nn.Linear import Linear
from model.nn.Activations import ReLU, Softmax
from model.Loss import CrossEntropyLoss

from model.Model import Model

np.random.seed(42)


class torchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(294, 3)
    def forward(self, x):
        out =  self.flat(self.pool(self.relu(self.conv(x))))
        print(out.shape)
        return self.fc1(out)

def build_models(add_softmax=False):
    conv = Conv2D(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0, bias=True)
    relu = ReLU()
    pool = MaxPool2D(kernel_size=2, stride=2)
    flat = Flatten()
    fc1 = Linear(294, 3)

    model_layers=[conv, relu, pool, flat, fc1]
    if add_softmax:
        sm = Softmax()
        model_layers.append(sm)

    model = Model(model_layers=model_layers)
    torch_model = torchModel()

    # load weights
    torch_model.conv.weight = nn.Parameter(torch.from_numpy(conv.state_dict["kernels"].copy()))
    torch_model.conv.bias = nn.Parameter(torch.from_numpy(conv.state_dict["bias"].copy()))
    torch_model.fc1.weight = nn.Parameter(torch.from_numpy(fc1.state_dict["weight"].copy()))
    torch_model.fc1.bias = nn.Parameter(torch.from_numpy(fc1.state_dict["bias"].copy()))

    return model, torch_model


def compare_layers():
    X = np.random.rand(2, 3, 16, 16)
    X_torch = torch.from_numpy(X.copy()); X_torch.requires_grad = True
    y = np.random.rand(2, 3)

    model, torch_model = build_models()

    # forward
    y_pred = model.forward(X)
    y_pred_torch = torch_model(X_torch)
    forward_check = np.allclose(y_pred, y_pred_torch.detach().numpy())

    assert y_pred.shape == y_pred_torch.shape, "Forward Shapes don't match"
    
    # backward
    dL_dy = y_pred - y.copy()
    dL_dX = model.backward(dL_dy)
    
    dL_dy_torch = y_pred_torch - torch.from_numpy(y.copy())
    y_pred_torch.backward(dL_dy_torch)
    dL_dX_torch = X_torch.grad

    assert dL_dX.shape == dL_dX_torch.shape, "Backward dX Shapes don't match"

    dx_compare = np.allclose(dL_dX, dL_dX_torch.detach().numpy())

    # return AND of all checks
    assert forward_check, "Forward check failed"
    assert dx_compare, "Backward dX check failed"
    return forward_check and dx_compare

def compare_layers_withloss():
    X = np.random.rand(2, 3, 16, 16)
    X_torch = torch.from_numpy(X.copy()); X_torch.requires_grad = True
    y = np.random.randint(3, size=(2))

    model, torch_model = build_models(add_softmax=True)

    # forward
    y_pred = model.forward(X)
    y_pred_torch = torch_model(X_torch)
    forward_check = np.allclose(y_pred, Softmax()(y_pred_torch.detach().numpy()))

    assert y_pred.shape == y_pred_torch.shape, "Forward Shapes don't match"
    
    # calculate loss
    label = np.zeros((2, 3))
    label[np.arange(2), y] = 1
    loss = CrossEntropyLoss()
    loss_torch = nn.CrossEntropyLoss()
    loss_value = loss(y_pred, label)
    loss_value_torch = loss_torch(y_pred_torch, torch.from_numpy(y.copy()).long())
    assert loss_value.shape == loss_value_torch.shape, "Loss Shapes don't match"
    
    loss_check = np.allclose(loss_value, loss_value_torch.detach().numpy())

    # backward    
    dL_dy = loss.get_grad_wrt_softmax(y_pred, label)
    dL_dX = model.backward(dL_dy)
    
    loss_value_torch.backward()
    dL_dX_torch = X_torch.grad

    assert dL_dX.shape == dL_dX_torch.shape, "Backward dX Shapes don't match"

    dx_compare = np.allclose(dL_dX, dL_dX_torch.detach().numpy())

    # return AND of all checks
    assert forward_check, "Forward check failed"
    assert loss_check, "Loss check failed"
    assert dx_compare, "Backward dX check failed"
    return forward_check and loss_check and dx_compare


def test_conv1():
    out = compare_layers() == True
    assert out, "Basic Layers Backprop failed"


# # TODO: this has problems -> whether the dL_dX and the torch version is correct?
# def test_conv1_withloss():
#     out = compare_layers_withloss() == True
#     assert out, "Basic Layers Backprop failed"