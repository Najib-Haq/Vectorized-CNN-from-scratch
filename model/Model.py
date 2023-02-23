import numpy as np
import pickle

from model.nn.Conv2d import *
from model.nn.MaxPool2d import *
from model.nn.Linear import *
from model.nn.Flatten import *
from model.nn.Activations import *

class Model:
    def __init__(self, config=False, model_layers=[]):
        self.config = config['model'] if config else False
        if self.config: self.layers = self.create_model()
        else: self.layers = model_layers

        if config:
            print("Testing model shapes with random X: ")
            X = np.random.randn(1, 3, config['augment']['img_shape'][0], config['augment']['img_shape'][1])
            self.forward(X, debug=True)
            print('#'*50)

    def forward(self, X, debug=False):
        if debug: print("Input X: -> \t\t", X.shape)
        for i, layer in enumerate(self.layers):
            X = layer(X)
            if debug: print(f"Layer {i}: {layer.name} ->\t", X.shape)
        return X

    def backward(self, dL_dy):
        # print("INPUT BACKWARD: ",dL_dy.shape)
        for layer in reversed(self.layers):
            dL_dy = layer.backward(dL_dy)
        return dL_dy

    def __call__(self, X):
        return self.forward(X)

    def __str__(self):
        print_data = "MODEL LAYERS & PARAMETERS: \n"
        for i, layer in enumerate(self.layers):
            print_data += f"Layer {i}: " + str(layer) + "\n"
        return print_data

    def create_model(self):
        model = []

        for layer in self.config:
            name = layer[0]
            params = layer[1]

            if name == "Conv2D":
                model.append(Conv2D(*params))
            elif name == "MaxPool2D":
                model.append(MaxPool2D(*params))
            elif name == "Flatten":
                model.append(Flatten())
            elif name == "Linear":
                model.append(Linear(None, params[0], lazy_init=True))
            elif name == "ReLU":
                model.append(ReLU())
            elif name == "Softmax":
                model.append(Softmax())
        return model


    def save_model(self, path, epoch, wandb_id, cur_lr):
        params = []
        for layer in self.layers:
            params.append(layer.state_dict)

        save_data = {
            'state_dict': params,
            'epoch': epoch,
            'wandb_id': wandb_id,
            'lr': cur_lr,
        }

        with open(path, "wb") as f:
            pickle.dump(save_data, f)

    def load_model(self, path, pretrained=False):
        with open(path, "rb") as f:
            params = pickle.load(f)
            if isinstance(params, dict): 
                print("Loading Epochs: ", params['epoch'], " LR: ", params['lr'], " W&B ID: ", params['wandb_id'])
                params = params['state_dict']
        if pretrained:
            if params[0]['kernels'].shape != self.layers[0].state_dict['kernels'].shape:
                print("Pretrained model input shape does not match model shape")
                # pretrained on grayscale images and train on rgb ones
                b = np.zeros(self.layers[0].state_dict['kernels'].shape)
                b[:, 0, :, :] = params[0]['kernels'][:, 0, :, :]
                b[:, 1, :, :] = params[0]['kernels'][:, 0, :, :]
                b[:, 2, :, :] = params[0]['kernels'][:, 0, :, :]
                print(b.shape, params[0]['kernels'].shape)
                params[0]['kernels'] = b
        for i, layer in enumerate(self.layers):
            layer.state_dict = params[i]
        print("Successfully loaded from " + path)

    # makes model trainable -> stores cache
    def train(self):
        for layer in self.layers:
            layer.trainable = True

    # makes model untrainable -> doesnt store cache
    def eval(self):
        for layer in self.layers:
            layer.trainable = False
        