import numpy as np

from model.nn.Base import Base

class Linear(Base):
    def __init__(self, in_features, out_features, lazy_init=False, bias=True):
        super().__init__()
        self.name = "Linear"
        self.params = {
            "in_features": None if lazy_init else in_features,
            "out_features": out_features,
            "bias": bias,
        }
        if not lazy_init: self.state_dict = self.initialize_parameters()
        else: self.state_dict = {"weight": None}
        self.cache = {}

    def initialize_parameters(self):
        # xavier initialization
        # https://paperswithcode.com/method/he-initialization
        std = np.sqrt(2 / self.params["in_features"])
        weights = np.random.randn(self.params["out_features"], self.params["in_features"]) * std
        if self.params["bias"]:
            bias = np.zeros(self.params["out_features"])
            return {"weight": weights, "bias": bias}
        return {"weight": weights}

    def forward(self, X):
        '''
        X shape should be (N, in_features)
        W shape is (out_features, in_features)
        so the output shape is (N, out_features)
        '''
        if self.state_dict["weight"] is None: 
            self.params["in_features"] = X.shape[1]
            self.state_dict = self.initialize_parameters()
        if self.trainable: self.cache['X'] = X
        output = np.dot(X, self.state_dict["weight"].T)
        if self.params["bias"]: output += self.state_dict["bias"]
        return output

    def backward(self, dL_dy):
        '''
        dL_dy = gradient of the cost with respect to the output of the linear layer -> (bs, out_features)
        '''
        # gradient of the cost with respect to the weights
        dL_dW = np.dot(dL_dy.T, self.cache['X'])  # (out_features, bs) * (bs, in_features) -> (out_features, in_features)
        # gradient of the cost with respect to the input
        dL_dX = np.dot(dL_dy, self.state_dict["weight"]) # (bs, out_features) * (out_features, in_features) -> (bs, in_features)
        # gradient of the cost with respect to the bias
        if self.params["bias"]: dL_db = np.sum(dL_dy, axis=0)

        # update weights and bias
        self.grads = {"weight": dL_dW} 
        if self.params["bias"]: self.grads["bias"] = dL_db
        
        return dL_dX