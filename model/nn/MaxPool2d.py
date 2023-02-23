import numpy as np

from model.nn.Base import Base

class MaxPool2D(Base):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.name = "MaxPool2D"
        self.params = {
            "kernel_size": kernel_size,
            "stride": stride,
        }
        self.cache = {}
        self.same_kernel_stride = kernel_size == stride  # fully vectorize when kernel_size == stride

    def forward(self, X):
        N, C, H, W = X.shape
        if self.trainable:
            self.cache['X_shape'] = X.shape
            self.cache['X_strides'] = X.strides
        kernel_size, stride = self.params["kernel_size"], self.params["stride"]

        # output shape
        H_out = (H - kernel_size) // stride + 1
        W_out = (W - kernel_size) // stride + 1

        # get kernel strided X
        N_strides, C_out_strides, H_strides, W_strides = X.strides
        strided_X = np.lib.stride_tricks.as_strided(
            X,
            shape=(N, C, H_out, W_out, kernel_size, kernel_size),
            strides=(N_strides, C_out_strides, stride * H_strides, stride * W_strides, H_strides, W_strides)
        )

        # max pooling
        output = np.max(strided_X, axis=(4, 5))
        if self.trainable: 
            if self.same_kernel_stride: 
                maxes_reshaped_to_original_window = output.repeat(stride, axis=-2).repeat(stride, axis=-1)
                # pad incase of odd shape
                pad_h = H - maxes_reshaped_to_original_window.shape[-2]
                pad_w = W - maxes_reshaped_to_original_window.shape[-1]
                maxes_reshaped_to_original_window = np.pad(maxes_reshaped_to_original_window, ((0,0), (0,0), (0,pad_h), (0,pad_w)))
                self.cache['mask'] = np.equal(X, maxes_reshaped_to_original_window)
            else: self.cache['strided_X'] = strided_X
        return output

    def fully_vectorized_backward(self, dL_dy):
        # not that much increase :/
        # https://stackoverflow.com/questions/61954727/max-pooling-backpropagation-using-numpy
        stride = self.params["stride"]
        N, C, H, W = self.cache['X_shape']
        dL_dy_reshaped_to_original_window = dL_dy.repeat(stride, axis=-2).repeat(stride, axis=-1)
        
        # pad incase of odd shape
        pad_h = H - dL_dy_reshaped_to_original_window.shape[-2]
        pad_w = W - dL_dy_reshaped_to_original_window.shape[-1]
        dL_dy_reshaped_to_original_window = np.pad(dL_dy_reshaped_to_original_window, ((0,0), (0,0), (0,pad_h), (0,pad_w)))
        
        dL_dy_reshaped_to_original_window = np.multiply(dL_dy_reshaped_to_original_window, self.cache['mask'])
        return dL_dy_reshaped_to_original_window

    def partially_vectorized_backward(self, dL_dy):
        N, C, H_out, W_out = dL_dy.shape
        kernel_size, stride = self.params["kernel_size"], self.params["stride"]

        # get cached strided_X
        strided_X = self.cache['strided_X']

        reshaped_strided_X = strided_X.reshape((N, C, H_out, W_out, -1)) # need to do this as cannot get max from multiple axis
        argmaxes = reshaped_strided_X.argmax(axis=-1)
        a1, a2, a3, a4 = np.indices((N, C, H_out, W_out)) # indices of axies
        argmaxes_indices = (a1, a2, a3, a4, argmaxes)
        
        # set to 1 and then multiply with gradient
        strided_X_maxes = np.zeros_like(reshaped_strided_X)
        strided_X_maxes[argmaxes_indices] = 1
        strided_X_maxes *= dL_dy[..., None]

        # reshape to original shape
        strided_X_maxes = strided_X_maxes.reshape(strided_X.shape)
        # print(strided_X_maxes)

        dL_dX = np.zeros(self.cache['X_shape'])
        
        for i in range(H_out):
            for j in range(W_out):
                dL_dX[:,:,i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size] += strided_X_maxes[:,:,i,j]
        
        return dL_dX


    def backward(self, dL_dy):
        if self.same_kernel_stride:
            return self.fully_vectorized_backward(dL_dy)
        else:
            return self.partially_vectorized_backward(dL_dy)

        

if __name__ == "__main__":
    np.random.seed(142)
    mp = MaxPool2D(2, 1)
    X = np.random.rand(1,1,3,3)
    out = mp.forward(X)
    print("X: ", X)
    print("Forward: ", out.shape)
    dL_dy = np.random.rand(1,1,2,2)
    print("dL_dy: ", dL_dy)
    dL_dX = mp.backward(dL_dy)
    print("Backward: ", dL_dX.shape)
    print("dL_dX: ", dL_dX)