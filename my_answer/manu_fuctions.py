import numpy as np

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = s * (1 - s)
    return ds

def image2vector(image):
    """
    
    :param image: a numpy array of shape (length, height, depth)
    """
    v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)
    return v

def normalizeRows(x):
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / x_norm
    return x

def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum
    
    return s

def L1(yhat, y):
    loss = np.sum(np.abs(y - yhat))
    return loss

def L2(yhat, y):
    loss = np.dot((y-yhat), (y-yhat).T)
    return loss