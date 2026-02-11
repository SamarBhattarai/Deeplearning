import numpy as np
import matplotlib.pyplot as plt

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def initialize_parameters(layers_dims):
    """
    Initializes parameters for a multi-layer  neural network:
    where layers_dims is a python list with no of nodes in each of the multi-layer.
    """
    np.random.seed(7)
    L = len(layers_dims)
    parameters = {} # Python Dictionary.
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 0.01
        parameters["b" + str(l)] = np.zeros(layers_dims[1], 1)
    
    return parameters

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implementing the forward propagation for the LINEAR->ACTIVATION layer
    """
    # Linear Step
    Z = np.dot(W, A_prev) + b
    linear_cache = (A_prev, W, b)

    # Activation Step
    if activation == "relu":
        A, activation_cache = relu(Z) 
    elif activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
        
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    """
    Implementing forward propagation for the [LINEAR->RELU]*(L-1) -> LINEAR->SIGMOID computation
    """
    caches = []
    A = X
    L = len(parameters) // 2 # Number of layers

    # Implementing [LINEAR -> RELU] for layers 1 to L-1
    for l in range(1, L):
        A_prev = A
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        A, cache = linear_activation_forward(A_prev, W, b, "relu")
        caches.append(cache)

    # Implementing [LINEAR -> SIGMOID] for layer L (Output Layer)
    W = parameters["W" + str(L)]
    b = parameters["b" + str(L)]
    AL, cache = linear_activation_forward(A, W, b, "sigmoid")
    caches.append(cache)

    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1] # No of training examples
    # Standard Cross-Entropy Cost
    cost = -1/m * np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL))
    return np.squeeze(cost)