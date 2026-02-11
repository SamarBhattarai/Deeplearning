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

def sigmoid_backward(dA, cache):
    Z = cache
    s = sigmoid(Z)
    dZ = dA * s * (1-s)
    return dZ

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0 # Gradient is 0 for negative inputs (Boolean Indexing)
    return dZ

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
    
def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    """
    linear_cache, activation_cache = cache
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    
    # 1. Backward Activation (dZ)
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)

    # 2. Backward Linear (dW, db, dA_prev)
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    """
    grads = {} # Empty dictionary
    L = len(caches) # No of layers
    m = AL.shape[1]
    Y = np.reshape(AL.shape) # after this line, Y is the same shape as AL

    # 1. Initializing the backpropagation (Derivative of Cost wrt AL)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # 2. Lth layer (SIGMOID -> LINEAR) gradients
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L-1)], grads["db" + str(L-1)] = \
    linear_activation_backward(dAL, current_cache, activation = "sigmoid")

    # 3. Loop from l=L-2 to l=0 (RELU -> LINEAR layers)
    for l in reversed(range(1, L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = \
            linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation="relu")
            
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - (learning_rate * grads["dW" + str(l+1)])
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - (learning_rate * grads["db" + str(l+1)])
    return parameters

# THE MAIN FUNCTION
def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    """
    np.random.seed(1)
    costs = []

    # Initializing parameters
    parameters = initialize_parameters(layers_dims)

    # Loop (Gradient Descent)
    for i in  range(num_iterations):
        # Forward Propagation
        AL, caches = L_model_forward(X, parameters)

        # Cost
        cost = compute_cost(AL, Y)
        
        # Backward Propagation
        grads = L_model_backward(AL, Y, caches)

        # Update Parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print cost every 100 iterations
        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")
            costs.append(cost)
            
    return parameters, costs