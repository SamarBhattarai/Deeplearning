import numpy as np

#  implementation using a basic Logistic Regression unit. This keeps the math clean while demonstrating exactly 
# how to flatten the Weights Matrix (W) and Bias Vector (b) into a single parameter vector (theta) for the check.

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_propagation(X, Y, W, b):
    m = X.shape[1]

    Z = np.dot(W.T, X) + b
    A = sigmoid(Z)

    cost = (-1/m) * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))
    return A, cost

def backward_propagation(X, Y, A, W, b):
    m = X.shape[1]
    dZ = A - Y
    dW = (1/m) * np.dot(dZ, X.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    
    return dW, db

# HELPER FUNCTIONS FOR FLATTENING AND RESHAPING 

def dictionary_to_vector(W, b):
    """
    Flattens W (matrix) and b (vector) into a single column vector theta.
    """
    # W.reshape(-1, 1) makes it a column vector
    # b.reshape(-1, 1) makes it a column vector
    # np.concatenate joins them vertically
    theta = np.concatenate((W.reshape(-1, 1), b.reshape(-1, 1)))
    return theta

def vector_to_dictionary(theta, W_shape, b_shape):
    """Restores W and b from flat theta vector. """

    W_size = W_shape[0] * W_shape[1]
    W_flat = theta[:W_size]
    W_restored = W_flat.reshape(W_shape)

    b_flat = theta[W_size:]
    b_restored = b_flat.reshape(b_shape)

    return W_restored, b_restored

# The core logic, that is gradient checking

def gradient_checking(X, Y, W, b, dW, db, epsilon = 1e-7):
    # Converting dictionary(W, b) to vector(theta)
    theta = dictionary_to_vector(W, b)
    # Converting dictionary(dW, db) to vector(d_theta)
    d_theta_analytic = dictionary_to_vector(dW, db)

    # Initializing Variables
    num_of_parameters = theta.shape[0]
    d_theta_numerical = np.zeros((num_of_parameters, 1))

    # Using Loop on every single parameter in theta
    for i in range(num_of_parameters):

        # Saving the original value
        theta_original = theta[i][0]

      # --- 1. Compute J(theta + epsilon) ---
        theta[i][0] = theta_original + epsilon
        # We must "un-flatten" theta to run forward prop
        W_plus, b_plus = vector_to_dictionary(theta, W.shape, b.shape)
        _, J_plus = forward_propagation(X, Y, W_plus, b_plus)
        
        # --- 2. Compute J(theta - epsilon) ---
        theta[i][0] = theta_original - epsilon
        W_minus, b_minus = vector_to_dictionary(theta, W.shape, b.shape)
        _, J_minus = forward_propagation(X, Y, W_minus, b_minus)

        # --- 3. Compute Numerical Gradient ---
        d_theta_numerical[i] = (J_plus - J_minus) / (2 * epsilon)
        
        # Restore original value for the next iteration
        theta[i][0] = theta_original

    # Step D: Compare Analytic vs Numerical
    numerator = np.linalg.norm(d_theta_analytic - d_theta_numerical)
    denominator = np.linalg.norm(d_theta_analytic) + np.linalg.norm(d_theta_numerical)
    difference = numerator / denominator

    print(f"Difference: {difference}")
    if difference < 1e-7:
        print("Gradient Check: PASSED")
    else:
        print("Gradient Check: FAILED")

# VALIDATING WITH THE HELP OF EXAMPLE #

X = np.array([[7, 8, 9],
            [0, 6, 5]])
# We can see above that there are two features and m = 3 training examples.
Y = np.array([[0, 1, 0]])

# Initializing Weight Parameters

np.random.seed(7)
# W = np.random.randn(2, 3)
# b = np.random.randn(3, 1)
W = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[1, 2, 3]])
# 3. Get Analytic Gradients
A, cost = forward_propagation(X, Y, W, b)
print(A, cost)
dW, db = backward_propagation(X, Y, A, W, b)
print(dW, db)

# 4. Validate with Gradient Checking
gradient_checking(X, Y, W, b, dW, db)
