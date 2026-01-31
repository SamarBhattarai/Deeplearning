import numpy as np

def forward_propagation(theta):
    """ Simple function: J(theta) = theta^3 """
    return theta ** 3

def backward_propagation(theta):
    """ Analytic Gradient: dJ/dtheta = 3 * theta^2 """
    return 3 * (theta ** 2)

def gradient_check(theta, epsilon=1e-7):
    # 1. Get the 'real' gradient from your backprop function
    grad_analytic = backward_propagation(theta)
    
    # 2. Compute the 'approximate' gradient using the two-sided formula
    # J(theta + epsilon)
    J_plus = forward_propagation(theta + epsilon)
    # J(theta - epsilon)
    J_minus = forward_propagation(theta - epsilon)
    
    # Rise / Run
    grad_approx = (J_plus - J_minus) / (2 * epsilon)
    
    # 3. Compare them
    # We use a normalized difference to handle very small or large values
    numerator = np.linalg.norm(grad_analytic - grad_approx)
    denominator = np.linalg.norm(grad_analytic) + np.linalg.norm(grad_approx)
    difference = numerator / denominator
    
    print(f"Theta value: {theta}")
    print(f"Analytic Gradient (Backprop): {grad_analytic:.8f}")
    print(f"Numerical Gradient (Approx):  {grad_approx:.8f}")
    print(f"Difference: {difference}")
    
    if difference < 1e-7:
        print("SUCCESS: The gradients match!")
    else:
        print("FAILURE: Something is wrong with backprop.")

# --- RUN THE TEST ---
gradient_check(theta=3.0)
