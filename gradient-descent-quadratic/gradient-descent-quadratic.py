def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Initial value of x
    x = x0

    for _ in range(steps):
        # Compute the derivative of the quadratic function f(x) = ax^2 + bx + c
        df_dx = 2 * a * x + b

        # Update x using gradient descent formula
        x = x - lr * df_dx

    return x