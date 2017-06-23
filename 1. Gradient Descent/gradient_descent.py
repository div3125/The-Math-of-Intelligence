# Import required libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def data_analysis():
    """
    Function to perform initial exploratory analysis
    """

    # Read first 100 rows of data from CSV file
    menu = pd.read_csv("menu.csv", nrows=100)
    # Display relevant features
    print(menu[["Category", "Item", "Serving Size", "Calories"]].head())
    # Extract gram values from Serving Size
    menu["Serving Size (grams)"] = menu["Serving Size"].apply(lambda s: int(s.split(" ")[2][1:]))

    # Extract feature and target
    X = np.array(menu["Serving Size (grams)"])
    y = np.array(menu["Calories"])

    # Plot the data
    plt.scatter(X, y)
    plt.xlabel("Serving Size (grams)")
    plt.ylabel("Calories")
    plt.title("Data")
    plt.show()

    return X, y


def normalize(X, y):
    """
    Function to normalize and then plot
    the normalized data
    """

    X_norm = (X - np.mean(X)) / np.std(X)
    y_norm = (y - np.mean(y)) / np.std(y)

    # Plot the normalized data
    plt.scatter(X_norm, y_norm)
    plt.title("Normalized Data")
    plt.show()

    return X_norm, y_norm


def compute_mse(y_true, y_pred):
    """
    Function to compute Mean Squared Error (MSE) 
    """

    return np.mean(np.square((y_true - y_pred)))


def print_stats(n_iter, m, c, error):
    """
    Function to print model statistics 
    """

    print("After {0} iterations, Error = {1}, m = {2} and c = {3}".format(n_iter, error, m, c))


def compute_parameter_updates(X, y, m, c, learning_rate):
    """
    Function to compute gradients at each step 
    and update the parameters
    """

    N = float(len(X))

    # Use vectorized operation of matrix multiplication
    # to calculate gradients without for loops
    m_gradient = np.sum((-2 / N) * X * (y - (m * X + c)))
    c_gradient = np.sum((-2 / N) * (y - (m * X + c)))

    # Update the parameters
    m_updated = m - (learning_rate * m_gradient)
    c_updated = c - (learning_rate * c_gradient)

    return m_updated, c_updated


def gradient_descent():
    """
    Gradient descent runner 
    """

    # Hyperparameters
    m = -1  # initial guess of slope
    c = -1  # initial guess of y-intercept
    alpha = 0.001  # learning rate
    n_iter = 1000  # Number of iterations

    # List to store MSE at each step of training
    errors = []

    # Get data
    X, y = data_analysis()
    # Normalize data
    X_norm, y_norm = normalize(X, y)

    # Initial guess
    y_pred = m * X_norm + c

    # Plot the initial guess
    plt.scatter(X_norm, y_norm)
    plt.plot(X_norm, y_pred, c="r")
    plt.title("Intial Guess")
    plt.show()

    # Run gradient descent
    for i in range(1, n_iter + 1):
        # Calculate updated parameters
        m, c = compute_parameter_updates(X_norm, y_norm, m, c, alpha)
        # Save error for each step
        errors.append(compute_mse(y_norm, (m * X_norm + c)))
        # Print model stats at every 100 iterations
        if not i % 100:
            print_stats(i, m, c, errors[i - 1])

    # Predictions after training
    y_pred = (m * X_norm) + c

    # Plot the trained model
    plt.scatter(X_norm, y_norm)
    plt.plot(X_norm, y_pred, c="r")
    plt.title("Model after training")
    plt.show()

    # Plot the errors
    plt.plot(range(1, n_iter + 1), errors)
    plt.xlabel("No. of iterations")
    plt.ylabel("Error")
    plt.title("Error vs. Training")
    plt.show()


if __name__ == '__main__':
    gradient_descent()
