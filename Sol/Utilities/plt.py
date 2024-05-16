import numpy as np
import matplotlib.pyplot as plt
import torch as th


def activation_functions():
    # Logistic function
    def logistic_function(x):
        return 1 / (1 + np.exp(-x))

    # tanh
    def tanh_function(x):
        return np.tanh(x)

    # ReLU
    def relu_function(x):
        return np.maximum(0, x)

    def leaky_relu_function(x):
        return th.nn.functional.leaky_relu(x, 0.1)

    def leaky_relu_function(x):
        return np.maximum(0.1 * x, x)

    x_values = np.linspace(-5, 5, 100)

    logistic_values = logistic_function(x_values)
    tanh_values = tanh_function(x_values)
    relu_values = relu_function(x_values)

    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.plot(x_values, logistic_values, label='Logistic')
    plt.title('Logisztikus')  # Logistic function
    plt.legend()

    plt.subplot(132)
    plt.plot(x_values, tanh_values, label='Tanh')
    plt.title('Hiperbolikus tangens')  #Hyperbolic tangent function
    plt.legend()

    plt.subplot(133)
    plt.plot(x_values, relu_values, label='ReLU')
    plt.title('ReLU')  # ReLU functions
    plt.legend()

    plt.subplot(133)
    plt.plot(x_values, leaky_relu_function(x_values), label='Leaky ReLU')
    plt.legend()

    plt.tight_layout()
    plt.show()


def reward_func():
    # Define the function g(x)
    def g(x):
        return 3 * np.exp(-4 * np.abs(x))

    # Create an array of x values
    x_values = np.linspace(-2, 2, 400)

    # Calculate the y values
    y_values = g(x_values)

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label=r"$g(x) = 3 e^{-4 |x|}$", color='blue')
    plt.xlabel('Distance to target')
    plt.ylabel('Reward')
    plt.title('Exp Reward based on distance $g(x) = 3 e^{-4 |x|}$')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    activation_functions()
    reward_func()

    # Define the function
    # def reciprocal(x):
    #     return 1 / np.abs(x)
    #
    #
    # # Generate x values
    # x_values = np.linspace(-1, 1, 100)  # Adjust the range as needed
    #
    # # Calculate corresponding y values
    # y_values = reciprocal(x_values)
    #
    # # Plot the function
    # plt.plot(x_values, y_values, label=r'$\frac{1}{|x|}$')
    # plt.axhline(0, color='black', linewidth=0.5)
    # plt.axvline(0, color='black', linewidth=0.5)
    # plt.title('Plot of $\\frac{1}{|x|}$')
    # plt.xlabel('Distance to Target')
    # plt.ylabel('Reward')
    # plt.legend()
    # plt.grid(color='gray', linestyle='--', linewidth=0.5)
    # plt.show()
    #
    #
    # # Define the reward function
    # def calculate_reward(distance_to_target):
    #     return np.exp(-distance_to_target * 5) * 50
    #
    #
    # # Generate a range of distance values
    # distance_values = np.linspace(0, 2, 100)
    #
    # # Calculate corresponding rewards
    # reward_values = calculate_reward(distance_values)
    #
    # # Plot the reward function
    # plt.plot(distance_values, reward_values, label='Reward Function')
    # plt.title('Reward Function Plot')
    # plt.xlabel('Distance to Target')
    # plt.ylabel('Reward')
    # plt.legend()
    # plt.grid(color='gray', linestyle='--', linewidth=0.5)
    # plt.show()
