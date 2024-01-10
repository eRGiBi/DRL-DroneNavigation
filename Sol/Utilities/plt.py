import numpy as np
import matplotlib.pyplot as plt

def activation_functions():
    # Logisztikus függvény
    def logistic_function(x):
        return 1 / (1 + np.exp(-x))

    # Tangens hiperbolikus függvény (tanh)
    def tanh_function(x):
        return np.tanh(x)

    # ReLU függvény
    def relu_function(x):
        return np.maximum(0, x)

    # Tartomány létrehozása
    x_values = np.linspace(-5, 5, 100)

    # Függvények kiértékelése
    logistic_values = logistic_function(x_values)
    tanh_values = tanh_function(x_values)
    relu_values = relu_function(x_values)

    # Függvények plotolása
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.plot(x_values, logistic_values, label='Logisztikus')
    plt.title('Logisztikus függvény')
    plt.legend()

    plt.subplot(132)
    plt.plot(x_values, tanh_values, label='Tanh')
    plt.title('Tangens hiperbolikus függvény')
    plt.legend()

    plt.subplot(133)
    plt.plot(x_values, relu_values, label='ReLU')
    plt.title('ReLU függvény')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
   # Define the function
    def reciprocal(x):
        return 1 / np.abs(x)

    # Generate x values
    x_values = np.linspace(-1, 1, 100)  # Adjust the range as needed

    # Calculate corresponding y values
    y_values = reciprocal(x_values)

    # Plot the function
    plt.plot(x_values, y_values, label=r'$\frac{1}{|x|}$')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.title('Plot of $\\frac{1}{|x|}$')
    plt.xlabel('Distance to Target')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.show()
    
    
    # Define the reward function
    def calculate_reward(distance_to_target):
        return np.exp(-distance_to_target * 5) * 50

    # Generate a range of distance values
    distance_values = np.linspace(0, 2, 100)

    # Calculate corresponding rewards
    reward_values = calculate_reward(distance_values)

    # Plot the reward function
    plt.plot(distance_values, reward_values, label='Reward Function')
    plt.title('Reward Function Plot')
    plt.xlabel('Distance to Target')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.show()
