import pickle

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from tqdm import tqdm

# Runtime Variables
training_depth = 300  # how many images to train on
m2_size = 128
learning_rate = 0.3
epochs = 20
batch_size = 64

# Code
rng = np.random.default_rng(88296743)
with open("mnist.pkl", "rb") as f:
    mnist = pickle.load(f)

x_train, t_train, x_test, t_test = (
    mnist["training_images"],
    mnist["training_labels"],
    mnist["test_images"],
    mnist["test_labels"],
)

# Simplify to float from 0 -> 1
x_train, t_train, x_test = (
    x_train[:training_depth] / 255,
    t_train[:training_depth],
    x_test[:training_depth] / 255,
)

t_train, t_test = (
    (t_train[..., None] == np.arange(10)[None]).astype(np.float64),
    (t_test[..., None] == np.arange(10)[None]).astype(np.float64),
)


@njit(cache=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@njit(cache=True)
def sigmoid_derivative(x):
    return x * (1 - x)


@njit(cache=True)
def forward_pass(input_data, w1, w2):
    layer1 = sigmoid(input_data.dot(w1))
    return layer1, sigmoid(layer1.dot(w2))


@njit(cache=True)
def backward_pass(input_data, layer1, layer2, target, weights_1, weights_2):
    # Backpropagation
    layer2_error = target - layer2
    layer2_delta = layer2_error * sigmoid_derivative(layer2)

    layer1_delta = layer2_delta.dot(weights_2.T) * sigmoid_derivative(layer1)

    # Update weights
    weights_2 += layer1.T.dot(layer2_delta.reshape(1, -1)) * learning_rate
    weights_1 += input_data.T.dot(layer1_delta) * learning_rate

    return np.mean(layer2_error**2)


def train(x_data, t_data):
    weights_1 = 0.2 * rng.random((784, m2_size)) - 0.1  # Translate the 784 - > 100
    weights_2 = 0.2 * rng.random((m2_size, 10)) - 0.1  # Translate the 100 -> 10

    num_samples = x_data.shape[0]
    costs = []

    for epoch in tqdm(range(epochs)):
        indices = np.arange(num_samples)
        rng.shuffle(indices)

        total_cost = 0

        for start_idx in range(0, num_samples, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            x_batch = x_data[batch_indices]
            t_batch = t_data[batch_indices]

            batch_cost = 0
            for i in range(len(x_batch)):
                layer1, layer2 = forward_pass(x_batch[i], weights_1, weights_2)

                batch_cost += backward_pass(
                    x_batch[i].reshape(1, -1),
                    layer1.reshape(1, -1),
                    layer2,
                    t_batch[i],
                    weights_1,
                    weights_2,
                )

            total_cost += batch_cost / len(x_batch)

        costs.append(total_cost / (num_samples / batch_size))

    return costs, weights_1, weights_2


# Train the network
costs, weights_1, weights_2 = train(x_train, t_train)

# Evaluate accuracy
error = 0
for i in range(x_test.shape[0]):
    _, result = forward_pass(x_test[i], weights_1, weights_2)
    if np.argmax(result) != np.argmax(t_test[i]):
        error += 1

print("Error:", np.round(100 * error / x_test.shape[0], 4), "%")

# Plot the cost history
if __name__ == "__main__":
    plt.figure(figsize=(10, 5))
    plt.plot(costs)
    plt.title("Training Cost")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.show()

np.save("weights_1.npy", weights_1)
np.save("weights_2.npy", weights_2)
