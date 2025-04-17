import pickle

import matplotlib.pyplot as plt
import numpy as np

data = np.zeros((10, 10))

with open("mnist.pkl", "rb") as f:
    mnist = pickle.load(f)

x_train, t_train, x_test, t_test = (
    mnist["training_images"],
    mnist["training_labels"],
    mnist["test_images"],
    mnist["test_labels"],
)


img = x_train[0, :].reshape(28, 28)  # First image in the training set.
plt.imshow(img, cmap="gray")
plt.show()  # Show the image
