import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

dataset_x = np.random.rand(50, 1) * 100
dataset_y = 3.5 * dataset_x + np.random.randn(50, 1) * 30
plt.scatter(dataset_x, dataset_y, color="blue")

plt.ylabel("Y")
plt.xlabel("X")

class LinearRegression:
  EPOCHS = 100
  LEARNING_RATE = 0.0001

  def __init__(self):
    self.weight = np.random.randn()
    self.bias = np.random.randn()


  def predict(self):
    for epoch in range(self.EPOCHS):
      if epoch % 50 == 0:
        print(f"Epoch {epoch}")

      n = len(dataset_x)

      y_pred = dataset_x * self.weight + self.bias
      error = dataset_y - y_pred

      mse_w = -2 / n * np.sum(
        dataset_x * error
      )
      mse_b = -2 / n * np.sum(error)

      self.weight -= self.LEARNING_RATE * mse_w
      self.bias -= self.LEARNING_RATE * mse_b

    plt.plot([x for x in range(0, 100)], [x * self.weight + self.bias for x in range(0, 100)], color="green")
    
LinearRegression().predict()
plt.show()