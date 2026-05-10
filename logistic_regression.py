from matplotlib import image
import numpy as np
import pandas as pd
import struct
import os

def read_ubyte_images(filepath):
  with open(filepath, 'rb') as f:
    magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
    images = np.frombuffer(f.read(), dtype=np.uint8)
    images = images.reshape(num, rows * cols)
  return pd.DataFrame(images)

def read_ubyte_labels(filepath):
  with open(filepath, 'rb') as f:
    magic, num = struct.unpack('>II', f.read(8))
    labels = np.frombuffer(f.read(), dtype=np.uint8)
  return pd.Series(labels, name='label')

images_df = read_ubyte_images(f'{os.path.dirname(__file__)}/mnist/train-images.idx3-ubyte')
labels_s = read_ubyte_labels(f'{os.path.dirname(__file__)}/mnist/train-labels.idx1-ubyte')

images = images_df.to_numpy() / 255.0
labels = labels_s.to_numpy()

class LogisticRegression:
  def __init__(self, X, labels):
    CLASSES = 10

    self.weights = np.random.randn(X.shape[1], CLASSES)
    self.bias = np.random.randn(CLASSES)

    self.X = X
    self.labels = labels

  def _softmax(self, Z):
    Z = Z - np.max(Z, axis=-1, keepdims=True)
    return np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)

  def predict(self, X):
    Y = np.dot(X, self.weights) + self.bias

    P = self._softmax(np.array([Y, np.zeros(Y.shape)]))
    P = P[0]

    return np.argmax(P)

  def train(self):
    EPOCHS = 300
    LEARNING_RATE = 0.0001

    for epoch in range(EPOCHS):
      if epoch % 50 == 0:
        print(f"Epoch {epoch}")

      Z = np.dot(self.X, self.weights) + self.bias
      P = self._softmax(Z)

      one_hot_labels = np.zeros((len(self.labels), 10))

      for i in range(len(self.labels)):
        one_hot_labels[i][self.labels[i]] = 1

      self.weights -= LEARNING_RATE * np.dot(self.X.T, P - one_hot_labels)
      self.bias -= LEARNING_RATE * np.sum(P - one_hot_labels, axis=0)

      print("Accuracy: ", np.mean(np.argmax(P, axis=1) == np.argmax(one_hot_labels, axis=1)))
      
logistic_regression = LogisticRegression(images, labels)
logistic_regression.train()

while True:
  index = int(input("Enter number index in the dataset to test: "))

  if index >= 0 and index < len(images):
    print("Correct label: ", labels[index])
    print("Predicted label: ", logistic_regression.predict(images[index]))
  else:
    print("Invalid index")