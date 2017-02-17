import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
digits = load_digits()
data = scale(digits.data)

def print_digits(images,y,max_n=10):
  # set up the figure size in inches\n",
  fig = plt.figure(figsize=(12, 12))
  fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
  i = 0
  while i < max_n and i < images.shape[0]:
      # plot the images in a matrix of 20x20\n",
      p = fig.add_subplot(20, 20, i + 1, xticks=[], yticks=[])
      p.imshow(images[i], cmap=plt.cm.bone)
      # label the image with the target value\n",
      p.text(0, 14, str(y[i]))
      i = i + 1

#print_digits(digits.images, digits.target, max_n=10)
#plt.show()

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(data, digits.target, digits.images, test_size=0.25, random_state=42)

n_samples, n_features = X_train.shape
n_digits = len(np.unique(y_train))
labels = y_train




