import tensorflow as tf
import numpy as np
print(tf.__version__)


x = np.array([1, 2, 3])
print (x)
idx = np.arange(x.shape[0])
np.random.shuffle(idx)
x = x[idx]
print (x)
