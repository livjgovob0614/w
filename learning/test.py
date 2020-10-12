import tensorflow as tf
import numpy as np
print(tf.__version__)


x = np.array([1, 2, 3])
print (x)
idx = np.arange(x.shape[0])
np.random.shuffle(idx)
x = x[idx]
print (x)


N = 5

li = []
a = np.array([0]*(N-1)+[1]+[0]*N)
li.append(a)
li.append(a)
li = np.array(li)
print (li)
print (li.shape)

print (a.argmax(1))
