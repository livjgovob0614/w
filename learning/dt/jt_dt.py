import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
import mglearn
from sklearn import svm
from sklearn.datasets.samples_generator import make_blobs
from sklearn.tree import export_graphviz
import graphviz
import pydot
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score

# gen training data #
cur_path = os.getcwd()
N = str(4500)

f = open(cur_path+'/../../input/large/jt'+N+'_hex.list', 'r')
hexcode = f.readlines()
f.close()
for i in range(0,len(hexcode)):
    hexcode[i] = hexcode[i].split(' ')

f = open(cur_path+'/../../input/large/jt'+N+'_hex.anal', 'r')
ff = f.readlines()
f.close()    
hexdata = []
for i in range(0,len(ff)):
    spl = ff[i].split(' ')
    hexdata.append(int(spl[2]) - 1)
x_train = np.array(hexcode)
y_train = np.array(hexdata)
ff = np.array(ff)

#  code:0 / data:1
c_sum, d_sum, t_sum = 0.0, 0.0, 0.0
step =1 
"""
y_code = [1]*hexcode.shape[0]
y_data = [0]*hexdata.shape[0]
x_train = (np.concatenate((hexcode,hexdata ), axis=0))
y_train = (np.concatenate((y_code,y_data ), axis=0))
"""
print ("x_train, y_train shape:")
print (x_train.shape)
print (y_train.shape)
print (y_train)
sys.exit()


max_features = 1
for i in range(0,len(x_train)):
    for j in range(0,len(x_train[i])):
      #x_train[i][j] = x_train[i][j].astype(np.int64)
        x_train[i][j] = int ("0x"+x_train[i][j], 16)


idx = np.arange(x_train.shape[0])
np.random.shuffle(idx)
ff = ff[idx]
ff = ff[:1000]
x_train = x_train[idx]
y_train = y_train[idx]
x_test = x_train[:1000]
x_train = x_train[1000:]
y_test = y_train[:1000]
y_train = y_train[1000:]
print (x_train.shape)
print (x_test.shape)

text_max_words = 4500
x_train = sequence.pad_sequences(x_train, maxlen=text_max_words)
x_test = sequence.pad_sequences(x_test, maxlen=text_max_words)
print (x_train.shape)
print (x_test.shape)


rf = DecisionTreeClassifier(min_samples_split=30, max_depth=40)
rf.fit(x_train, y_train)
rf.predict_proba(x_test)
print(rf.score(x_test, y_test))
cls = []
for i in range(255):
  cls.append(str(int("0x"+str(i), 16)))
cls = np.array(cls)
#export_graphviz(rf, out_file="tree.dot", class_names=["code", "data"],
export_graphviz(rf, out_file="tree.dot", class_names=cls,
                feature_names=None, impurity=False, filled=True)

(graph,) = pydot.graph_from_dot_file('tree.dot', encoding='utf8')
graph.write_png('tree.png')
