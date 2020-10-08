import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import mglearn
from sklearn import svm
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score

# gen training data #
cur_path = os.getcwd()
N = str(20)

f = open(cur_path+'/../../input/FandLP/large/f'+N+'_hex.list', 'r')
hexcode = f.readlines()
f.close()
for i in range(0,len(hexcode)):
    hexcode[i] = hexcode[i].split(' ')

f = open(cur_path+'/../../post_recent/output/df_'+N+'.list', 'r')
#f = open(cur_path+'/../../post_recent/output/df_1007test_'+N+'.list', 'r')
hexdata = f.readlines()
f.close()    
for i in range(0,len(hexdata)):
    hexdata[i] = hexdata[i].split(' ')
hexcode = np.array(hexcode)
hexdata = np.array(hexdata)
print ("code, data shape:")
print (hexcode.shape)
print (hexdata.shape)

#  code:0 / data:1
c_sum, d_sum, t_sum = 0.0, 0.0, 0.0
step =1 
y_code = [1]*hexcode.shape[0]
y_data = [0]*hexdata.shape[0]
x_train = (np.concatenate((hexcode,hexdata ), axis=0))
y_train = (np.concatenate((y_code,y_data ), axis=0))
print ("x_train, y_train shape:")
print (x_train.shape)
print (y_train.shape)


max_features = 1
for i in range(0,len(x_train)):
    for j in range(0,len(x_train[i])):
      #x_train[i][j] = x_train[i][j].astype(np.int64)
        x_train[i][j] = int ("0x"+x_train[i][j], 16)




idx = np.arange(x_train.shape[0])
np.random.shuffle(idx)
x_train = x_train[idx]
y_train = y_train[idx]
x_test = x_train[:80000]
x_train = x_train[80000:]
y_test = y_train[:80000]
y_train = y_train[80000:]
"""
y_test = np.concatenate((y_train[:5000],y_train[158343:163000]), axis=0)
x_test = np.concatenate((x_train[:5000],x_train[158343:163000]), axis=0)
x_train = np.concatenate((x_train[5000:158343],x_train[163000:]), axis=0)
y_train = np.concatenate((y_train[5000:158343],y_train[163000:]), axis=0)
idx = np.arange(x_train.shape[0])
np.random.shuffle(idx)
x_train = x_train[idx]
y_train = y_train[idx]
idx = np.arange(x_test.shape[0])
np.random.shuffle(idx)
x_test = x_test[idx]
y_test = y_test[idx]
"""
print (x_train.shape)
print (x_test.shape)

text_max_words = int(N)
x_train = sequence.pad_sequences(x_train, maxlen=text_max_words)
x_test = sequence.pad_sequences(x_test, maxlen=text_max_words)
print (x_train.shape)
print (x_test.shape)


rf = RandomForestClassifier()
rf.fit(x_train, y_train)
rf.predict_proba(x_test)
print(rf.score(x_test, y_test))

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(1)
plt.title('Feature Importances')
index = np.arange(int(N))
plt.bar(index, importances[index], color='b', align='center')
plt.ylabel('Relative Importance')
plt.xlabel('Feature')
plt.show()
plt.savefig('featureI_1007test.png')


print ("Feature ranking:")
for f in range(x_train.shape[1]):
  print ("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

