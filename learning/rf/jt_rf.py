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
from sklearn.inspection import permutation_importance

# gen training data #
cur_path = os.getcwd()
N = str(4500)

#f = open(cur_path+'/../../input/large/jt'+N+'_hex.list', 'r')
f = open(cur_path+'/../../input/1009test_jt'+N+'_hex.list', 'r')
hexcode = f.readlines()
f.close()
for i in range(0,len(hexcode)):
    hexcode[i] = hexcode[i].split(' ')

#f = open(cur_path+'/../../input/large/jt'+N+'_hex.anal', 'r')
f = open(cur_path+'/../../input/1009test_jt'+N+'_hex.anal', 'r')
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


rf2 = RandomForestClassifier()
score = cross_val_score(rf2, x_train, y_train, cv=5)
print ("** ** ** cross: ", np.round(score,4))
print ("** ** ** cross_avg: ", np.round(np.mean(score),4))




rf = RandomForestClassifier()
rf.fit(x_train, y_train)
#rf.predict_proba(x_test)
#print(rf.score(x_test, y_test))

'''
r = permutation_importance(rf, x_test, y_test, n_repeats=1, random_state=0)
indices = np.arange(int(N))
plt.figure(1)
plt.title('Permutation Importances')
plt.barh(range(len(indices)), r.importances.mean(axis=1).T, color='b', align='center')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
plt.savefig('jt1009_perImp_bar.png')
'''


d_o = open("oo", "w")
d_x = open("xx", "w")

#rf.predict_proba(x_test)
print(rf.score(x_test, y_test))
# TODO: feature importance
fo, fx, do, dx = 0, 0, 0, 0
# result #
for i, pred in enumerate(rf.predict(x_test)):
  # data
  if y_test[i] == pred:
    d_o.write(ff[i])
    do += 1
  else:
    d_x.write(ff[i])
    dx += 1

d_o.close()
d_x.close()
#model = Sequential


importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(1)
plt.title('Feature Importances')
index = np.arange(20)
plt.bar(index, importances[index], color='b', align='center')
plt.ylabel('Relative Importance')
plt.xlabel('Index of Input Vector')
plt.show()
plt.savefig('jt_df.png')


print ("Feature ranking:")
#for f in range(x_train.shape[1]):
for f in range(20):
  print ("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
sys.exit()

