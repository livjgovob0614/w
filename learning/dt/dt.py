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
from sklearn import tree
from sklearn.model_selection import GridSearchCV

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


val_x = np.array(x_train)
val_y = np.array(y_train)


idx = np.arange(x_train.shape[0])
np.random.shuffle(idx)
x_train = x_train[idx]
y_train = y_train[idx]
x_test = x_train[:30000]
x_train = x_train[30000:]
y_test = y_train[:30000]
y_train = y_train[30000:]
print (x_train.shape)
print (x_test.shape)

text_max_words = int(N)
x_train = sequence.pad_sequences(x_train, maxlen=text_max_words)
x_test = sequence.pad_sequences(x_test, maxlen=text_max_words)
print (x_train.shape)
print (x_test.shape)


#dt = DecisionTreeClassifier(min_samples_split=2, max_depth=3)
#dt.fit(x_train, y_train)
#dt.predict_proba(x_test)
#print(dt.score(x_test, y_test))
dt = DecisionTreeClassifier(min_samples_split=2, max_depth=3)
score = cross_val_score(dt, x_train, y_train, cv=5)
print ("cross: ", np.round(score,4))
print ("cross_avg: ", np.round(np.mean(score),4))
export_graphviz(dt, out_file="tree.dot", class_names=["data", "code"],
                feature_names=None, impurity=False, filled=True)

(graph,) = pydot.graph_from_dot_file('tree.dot', encoding='utf8')
graph.write_png('tree_1007test.png')


param = {'max_depth':range(5)}
clf = GridSearchCV(tree.DecisionTreeClassifier(), param, n_jobs=4)
clf.fit(val_x, val_y)
tree_model = clf.best_estimator_
print (clf.best_score_, clf.best_params_)

# feature importance
idx = np.arange(int(N))
feature_imp = dt.feature_importances_
plt.barh(idx, feature_imp, align='center')
#plt.yticks(idx, ["data", "code"])
plt.xlabel('feature importance')
plt.ylabel('feature')
plt.show()
plt.savefig('dt_featureImp.png')

