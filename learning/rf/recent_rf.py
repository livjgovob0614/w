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
N = str(40)

f = open(cur_path+'/../../input/FandLP/large/f'+N+'_hex.list', 'r')
hexcode = f.readlines()
f.close()
for i in range(0,len(hexcode)):
    hexcode[i] = hexcode[i].split(' ')

#f = open(cur_path+'/../../post_recent/output/df_'+N+'.list', 'r')
f = open(cur_path+'/../../post_recent/output/df_1007test_'+N+'.list', 'r')
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
x_test = x_train[:30000]
x_train = x_train[30000:]
y_test = y_train[:30000]
y_train = y_train[30000:]
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
plt.xlabel('Index of Input Vector')
plt.show()
plt.savefig('featureI_1007test.png')


print ("Feature ranking:")
for f in range(x_train.shape[1]):
  print ("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
sys.exit()


model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
#model.compile('adam', 'mae')

#class_weight = { 1: 0.05, 0: 0.95}
class_weight = { 1: 0.3, 0: 0.7}
hist = model.fit(x_train, y_train, epochs=10, batch_size=100, class_weight=class_weight)
#hist = model.fit(x_train, y_train, epochs=100, batch_size=64)

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.set_ylim([0.0, 1.0])
acc_ax.set_ylim([0.0, 1.0])

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()
plt.savefig('rf.png')

l_and_m = model.evaluate(x_test, y_test, batch_size=32)
print ('loss_and_metrics: ' + str(l_and_m))


f_o = open("fo", "w")
f_x = open("fx", "w")
d_o = open("do", "w")
d_x = open("dx", "w")

# TODO: feature importance
fo, fx, do, dx = 0, 0, 0, 0
# result #
for i, pred in enumerate(model.predict(x_test)):
  # data
  if y_test[i] == 0:
    if pred < 0.5:
      d_o.write(info_test[i])
      do += 1
    else:
      d_x.write(info_test[i])
      dx += 1
  # code
  if y_test[i] == 1:
    if pred >= 0.5: #TODO: >= or > ?
      f_o.write(info_test[i])
      fo += 1
    else:
      f_x.write(info_test[i])
      fx += 1

f_o.close()
f_x.close()
d_o.close()
d_x.close()
#model = Sequential


sys.exit()

