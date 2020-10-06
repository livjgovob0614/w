import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import mglearn
from sklearn import svm
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import cross_val_score, KFold

f = open("/home/donghoon/ssd/jg/disasm/output/function_onlySub.anal", 'r')
f_info = f.readlines()
f.close()
# gen training data #
f = open('/home/donghoon/ssd/jg/disasm/output/function_onlySub.feature', 'r')
hexcode = f.readlines()
f.close()
for i in range(0,len(hexcode)):
    hexcode[i] = hexcode[i].split(' ')

f = open("/home/donghoon/ssd/jg/disasm/output/data.anal", 'r')
d_info = f.readlines()
f.close()
f = open('/home/donghoon/ssd/jg/disasm/output/data.feature', 'r')
hexdata = f.readlines()
f.close()    
for i in range(0,len(hexdata)):
    hexdata[i] = hexdata[i].split(' ')
hexcode = np.array(hexcode)
hexdata = np.array(hexdata)
print ("code, data shape:")
print (hexcode.shape)
print (hexdata.shape)

f_info = np.array(f_info)
d_info = np.array(d_info)

info = (np.concatenate((f_info, d_info), axis=0))
#  code:0 / data:1
c_sum, d_sum, t_sum = 0.0, 0.0, 0.0
step =1 
y_code = [1]*hexcode.shape[0]
y_data = [0]*hexdata.shape[0]
x_train = (np.concatenate((hexcode,hexdata ), axis=0))
y_train = (np.concatenate((y_code,y_data ), axis=0))
print ("x_train, x_train shape:")
print (x_train.shape)
print (y_train.shape)
print (info.shape)


max_features = 1
for i in range(0,len(x_train)):
    for j in range(0,len(x_train[i])):
      x_train[i][j] = x_train[i][j].astype(np.int64)

info_test = np.concatenate((info[:5000],info[158343:163000]), axis=0)

y_test = np.concatenate((y_train[:5000],y_train[158343:163000]), axis=0)
x_test = np.concatenate((x_train[:5000],x_train[158343:163000]), axis=0)
x_train = np.concatenate((x_train[5000:158343],x_train[163000:]), axis=0)
y_train = np.concatenate((y_train[5000:158343],y_train[163000:]), axis=0)
idx = np.arange(x_train.shape[0])
np.random.shuffle(idx)
x_train = x_train[idx]
y_train = y_train[idx]
print (x_train.shape)
print (x_test.shape)


text_max_words = 33 
x_train = sequence.pad_sequences(x_train, maxlen=text_max_words)
x_test = sequence.pad_sequences(x_test, maxlen=text_max_words)
print (x_train.shape)
print (x_test.shape)

model = Sequential()
model.add(Dense(64, input_dim=33, activation='relu'))
# deep
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
#model.compile('adam', 'mae')

class_weight = { 1: 0.1, 0: 0.9}
hist = model.fit(x_train, y_train, epochs=100, batch_size=100, class_weight=class_weight)
#hist = model.fit(x_train, y_train, epochs=100, batch_size=64)

l_and_m = model.evaluate(x_test, y_test, batch_size=32)
print ('loss_and_metrics: ' + str(l_and_m))
sys.exit()

# TODO: feature importance

f_o = open("f_o", "w")
f_x = open("f_x", "w")
d_o = open("d_o", "w")
d_x = open("d_x", "w")
# result #
for i, pred in enumerate(model.predict(x_test)):
  # data
  if y_test[i] == 0:
    if pred < 0.5:
      d_o.write(info_test[i])
    else:
      d_x.write(info_test[i])
	# code
  if y_test[i] == 1:
    if pred >= 0.5:
      f_o.write(info_test[i])
    else:
      f_x.write(info_test[i])
f_o.close()
f_x.close()
d_o.close()
d_x.close()

#score = model.score(x_test, y_test)
#print (score)

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
plt.savefig('m_p.png')

l_and_m = model.evaluate(x_test, y_test, batch_size=32)
print ('loss_and_metrics: ' + str(l_and_m))
sys.exit()


#model = Sequential


sys.exit()

