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
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.inspection import permutation_importance


# gen training data #
cur_path = os.getcwd()
N = str(20)

# info #
f = open(cur_path+'/../../input/FandLP/large/f'+N+'_hex.anal', 'r')
info = f.readlines()
f.close()
info = np.array(info)
########

ori_f_hex = 0
f = open(cur_path+'/../../input/FandLP/large/f'+N+'_hex.list', 'r')
ori_f_hex = f.readlines()
f.close()

# function #
f = open(cur_path+'/../../input/FandLP/large/f'+N+'_hex.list', 'r')
hexcode = f.readlines()
f.close()
for i in range(0,len(hexcode)):
    hexcode[i] = hexcode[i].split(' ')

# data #
f = open(cur_path+'/../../post_recent/output/df_1009test_'+N+'.list', 'r')
#f = open(cur_path+'/../../post_recent/output/df_1007test_'+N+'.list', 'r')
hexdata = f.readlines()
f.close()    
for i in range(0,len(hexdata)):
    hexdata[i] = hexdata[i].split(' ')
hexcode = np.array(hexcode)
hexdata = np.array(hexdata)
ori_f_hex = np.array(ori_f_hex)
print ("code, data shape:")
print (hexcode.shape)
print (hexdata.shape)

#  code:0 / data:1
c_sum, d_sum, t_sum = 0.0, 0.0, 0.0
step =1 
y_code = [1]*hexcode.shape[0]
y_data = [0]*hexdata.shape[0]
info_data=["***"]*hexdata.shape[0]
x_train = (np.concatenate((hexcode,hexdata ), axis=0))
info = (np.concatenate((info,info_data ), axis=0))
ori_f_hex = (np.concatenate((ori_f_hex,info_data ), axis=0))
y_train = (np.concatenate((y_code,y_data ), axis=0))
print ("x_train, y_train shape:")
print (x_train.shape)
print (y_train.shape)
print ("info, ori shape:")
print (info.shape)
print (ori_f_hex.shape)


max_features = 1
for i in range(0,len(x_train)):
    for j in range(0,len(x_train[i])):
      #x_train[i][j] = x_train[i][j].astype(np.int64)
        x_train[i][j] = int ("0x"+x_train[i][j], 16)




idx = np.arange(x_train.shape[0])
np.random.shuffle(idx)
x_train = x_train[idx]
y_train = y_train[idx]
info = info[idx]
info = info[:80000]
ori_f_hex = ori_f_hex[idx]
ori_f_hex = ori_f_hex[:80000]
x_test = x_train[:80000]
x_train = x_train[80000:]
y_test = y_train[:80000]
y_train = y_train[80000:]
print (x_train.shape)
print (x_test.shape)

text_max_words = int(N)
x_train = sequence.pad_sequences(x_train, maxlen=text_max_words)
x_test = sequence.pad_sequences(x_test, maxlen=text_max_words)
print (x_train.shape)
print (x_test.shape)


#rf = RandomForestClassifier()
#rf.fit(x_train, y_train)
#rf.predict_proba(x_test)
#print(rf.score(x_test, y_test))

rf = RandomForestClassifier()
#score = cross_val_score(rf, x_train, y_train, cv=5)
#print ("cross: ", np.round(score,4))
#print ("cross_avg: ", np.round(np.mean(score),4))
rf.fit(x_train, y_train)
print(rf.score(x_test, y_test))



fo = open("1009fo", "w")
fx = open("1009fx", "w")
fo_h = open("1009fo_hex", "w")
fx_h = open("1009fx_hex", "w")

o = []
oh = []
x = []
xh = []

# TODO: feature importance
# result #
for i, pred in enumerate(rf.predict(x_test)):
  # code
  if y_test[i] == 1:
    if pred == 1:
      o.append(info[i])
      oh.append(ori_f_hex[i])
      #fo.write(info[i])
      #fo_h.write(ori_f_hex[i])
    else:
      x.append(info[i])
      xh.append(ori_f_hex[i])
      #fx.write(info[i])
      #fx_h.write(ori_f_hex[i])
o = np.array(o)
oh = np.array(oh)
x = np.array(x)
xh = np.array(xh)
inds1 = o.argsort()
o = o[inds1].tolist()
oh = oh[inds1].tolist()
inds2 = x.argsort()
x = x[inds2].tolist()
xh = xh[inds2].tolist()

fo.write(''.join(o))
fo_h.write(''.join(oh))
fx.write(''.join(x))
fx_h.write(''.join(xh))

fo.close()
fx.close()
fo_h.close()
fx_h.close()


r = permutation_importance(rf, x_test, y_test, n_repeats=1, random_state=0)
'''
for idx, i in enumerate(r.importances_mean.argsort()[::-1]):
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{i:<8}"
              f"{r.importances_mean[i]:.3f}"
              f" +/- {r.importances_std[i]:.3f}")
'''
sorted_r = r.importances_mean.argsort()

'''
fig, ax = plt.subplots()
ax.boxplot(r.importances[sorted_r].T,
           vert=False, labels=np.arange(int(N))[sorted_r], flierprops=dict(markerfacecolor='g',marker='D'))
ax.set_title("Permutation Importances")
fig.tight_layout()
plt.show()
plt.savefig('1009perImp.png')
'''


'''
fig, ax = plt.subplots()
indices = np.arange(int(N))
ax.barh(range(len(indices)), r.importances.mean(axis=1).T, color='b', align='center')
#ax.ylabel('Relative Importance')
#ax.xlabel('Feature')
ax.set_title("Permutation Importances")
#fig.tight_layout()
plt.show()
plt.savefig('1009perImp_bar.png')
'''

plt.figure(1)
indices = np.arange(int(N))
plt.title('Permutation Importances')
plt.barh(range(len(indices)), r.importances.mean(axis=1).T, color='b', align='center')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
plt.savefig('1009perImp_bar2.png')



sys.exit()


importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
print ("indices.shape:")
print (indices.shape)
plt.figure(1)
plt.title('Feature Importances')
index = np.arange(int(N))
#plt.bar(index, importances[index], color='b', align='center')
plt.bar(range(len(indices)), importances[indices], color='b', align='center')
plt.ylabel('Relative Importance')
plt.xlabel('Feature')
plt.show()
plt.savefig('1009featureI_1007test.png')


print ("Feature ranking:")
for f in range(x_train.shape[1]):
  print ("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))




