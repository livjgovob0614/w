import sys
import numpy as np
import matplotlib.pyplot as plt
import mglearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz
import graphviz
import pydot

# gen training data #
f = open('/home/donghoon/ssd/test_kby/csvFiles/hexcode200629_150352.csv','r')
hexcode = f.readlines()
f.close()
for i in range(0,len(hexcode)):
    hexcode[i] = hexcode[i].split(' ')

f = open('/home/donghoon/ssd/test_kby/csvFiles/hexdata200629_150352.csv','r')
hexdata = f.readlines()
f.close()    
for i in range(0,len(hexdata)):
    hexdata[i] = hexdata[i].split(' ')
"""
f = open('../csvFiles/hexdata200702_123446.csv','r')
hexdata_n = f.readlines()
f.close()
for i in range(0,len(hexdata_n)):
    hexdata_n[i] = hexdata_n[i].split(' ')

f = open('../csvFiles/hexcode200702_123446.csv','r')
hexcode_n = f.readlines()
f.close()
for i in range(0,len(hexcode_n)):
    hexcode_n[i] = hexcode_n[i].split(' ')
"""
hexcode = np.array(hexcode)
hexdata = np.array(hexdata)
"""
hexcode_n = np.array(hexcode_n)
hexdata_n = np.array(hexdata_n)
hexcode = np.concatenate((hexcode,hexcode_n), axis=0)
hexdata = np.concatenate((hexdata,hexdata_n), axis=0)
""" 

#  code:0 / data:1
c_sum, d_sum, t_sum = 0.0, 0.0, 0.0
for j in range(5):
  y_code = [0]*hexcode.shape[0]
  y_data = [1]*hexdata.shape[0]
  x_train = (np.concatenate((hexcode,hexdata ), axis=0))
  y_train = (np.concatenate((y_code,y_data ), axis=0))
#  print (x_train.shape)
#  print (y_train.shape)

# In[5]:
  max_features = 256
  text_max_words = 84
  for i in range(0,len(x_train)):
      for j in range(0,len(x_train[i])):
          x_train[i][j] = int("0x"+x_train[i][j], 16)

  idx = np.arange(x_train.shape[0])
  np.random.shuffle(idx)
  x_train = x_train[idx]
  y_train = y_train[idx]

  y_test = y_train[12000:]
  x_test = x_train[12000:]
  x_train = x_train[:12000]
  y_train = y_train[:12000]
#  print (x_test.shape)
#  print (y_test.shape)

# In[13]:
  x_train = sequence.pad_sequences(x_train, maxlen=text_max_words)
  x_test = sequence.pad_sequences(x_test, maxlen=text_max_words)


# learn SVM #
  #dt_ = DecisionTreeClassifier()
  #dt_ = DecisionTreeClassifier(min_samples_split=10)
  dt_ = DecisionTreeClassifier(min_samples_split=10, max_depth=3)
  dt_.fit(x_train, y_train)

#f = open('predict','w')
  code_cnt, data_cnt = 0, 0
  code_cor, data_cor = 0, 0
  for i, pred in enumerate(dt_.predict(x_test)):
    # code
    if y_test[i] == 0:
      code_cnt += 1
      if pred == y_test[i]:
        code_cor += 1 
        #f.write (str(i)+"(fail, (x,y)):"+str(x_test[i])+str(y_test[i])+"(pred:)"+str(pred)+"\n")
    if y_test[i] == 1:
      data_cnt += 1
      if pred == y_test[i]:
        data_cor += 1 

#  print ("code, data caes:")
  c_sum += code_cor / code_cnt * 100
  d_sum += data_cor / data_cnt * 100
#  print (code_cor / code_cnt * 100)
#  print (data_cor / data_cnt * 100)
#f.close()

  score = dt_.score(x_test, y_test)
  t_sum += score
#  print (score)
#print (cross_val_score(svm, x_train, y_train, cv=10))

print ("c, d, total avg:")
print (c_sum / 5, d_sum / 5, t_sum / 5)

score = dt_.score(x_test, y_test)
print (score)
#print (cross_val_score(svm, x_train, y_train, cv=10))

export_graphviz(dt_, out_file="tree.dot", class_names=["code", "data"],
                feature_names=None, impurity=False, filled=True)

(graph,) = pydot.graph_from_dot_file('tree.dot', encoding='utf8')
graph.write_png('tree.png')

sys.exit()

# TODO
#print (svm.score(x_test_new, y_test_new))
#print (cross_val_score(svm, x_test_new, y_test_new, cv=10))





######### TODO

#mglearn.discrete_scatter(x_train[:, 0], x_train[:, 1], y_train)
#mglearn.plots.plot_2d_separator(svm, x_train)
#plt.show()
#plt.savefig('3.png')
#sys.exit()



# 샘플 데이터 표현
xmax = x_train[:,0].max()+0.1
xmin = x_train[:,0].min()-.1
ymax = x_train[:,1].min()+.1
ymin = x_train[:,1].min()-.1

plt.scatter(x_train[:,0], x_train[:,1], c=y_train, s=30, cmap=plt.cm.Paired)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.xlabel("x1")
plt.ylabel("x2")
plt.plot()
plt.savefig('2.png')


sys.exit()

# ==================================================================================================

# 초평면(Hyper-Plane) 표현
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
print ("xlim, ylim:")
print (len(xlim))
print (len(ylim))
print (type(xlim))

xx = np.linspace(xmin, xmax, 84)
yy = np.linspace(ymin, ymax, 84)
print ("xx, yy:")
print (xx.shape)
print (yy.shape)
YY, XX = np.meshgrid(yy, xx)
print ("XX, YY:")
print (XX.shape)
print (YY.shape)
print ("XX, YY reshape:")
print (XX.reshape(-1,84).shape)
print (YY.reshape(-1,84).shape)
#xy = np.vstack([XX.ravel(), YY.ravel()]).reshape(84,-1).T
xy = np.vstack([XX.reshape(-1,84), YY.reshape(-1,84)]).T
print ("xy:")
print (xy.shape)
Z = svm.decision_function(xy).reshape(XX.shape)
print ("z:")
print (z.shape)
ax.contour(XX, YY, Z, colors='k', levels=[-1,0,1], alpha=0.5, linestyles=['--', '-', '--'])

# 지지벡터(Support Vector) 표현
ax.scatter(svm.support_vectors_[:,0], svm.support_vectors_[:,1], s=60, facecolors='r')
plt.show()
plt.savefig('1.png')
