import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import mglearn
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, BatchNormalization, Activation
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
import seaborn as sns


def main(argv):
  if len(argv) < 3:
    print ("Usage: python3 mphex.py d|mp inputVecSize")
    sys.exit()

  cur_path = os.getcwd()
  N = argv[2]
  n = 0
  while os.path.isdir(cur_path+"/result/"+argv[1]+N+"_"+str(n)):
    n += 1
  wDir = cur_path+"/result/"+argv[1]+N+"_"+str(n)
  os.makedirs(wDir)
  os.chdir(wDir)

  #if os.path.isfile(os.getcwd()+"/result/fo_"+N):
  #  print ("Result files for N="+N+" are already exist.")
  #  sys.exit()
  if not os.path.isfile(cur_path+'/../post_recent/output/df_1007test_'+N+'.list'):
    print ("Input hex file does not exist.")
    sys.exit()

# gen training data #
  f = open(cur_path+'/../input/FandLP/large/f'+N+'_hex.list', 'r')
  hexcode = f.readlines()
  f.close()
  for i in range(0,len(hexcode)):
      hexcode[i] = hexcode[i].split(' ')
  f = open(cur_path+'/../input/FandLP/large/f'+N+'_hex.anal', 'r')
  f_info = f.readlines()
  f.close()

  f = open(cur_path+'/../post_recent/output/df_1007test_'+N+'.list', 'r')
  #f = open(cur_path+'/../post_recent/output/df2_'+N+'.list', 'r')
  hexdata = f.readlines()
  f.close()    
  for i in range(0,len(hexdata)):
      hexdata[i] = hexdata[i].split(' ')
  f = open(cur_path+'/../post_recent/output/df_1007test_'+N+'.anal', 'r')
  #f = open(cur_path+'/../post_recent/output/df2_'+N+'.anal', 'r')
  d_info = f.readlines()
  f.close()


  ### 10.07 for test ###
  f = open(cur_path+'/../input/FandLP/large/d'+N+'_hex.list', 'r')
  testD = f.readlines()
  f.close()
  for i in range(0,len(testD)):
      testD[i] = testD[i].split(' ')
  f = open(cur_path+'/../input/FandLP/large/d'+N+'_hex.anal', 'r')
  for i in range(0,len(testD)):
      for j in range(0,len(testD[i])):
        testD[i][j] = int ("0x"+testD[i][j], 16)
  testI = f.readlines()
  f.close()
  testD = np.array(testD)
  testI = np.array(testI)
  testA = [0]*testD.shape[0]
  ######################

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



  iter_n = 1
  # it here
  summ = 0.0
  val_summ = 0.0
  fo_s, fx_s, do_s, dx_s = 0.0, 0.0, 0.0, 0.0

  f_o = open("fo", "w")
  f_x = open("fx", "w")
  d_o = open("do", "w")
  d_x = open("dx", "w")
  for i in range(iter_n):
    print ("**************** "+str(i)+"th iteration *****************")
    x_train = (np.concatenate((hexcode,hexdata ), axis=0))
    y_train = (np.concatenate((y_code,y_data ), axis=0))
    print ("x_train, x_train shape:")
    print (x_train.shape)
    print (y_train.shape)


    for i in range(0,len(x_train)):
        for j in range(0,len(x_train[i])):
          #x_train[i][j] = int ("0x" + x_train[i][j].astype(np.int64), 16)
          x_train[i][j] = int ("0x"+x_train[i][j], 16)

    idx = np.arange(x_train.shape[0])
    np.random.shuffle(idx)
    info = info[idx]
    info_test = info[:1500]
    x_train = x_train[idx]
    y_train = y_train[idx]
    x_test = x_train[:1500]
    x_train = x_train[1500:]
    y_test = y_train[:1500]
    y_train = y_train[1500:]
    print (x_train.shape)
    print (x_test.shape)

    """
    info_test = np.concatenate((info[:5000],info[158343:163000]), axis=0)
    y_test = np.concatenate((y_train[:5000],y_train[158343:163000]), axis=0)
    x_test = np.concatenate((x_train[:5000],x_train[158343:163000]), axis=0)
    x_train = np.concatenate((x_train[5000:158343],x_train[163000:]), axis=0)
    y_train = np.concatenate((y_train[5000:158343],y_train[163000:]), axis=0)
    """

# TODO
    text_max_words = int(N)# 33 
    x_train = sequence.pad_sequences(x_train, maxlen=text_max_words)
    x_test = sequence.pad_sequences(x_test, maxlen=text_max_words)
    print (x_train.shape)
    print (x_test.shape)
    print (x_train)

    fold_n = 5
    #kfold = KFold(n_splits=fold_n, shuffle=True, random_state=7)
    kfold = StratifiedKFold(n_splits=fold_n, shuffle=True, random_state=7)

    acc = 0.0

    for tr, val in kfold.split(x_train, y_train):
      model = Sequential()
      model.add(Dense(64, input_dim=text_max_words, activation='relu'))
# deep
      if argv[1] == "d":
        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        #model.add(Dense(64, activation='relu'))
      model.add(Dense(1, activation='sigmoid'))
      #model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
      model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#model.compile('adam', 'mae')

      #class_weight = { 1: 0.4, 0: 0.6}
      #hist = model.fit(x_train[tr], y_train[tr], epochs=10, batch_size=128, class_weight=class_weight)
      hist = model.fit(x_train[tr], y_train[tr], epochs=10, batch_size=128)


      #model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=100, verbose=0)

      
      acc += float('%.4f' % (model.evaluate(x_train[val], y_train[val])[1]))
      #pred = model.predict(x_train[val])
      #acc.append(np.round(accuracy_score(y_train[val], pred), 4))
      print ("************x_train[va;],******")
      print (x_train[val])
      print (x_train.shape)

    val_summ += (acc / fold_n)
    #val_summ += acc

    l_and_m = model.evaluate(x_test, y_test, batch_size=32)
    print ('loss_and_metrics: ' + str(l_and_m))

    summ += l_and_m[1]




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

    fo_s += fo
    fx_s += fx
    do_s += do
    dx_s += dx

  res = open("result", "w")
  res.write("loss & mertircs: "+str(summ/iter_n)+"\nfo: "+str(fo_s/iter_n)+ ", fx: "+str(fx_s/iter_n)+", do: "+str(do_s/iter_n)+", dx: "+str(dx_s/iter_n))
  print("val_avg: " + str(val_summ), " loss & mertircs: "+str(summ/iter_n)+"\nfo: "+str(fo_s/iter_n)+ ", fx: "+str(fx_s/iter_n)+", do: "+str(do_s/iter_n)+", dx: "+str(dx_s/iter_n))

  pred = model.predict(testD)
  print ("*** testD prediction: ")
  print (np.round(accuracy_score(testA, pred.round(),normalize=False), 4))
  print (accuracy_score(testA, pred.round(),normalize=False))
  print (accuracy_score(testA, pred.round()))

  print ("Write results in:" + wDir)



  f_o.close()
  f_x.close()
  d_o.close()
  d_x.close()

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
  plt.savefig(wDir+'/res.png')

if __name__ == '__main__':
  main(sys.argv)
