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
from keras.layers import Dense, LSTM, Embedding
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
  while os.path.isdir(cur_path+"/result/jt/"+argv[1]+N+"_"+str(n)):
    n += 1
  wDir = cur_path+"/result/jt/"+argv[1]+N+"_"+str(n)
  os.makedirs(wDir)
  os.chdir(wDir)

  #if os.path.isfile(os.getcwd()+"/result/fo_"+N):
  #  print ("Result files for N="+N+" are already exist.")
  #  sys.exit()

# gen training data #
  f = open(cur_path+'/../input/jt'+N+'.list', 'r')
  hexcode = f.readlines()
  f.close()
  for i in range(0,len(hexcode)):
      hexcode[i] = hexcode[i].split(' ')
  #f = open(cur_path+'/../input/FandLP/f'+N+'_hex.anal', 'r')
  #f_info = f.readlines()
  #f.close()

  f = open(cur_path+'/../input/jt'+N+'.anal', 'r')
  ff = f.readlines()
  f.close()    
  hexdata = []
  for i in range(0,len(ff)):
      spl = ff[i].split(' ')
      hexdata.append(np.array([0]*(int(spl[2])-1) + [1] + [0]*(int(N)-int(spl[2]))))
      #hexdata.append(int(spl[2]) - 1)
  print (hexdata[-1])
  x_train = np.array(hexcode)
  y_train = np.array(hexdata)

  print (y_train.shape)
  uniq = np.unique(y_train)
  print (uniq.shape)
  
  ff = np.array(ff)

#  code:0 / data:1
  c_sum, d_sum, t_sum = 0.0, 0.0, 0.0
  step = 1 


  iter_n = 1
  # it here
  summ = 0.0
  val_summ = 0.0
  fo_s, fx_s, do_s, dx_s = 0.0, 0.0, 0.0, 0.0

  outf = open("result", 'w')
  #f_o = open("fo", "w")
  #f_x = open("fx", "w")
  #d_o = open("do", "w")
  #d_x = open("dx", "w")
  for i in range(iter_n):
    print ("**************** "+str(i)+"th iteration *****************")
    print ("x_train, x_train shape:")
    print (x_train.shape)
    print (y_train.shape)


    for i in range(0,len(x_train)):
        for j in range(0,len(x_train[i])):
          #x_train[i][j] = int ("0x" + x_train[i][j].astype(np.int64), 16)
          x_train[i][j] = int ("0x"+x_train[i][j], 16)


    idx = np.arange(x_train.shape[0])
    np.random.shuffle(idx)
    ff = ff[idx]
    ff = ff[:1500]

    # 1
    x_train = x_train[idx]
    y_train = y_train[idx]

    # 2
    #x_test = x_train[:1500]
    #x_train = x_train[1500:]
    #y_test = y_train[:1500]
    #y_train = y_train[1500:]

# TODO
    text_max_words = int(N)
    x_train = sequence.pad_sequences(x_train, maxlen=text_max_words)
    #x_test = sequence.pad_sequences(x_test, maxlen=text_max_words)
    print (x_train.shape)
    #print (x_test.shape)


    fold_n=5
    kfold = StratifiedKFold(n_splits=fold_n, shuffle=True, random_state=7)
    y_train_cat = [np.argmax(y_train[j]) for j in range(len(y_train))]

    acc = 0.0
    #for tr, val in kfold.split(x_train, y_train):
    #for tr, val in kfold.split(x_train, y_train.categorical.argmax(1)):
    ii = 0
    for tr, val in kfold.split(x_train, y_train_cat):
      model = Sequential()
      model.add(Embedding(256,64))
      model.add(LSTM(64))
# deep
      if argv[1] == "d":
        model.add(Dense(64, activation='relu'))
      model.add(Dense(units=int(N), activation='softmax'))
      model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

      hist = model.fit(x_train[tr], y_train[tr], epochs=100, batch_size=64)

      eval_ = float('%.4f' % (model.evaluate(x_train[val], y_train[val])[1])) 
      acc += eval_
      #acc += float('%.4f' % (model.evaluate(x_train[val], y_train[val])[1]))
      #print ("******* acc:", acc)
      outf.write("******" + str(ii) + "th c&v ******\n")
      outf.write(str(eval_)+"\n\n")
      ii += 1


      """
      outf = open("out_jt_1012", 'w')
      for idx, pred in enumerate(model.predict(x_test)):
        outf.write("\n## "+ str(idx)+"\n")
        outf.write("\t- ans:")
        outf.write(str(np.argmax(y_test[idx])))
        #outf.write(str(y_test[idx]))
        outf.write("\n- pred:")
        outf.write(str(np.argmax(pred)))
        #outf.write(str(pred))
      outf.close()
      """

    val_summ = acc / fold_n
    print (val_summ)
    outf.close()

    """
    l_and_m = model.evaluate(x_test, y_test, batch_size=64)
    print ("## eval loss and metrics")
    print (l_and_m)

    #model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=100, verbose=0)
    
    acc = float('%.4f' % (model.evaluate(x_test, y_test)[1]))

  
    pred = model.predict(x_test)
    print (pred)
    print (accuracy_score(y_test, pred))
    print (np.round(accuracy_score(y_test, pred), 4))
    val_summ += (acc)
    #val_summ += acc

    print ("acc:", acc)
    """



if __name__ == '__main__':
  main(sys.argv)
