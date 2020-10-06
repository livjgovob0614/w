import sys, os
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
import seaborn as sns


def main(argv):
  if len(argv) < 3:
    print ("Usage: python3 mphex.py d|mp inputVecSize")
    sys.exit()

  N = argv[2]
  n = 0
  while os.path.isdir(os.getcwd()+"/result/"+argv[1]+N+"_"+str(n)):
    n += 1
  wDir = os.getcwd()+"/result/"+argv[1]+N+"_"+str(n)
  os.makedirs(wDir)
  os.chdir(wDir)

  #if os.path.isfile(os.getcwd()+"/result/fo_"+N):
  #  print ("Result files for N="+N+" are already exist.")
  #  sys.exit()
  if not os.path.isfile('/home/donghoon/ssd/jg/disasm/input/FandLP/f'+N+'_hex.list'):
    print ("Input hex file does not exist.")
    sys.exit()

# gen training data #
#f = open('/home/donghoon/ssd/jg/disasm/input/function_onlySub_hex.list', 'r')
  f = open('/home/donghoon/ssd/jg/disasm/input/FandLP/f'+N+'_hex.list', 'r')
  hexcode = f.readlines()
  f.close()
  for i in range(0,len(hexcode)):
      hexcode[i] = hexcode[i].split(' ')
  f = open("/home/donghoon/ssd/jg/disasm/output/function_onlySub.anal", 'r')
  f_info = f.readlines()
  f.close()

#f = open('/home/donghoon/ssd/jg/disasm/input/LPdata_hex.list', 'r')
  f = open('/home/donghoon/ssd/jg/disasm/input/FandLP/d'+N+'_hex.list', 'r')
  hexdata = f.readlines()
  f.close()    
  for i in range(0,len(hexdata)):
      hexdata[i] = hexdata[i].split(' ')
  f = open("/home/donghoon/ssd/jg/disasm/output/data.anal", 'r')
  d_info = f.readlines()
  f.close()

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



  # it here
  summ = 0.0
  val_summ = 0.0
  fo_s, fx_s, do_s, dx_s = 0.0, 0.0, 0.0, 0.0
  for i in range(5):
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


# TODO
    text_max_words = int(N)# 33 
    x_train = sequence.pad_sequences(x_train, maxlen=text_max_words)
    x_test = sequence.pad_sequences(x_test, maxlen=text_max_words)
    print (x_train.shape)
    print (x_test.shape)

    def create_model():
      model = Sequential()
      model.add(Dense(64, input_dim=text_max_words, activation='relu'))
# deep
      if argv[1] == "d":
        model.add(Dense(64, activation='relu'))
      model.add(Dense(1, activation='sigmoid'))
      model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
      return model
#model.compile('adam', 'mae')

    #class_weight = { 1: 0.1, 0: 0.9}
    class_weight = { 1: 0.3, 0: 0.7}
    #hist = model.fit(x_train, y_train, epochs=10, batch_size=100, class_weight=class_weight)
    #model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=100, verbose=0, class_weight = class_weight)
    model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=100, verbose=0)

    kfold = KFold(n_splits=5, shuffle=True, random_state=7)
    results = cross_val_score(model, x_train, y_train, cv=kfold)
    #sns_plot = sns.lineplot(data=results)
    sns_plot = sns.regplot(x='hex',y='addr type', data=x_train)
    fig = sns_plot.get_figure()
    fig.savefig("/home/jg/kby/disasm/test/DNN/20.09~/recent/out.png")
    
    print (results)
    for i in range(5):
      val_summ += results[i]
    val_summ /= 5.0

    #l_and_m = model.evaluate(x_test, y_test, batch_size=32)
    #print ('loss_and_metrics: ' + str(l_and_m))


# TODO: feature importance
    fo, fx, do, dx = 0, 0, 0, 0
# result #
    for i, pred in enumerate(model.predict(x_test)):
      # data
      if y_test[i] == 0:
        if pred < 0.5:
          #d_o.write(info_test[i])
          do += 1
        else:
          #d_x.write(info_test[i])
          dx += 1
      # code
      if y_test[i] == 1:
        if pred >= 0.5: #TODO: >= or > ?
          #f_o.write(info_test[i])
          fo += 1
        else:
          #f_x.write(info_test[i])
          fx += 1

    fo_s += fo
    fx_s += fx
    do_s += do
    dx_s += dx

  res = open("result", "w")
  res.write("loss & mertircs: "+str(summ/5.0)+"\nfo: "+str(fo_s/5.0)+ ", fx: "+str(fx_s/5.0)+", do: "+str(do_s/5.0)+", dx: "+str(dx_s/5.0))
  print("val_avg: " + str(val_summ/5.0) + "loss & mertircs: "+str(summ/5.0)+"\nfo: "+str(fo_s/5.0)+ ", fx: "+str(fx_s/5.0)+", do: "+str(do_s/5.0)+", dx: "+str(dx_s/5.0))

  sys.exit()

  f_o = open("fo", "w")
  f_x = open("fx", "w")
  d_o = open("do", "w")
  d_x = open("dx", "w")

  f_o.close()
  f_x.close()
  d_o.close()
  d_x.close()

  '''
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
  '''

if __name__ == '__main__':
  main(sys.argv)
