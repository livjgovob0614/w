import sys, os, glob
import numpy as np
import subprocess

STR = 0
ARY = 1
NUM = 2
FUNC = 3

def main(argv):
  if len(argv) < 2:
    print ("Usage: python3 make_out..py byteSize")
    sys.exit()

  size = int(argv[1])
  cur_path = os.getcwd()
  in_path = cur_path+"/../input/TypeandF/"

  N = size
  #if os.path.isfile(in_path+"f"+size+"_hex.list"):
  #  print ("Hex files for N="+size+" are already exist.")
  #  sys.exit()

  os.chdir(in_path)

  flist = ["string_hex.list", "array_hex.list", "else_hex.list", "/home/donghoon/ssd/jg/20/input/FandLP/large/f100_hex.list"]

  hexf = []
  hexf_size = []
  for i in range(len(flist)):
    f = open(flist[i])
    hexf.append(f.readlines())
    hexf_size.append(len(hexf[i]))
    f.close()

  out_f = open(cur_path+"/output/df_1007test_"+str(size)+".list", "w")
  anal_f = open(cur_path+"/output/df_1007test_"+str(size)+".anal", "w")

#  out_f_mul = open("tyN+f_hex.list")
#  anal_f_mul = open("tyN+f_hex.anal")

  """
  # TODO: If exist, exit
  out_f = open("dmix"+size+"_hex.list", 'w')
  anal_f = open("dmix"+size+"_hex.anal", 'w')
  """
  n_of_data = 140000
  mode = 0

  #np.random.seed(size+7777)
  # To compare ..
  np.random.seed(77777)
  N = 0
  for i in range(n_of_data):
    if not i % 10000:
      print ("Extracting data ...")

    write_str, line = [], []
    if i == 50000:
      mode = 1
    elif i == 80000:
      mode = 3
    elif i == 120000:
      mode = 2
    
    # type1 + f
    if not mode:
      # Read 1 random type hex #
      ty = np.random.randint(3)
      N = np.random.randint(hexf_size[ty])

      line = hexf[ty][N][:-1].split() # remove '\n'

      anal_f.write("type "+ str(ty)+ "["+str(N)+ "]")
      while len(write_str) < size and line:
        write_str.append(line.pop(0))
    elif mode == 1:
      n_ty = np.random.randint(1, 4)
      for j in range(n_ty):
        ty = np.random.randint(3)
        N = np.random.randint(hexf_size[ty])
        anal_f.write("type "+str(ty)+ "["+ str(N)+ "]")

        line = hexf[ty][N][:-1].split() # remove '\n'
        while len(write_str) < size and line:
          write_str.append(line.pop(0))

        if len(write_str) == size:
          break
    elif i == 2:
      n_int = np.random.randint(2, 8)
      for j in range(n_int):
        N = np.random.randint(hexf_size[NUM])
        anal_f.write("type NUM["+str(N)+ "]")
        line = hexf[NUM][N][:-1].split() # remove '\n'
        while len(write_str) < size and line:
          write_str.append(line.pop(0))
        if len(write_str) == size:
          break

      n_ty = np.random.randint(3)
      for j in range(n_ty):
        ty = np.random.randint(3)
        N = np.random.randint(hexf_size[ty])
        anal_f.write("type "+str(ty)+ "["+ str(N)+ "]")
        line = hexf[ty][N][:-1].split() # remove '\n'
        while len(write_str) < size and line:
          write_str.append(line.pop(0))
        if len(write_str) == size:
          break
    else:
      while (True):
        ty = np.random.randint(3)
        N = np.random.randint(hexf_size[ty])
        spl = hexf[ty][N][:-1].split() # remove '\n'
        if len(spl) >= 7 and not (int(spl[3],16) <= 143 or int(spl[2],16) <= 191 or int(spl[6],16) <= 96):
          break

      anal_f.write("type "+ str(ty)+ "["+str(N)+ "]")

      line = hexf[ty][N][:-1].split() # remove '\n'
      while len(write_str) < size and line:
        write_str.append(line.pop(0))
      


    # Read function hex #
    while len(write_str) < size:
      N = np.random.randint(hexf_size[FUNC])
      line = hexf[FUNC][N][:-1].split()
      anal_f.write("  +  type func["+str(N)+"]")
      while len(write_str) < size and line:
        write_str.append(line.pop(0))

    out_f.write(' '.join(write_str) + "\n")
    anal_f.write("\n")

  out_f.close()
  anal_f.close()

    
if __name__ == '__main__':
    main(sys.argv)
