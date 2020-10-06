import sys, os, glob
import numpy as np
import subprocess

STR = 0
ARY = 1
ELSE = 2
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

  flist = ["string_hex.list", "array_hex.list", "else_hex.list", "/home/donghoon/ssd/jg/20/input/FandLP/f100_hex.list"]

  hexf = []
  hexf_size = []
  for i in range(len(flist)):
    f = open(flist[i])
    hexf.append(f.readlines())
    hexf_size.append(len(hexf[i]))
    f.close()

  out_f = open(cur_path+"/output/dftest_"+str(size)+".list", "w")
  anal_f = open(cur_path+"/output/dftest_"+str(size)+".anal", "w")

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
  selected = [0, 0, 0]
  N = 0
  for i in range(n_of_data):
    if not i % 10000:
      print ("Extracting data ...")

    write_str, line = [], []
    if i == 50000:
      mode = 1
    
    # type1 + f
    if not mode:
      # Read 1 random type hex #
      selected[0] = np.random.randint(3)
      N = np.random.randint(hexf_size[selected[0]])
      anal_f.write("type "+ str(selected[0])+ "["+str(N)+ "]")

      line = hexf[selected[0]][N][:-1].split() # remove '\n'
      while len(write_str) < size and line:
        write_str.append(line.pop(0))
    elif mode == 1:
      n_ty = np.random.randint(1, 4)
      for j in range(n_ty):
        selected[j] = np.random.randint(3)
        N = np.random.randint(hexf_size[selected[j]])
        anal_f.write("type "+str(selected[j])+ "["+ str(N)+ "]")

        line = hexf[selected[j]][N][:-1].split() # remove '\n'
        while len(write_str) < size and line:
          write_str.append(line.pop(0))

        if len(write_str) == size:
          break


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
