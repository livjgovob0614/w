import sys, os, glob
import subprocess

def main(argv):
  in_path = "/home/donghoon/ssd/jg/20/input/TypeandF/"

  os.chdir(in_path)

  flist = ["string.list", "array.list", "else.list"]
  outlist = ["string.anal", "array.anal", "else.anal"]

  n = -1
  for i in flist:
    n += 1
    f = open(i, 'r')
    adrlist = f.readlines()
    f.close()

    out_f = open(outlist[n], 'w')

    maxN = 0
    j = 0
    maxF = ""
    while j < len(adrlist):
      if not "***" in adrlist[j]:
        print ("NO *** in line")
        sys.exit()

      #out_f.write(adrlist[j])
      fn = adrlist[j].split()[1]
      if fn[-4:] == ".lst":
        fn = fn[:-4]

      print (" *** Extracting jt type hex codes from " + fn + " *** ")
      #fpath = "/home/jg/kby/disasm/output/200712_232705/"+fn+"/hexText"
      #f = open(fpath, 'r')
      #hex_f = f.readlines()
      #f.close()
      #hexlist = []

      j += 1
      while j < len(adrlist) and not "***" in adrlist[j]:
        spl = adrlist[j].split()
        out_f.write(fn + " " + spl[0] + "\n")



        # jt #
        '''
        if int(spl[1]) > 5000:
          spl = adrlist[j].split()
          j += 1
          continue

        out_f.write(adrlist[j])
        if int(spl[1]) > maxN:
          maxN = int(spl[1])
          maxF = fn + " " + adrlist[j]
        '''
        j += 1

  #print (maxN)
  #print (maxF)
  out_f.close



if __name__ == '__main__':
    main(sys.argv)
