import sys, os, glob
import subprocess

def main(argv):

  if len(argv) < 2:
    print ("Usage: python3 make_out..py byteSize")
    sys.exit()

  in_path = "/home/donghoon/ssd/jg/20/input/"
  os.chdir(in_path)

  N = argv[1]
  flist = ["jt.list"]
  # TODO: If exist, exit
  #out_f = open("1009test_jt"+argv[1]+"_hex.list", 'w')
  #anal_f = open("1009test_jt"+argv[1]+"_hex.anal", 'w')
  out_f = open("jt"+N+".list", 'w')
  anal_f=open("jt"+N+".anal", 'w')

  first = True
  for i in flist:
    f = open(i, 'r')
    adrlist = f.readlines()
    f.close()

    j = 0
    while j < len(adrlist):
      #if not "***" in adrlist[j]:
      #  print ("NO *** in line")
      #  sys.exit()

      fn = adrlist[j].split()[1]
      if fn[-4:] == ".lst":
        fn = fn[:-4]
      '''
##### test
      if fn != "libc_malloc_debug.so":
        if not first:
          out_f.close()
          anal_f.close()
          sys.exit()
        j += 1
        continue
      else:
        first = False
      '''

      print (" *** Extracting jt type hex codes from " + fn + " *** ")
      fpath = "/home/jg/kby/disasm/output/200712_232705/"+fn+"/hexText"
      #fpath = "/home/donghoon/ssd/lst/"
      f = open(fpath, 'r')
      hex_f = f.readlines()
      f.close()

      hexlist = []

      j += 1
      while j < len(adrlist) and not "***" in adrlist[j]:
        hexlist = []
        spl = adrlist[j].split()

        if int(spl[1]) > int(N):
          j += 1
          continue

        startAdr = int(spl[0], 16)
        s_line = int(startAdr / 4)
        s_idx = (startAdr % 4)
        #print (adrlist[j])
        #print (startAdr)

        # TODO
        size = int(N) 
        e_line = s_line + int((size-1) / 4) + int((s_idx+(size-1)%4) / 4)
        e_idx = (s_idx + (size-1)%4) % 4

        hexBytes = ""
        for k in range(e_line - s_line + 1):
          s, e = 0, 4
          if k == 0:
            s = s_idx
          elif k == e_line - s_line:
            e = e_idx + 1
          hexBytes = "".join(hex_f[s_line+k].split()[1:3])
          for h in range(s, e):
            hexlist.append(hexBytes[h*2:h*2+2])
            #print (hex_f[s_line+k])
            #sys.exit()
            #print (hexBytes[h*2:h*2+2])

        
        out_f.write(" ".join(hexlist) + "\n")
        anal_f.write(fn + " " + adrlist[j])
        j += 1
    out_f.close()
    anal_f.close()


if __name__ == '__main__':
    main(sys.argv)
