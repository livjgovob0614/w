import sys, os, glob
import subprocess

ELSE = 0
STRING = 1
ARRAY = 2
JT = 3
EXPORT = 4


def main(argv):
  #if len(argv) < 2:
  #  print ("python3 fn.py")

  os.chdir("/home/donghoon/ssd/jg/20/input/TypeandF/")

  #flist = ["LPdata.list"]
  #flist = ["string.list", "array.list", "else.list", "function_onlySub.list"]
  flist = ["string.list", "array.list", "else.list"]
  outf = ["string", "array", "else"]
  z = -1
  for i in flist:
    z += 1
    f = open(i, 'r')
    adrlist = f.readlines()
    f.close()

    ty = i.split('.')[0]

    # XXX out_f = open(ty+"_hex.list", 'w')
    out_f = open(outf[z], 'w')

    j = 0
    while j < len(adrlist):
      if not "***" in adrlist[j]:
        print ("NO *** in line")
        sys.exit()

      fn = adrlist[j].split()[1]
      print (ty)
      if not "function" in ty and not "LPdata" in ty:
        fn = fn[:-4]
      print (" *** Getting "+ty+" type hex codes from " + fn + " *** ")
      fpath = "/home/jg/kby/disasm/output/200712_232705/"+fn+"/hexText"
      f = open(fpath, 'r')
      hex_f = f.readlines()
      f.close()

      hexlist = []

      j += 1
      while j < len(adrlist) and not "***" in adrlist[j]:
        hexlist = []
        spl = adrlist[j].split()
        startAdr = int(spl[0], 16)
        s_line = int(startAdr / 4)
        s_idx = (startAdr % 4)

        e_line = 0
        e_idx = 0
        size = 100
        if not "function" in ty and not "LPdata" in ty:
          size = int(spl[1])
          # TODO
          if size > 100:
            size = 100
        e_line = s_line + int((size-1) / 4) + int((s_idx+(size-1)%4) / 4)
        e_idx = (s_idx + (size-1)%4) % 4
        #else:
        #  e_line = s_line + 25
        #  e_idx = s_idx

        # test
        #print ("**** startAdr:", startAdr, "   size:", size)
        out_f.write("fn:" + fn + "startAdr:" + str(hex(startAdr)) + ",  size:" + str(size) + "\n")
        hexBytes = ""
        for k in range(e_line - s_line + 1):
          s, e = 0, 4
          if k == 0:
            s = s_idx
          elif k == e_line - s_line:
            e = e_idx + 1
          #if fn == "adrlist[j].split()[1][:-4]":
          #  print ("s_line:" + str(s_line))
          ### test ###
          # test
          #print ("cor hex line:", hex_f[s_line+k])
          out_f.write(hex_f[s_line+k])



#####################
          hexBytes = "".join(hex_f[s_line+k].split()[1:3])
          for h in range(s, e):
            hexlist.append(hexBytes[h*2:h*2+2])

        
        # TODO XXX 100 bytes =====> nono match with size T.T
        '''
        for k in range(26):
          s, e = 0, 4
          if k == 0:
            s = idx
          elif k == 25:
            e = idx
          hexBytes = "".join(hex_f[line].split()[1:3])
          for h in range(s, e):
            hexlist.append(hexBytes[h*2:h*2+1])
          line += 1
        '''

        # XXX out_f.write(" ".join(hexlist) + "\n")
        j += 1
    out_f.close()



if __name__ == '__main__':
    main(sys.argv)
