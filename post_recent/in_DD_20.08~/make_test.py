import sys, os, glob
import subprocess

def main(argv):
  os.chdir("/home/donghoon/ssd/jg/disasm/input/")

  #flist = ["LPdata.list", "function_onlySub.list"]
  flist = ["0922_function_onlySub_and_LP.list"]
  out_f = open("test_f10.list", 'w')
  out_d = open("test_d10.list", 'w')
  for i in flist:
    f = open(i, 'r')
    adrlist = f.readlines()
    f.close()

    #ty = i.split('.')[0]

    j = 0
    while j < len(adrlist):
      if not "***" in adrlist[j]:
        print ("NO *** in line")
        sys.exit()

      fn = adrlist[j].split()[1]
      if fn == "ibOpenCv.camera.samsung.so":
        break

      print (" *** Getting F & D type hex codes from " + fn + " *** ")
      fpath = "/home/jg/kby/disasm/output/200712_232705/"+fn+"/hexText"
      f = open(fpath, 'r')
      hex_f = f.readlines()
      f.close()

      hexlist = []

      j += 1
      while j < len(adrlist) and not "***" in adrlist[j]:
        spl = adrlist[j].split()
        startAdr = int(spl[1], 16)
        s_line = int(startAdr / 4)
        s_idx = (startAdr % 4)

        # TODO
        size = 10
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

        
        # TODO XXX 100 bytes =====> nono match with size T.T
        '''
        for k in range(26):
          s, e = 0, 4
          if k == 0:
            s = idx
          elif k == 25:
            e = idx
          hexBytes = "".join(hex_f[line+k].split()[1:3])
          for h in range(s, e):
            hexlist.append(hexBytes[h*2:h*2+1])
        '''

        if spl[0] == "f":
          out_f.write(" ".join(hexlist) + "\n")
        else:
          out_d.write(" ".join(hexlist) + "\n")
        j += 1
    out_f.close()
    out_d.close()


if __name__ == '__main__':
    main(sys.argv)
