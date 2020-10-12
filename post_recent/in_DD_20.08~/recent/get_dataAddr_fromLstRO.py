import sys, os, glob
import subprocess

ELSE = 0
STRING = 1
ARRAY = 2
JT = 3
EXPORT = 4

#os.chdir("/home/jg/kby/disasm/output/200712_232705/")

os.chdir("/home/donghoon/ssd/lst/")
file_list = glob.glob("/home/donghoon/ssd/lst/*.lst")

out_list = []
out_list.append(open("/home/donghoon/ssd/jg/disasm/input/else.list", "w"))
out_list.append(open("/home/donghoon/ssd/jg/disasm/input/string.list", "w"))
out_list.append(open("/home/donghoon/ssd/jg/disasm/input/array.list", "w"))
out_list.append(open("/home/donghoon/ssd/jg/disasm/input/jt.list", "w"))
'''
str_f = open("/home/donghoon/ssd/jg/disasm/input/string.list", "w")
array_f = open("/home/donghoon/ssd/jg/disasm/input/array.list", "w")
jt_f = open("/home/donghoon/ssd/jg/disasm/input/jt.list", "w")
else_f = open("/home/donghoon/ssd/jg/disasm/input/else.list", "w")
'''

#test = False
for lst in file_list:
#  if not test:
#    if os.path.basename(lst) != "libBarcode.camera.samsung.so":
#      continue
#    test = True

  print ("\nGet LP list of:", os.path.basename(lst))

  f = open(lst, "r")
  lines = f.readlines()
  f.close()


  startCheck = True
  contCheck = False
  d_ty = 0
  startAddr = ""
  first_list = [True, True, True, True]
  #str_first = array_first = jt_first = else_first = True
  isROsection = False

  # of course LOAD < .rodata < .eh_frame_hdr z
  l = -1
  while l < len(lines)-1:
    l += 1
    if not isROsection:
      if not "rodata" in lines[l][:8]:
        continue
      isROsection = True
    elif not "rodata" in lines[l][:8]:
        break

    spl = lines[l].split()
    if len(spl) < 4:
      continue
    ## Checking Start Address of Data: normal data line | annotation lines | excp ##
    if startCheck:
      if "DC" in spl[2] or "EXPORT" in spl[2]:
        if "EXPORT" in spl[2]:
          d_ty = EXPORT
        elif "\"" == spl[3][0] or "asc_" == spl[1][:4] or spl[3][:6] == "0x22,\"":
#listf = open("/home/donghoon/ssd/jg/disasm/input/0922_function_onlySub_and_LP.list", "w")
          d_ty = STRING
        elif "," == spl[3][-1]: # XXX
          d_ty = ARRAY
        # XXX elif "unk_" == spl[1][:4] or "jpt_" == spl[1][:4]:
        elif "jpt_" == spl[1][:4]:
          d_ty = JT
        else:
          d_ty = ELSE
        startCheck = False
        contCheck = True
        startAddr = spl[0].split(":")[1]
      '''
      elif "DC" in spl[1]:
        print ("exception 1!")
        print (lines[l])
        sys.exit()
      '''
    ## Checking Data Size: Increment l while same address appear, except JT case ##
    elif contCheck:
      if d_ty < 3:
        try:
          curAddr = spl[0].split(":")[1]
          while startAddr == curAddr:
            l += 1
            curAddr = lines[l].split()[0].split(":")[1]
        except IndexError:
          if "ERROR" in lines[l]:
            l += 1
            curAddr = lines[l].split()[0].split(":")[1]
          else:
            print ("exception 2")
            print (lines[l])
            sys.exit()
      else:
        while startAddr == lines[l].split()[0].split(":")[1]:
          l += 1
        while lines[l][25] == ' ' and lines[l][41:43] == "DC":
          l += 1
        curAddr = lines[l].split()[0].split(":")[1]

      l -= 1
      contCheck = False
      startCheck = True
          
      if d_ty == EXPORT:
        continue
      data_size = int(curAddr, 16) - int(startAddr, 16)

      ## Add align line to data_size  ##
      if lines[l+1][41:46] == "ALIGN":
        while lines[l+1][41:46] == "ALIGN":
          l += 1
        data_size += int(lines[l+1].split()[0].split(":")[1], 16) - int(curAddr, 16)


      ## Write {start address, size} into the file correspond with data type ## 
      if first_list[d_ty]:
        out_list[d_ty].write("*** "+os.path.basename(lst)+" ***\n")
        first_list[d_ty] = False
      out_list[d_ty].write(str(hex(int(startAddr, 16))) + " " + str(data_size) + "\n")

for i in range(4):
  out_list[i].close()



