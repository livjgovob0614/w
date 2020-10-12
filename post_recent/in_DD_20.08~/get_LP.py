import sys, os, glob
import subprocess

os.chdir("/home/jg/kby/disasm/output/200712_232705/")
file_list = glob.glob("/home/jg/kby/disasm/output/200712_232705/*.so")

f = open("/home/jg/kby/disasm/postproc/20.08~/LP.list", "w")
#listf = open("/home/jg/kby/disasm/postproc/20.08~/LPnum.list", "w")
listf = open("/home/donghoon/ssd/jg/disasm/input/FandLP/0923_getLPTest.list", "w")
elsef = open("/home/jg/kby/disasm/postproc/20.08~/LP_elseCase.list", "w")
test = False
ttest = False
for lst in file_list:
#  if not ttest:
#    if os.path.basename(lst) != "librs_jni.so":
#      continue
#    else:
#      ttest = True
#  print (os.path.basename(lst))
  
#  if not test:
#    if os.path.basename(lst) != "libBarcode.camera.samsung.so":
#      continue
#    test = True

  print ("\nGet LP list of:", os.path.basename(lst))
  #f.write("*** "+os.path.basename(lst)+" ***\n")
  #elsef.write("*** "+os.path.basename(lst)+" ***\n")
  #listf.write("*** "+os.path.basename(lst)+" ***\n")

  f2 = open(lst+"/funcText", "r")
  lines = f2.readlines()
  f2.close()

  first = True
  checkNext = False
  writeNext = False
  write_check = []
  addr_ = ""
  new_addr_ = ""
  addr_list = []
  for l in range(len(lines)):
    if checkNext:
      if ttest:
        adr = str(hex(int(lines[l].split()[0].split(':')[1], 16)))
        if adr == "0x9260":
          print ("0x9260")
      len_line = len(lines[l])
      if len_line < 33:
      #if len_line < 33 and not "ALIGN" in lines[l][49:45]:
        elsef.write(write_check[0])
        elsef.write(str(l)+"(size):\t"+lines[l]+"\n")
        # continue x??

      check = lines[l][22:33]
      if "__unwind" in check:
        continue
      elif not "loc" in check and not "try" in check and not "                         " in lines[l][22:50]:
#        print (str(l))
# int, string.. ? TODO
        if "float" in check or (len_line > 40 and (lines[l][39:41] == 'DC' or "word" in check)):

          addr_ = str(hex(int(lines[l].split()[0].split(':')[1], 16)))
          if ttest:
            print ("append: " + addr_)
          addr_list.append(addr_)
          write_check.append(str(hex(int(lines[l].split()[0].split(':')[1], 16))) + "\n")
# TODO
          #write_check.append(str(l)+"\n")
          write_check.append(lines[l-1]+lines[l])
          writeNext = True
        #elif len_line < 40 or (not "NOP" in check and not "ALIGN" in lines[l][39:45] and not ("                " in lines[l][22:39]) or "     " in lines[l][39:44]):
         # elsef.write(str(l)+"(case):\t"+lines[l])
        checkNext = False
    elif writeNext:
      if "S U B R O U T I N E" in lines[l] or "ALIGN" in lines[l]:
        writeNext = False
      elif "--------------------------------" in lines[l] or "__unwind" in lines[l]:
        writeNext = False
        checkNext = True

      if not writeNext and write_check:
        #if not (len(write_check) == 2 and "DCD 0" in write_check[1] and len(write_check[1]) < 46):
        if not (len(write_check) == 2 and "DCD 0\n" in write_check[1]):
          if first:
            f.write("*** "+os.path.basename(lst)+" ***\n")
            listf.write("*** "+os.path.basename(lst)+" ***\n")
            first = False
          f.write(addr_ + "\n")
          for w in write_check:
            f.write(w)
          if ttest:
            print ("real append")
          for w in addr_list:
            listf.write(w+"\n")
          #listf.write(write_check[0])
          #listf.write(addr_+"\n")
        write_check = []
        addr_list = []
        continue

      new_addr_ = str(hex(int(lines[l].split()[0].split(':')[1], 16)))
      # XXXif addr_ != new_addr_:
      if addr_list[len(addr_list)-1] != new_addr_:
        if first:
          f.write("*** "+os.path.basename(lst)+" ***\n")
          listf.write("*** "+os.path.basename(lst)+" ***\n")
          first = False
        f.write(addr_ + "\n")
        for w in write_check:
          f.write(w)
        if addr_list[len(addr_list)-1] != new_addr_:
          if ttest:
            print ("append: "+new_addr_)
          addr_list.append(new_addr_)
        #listf.write(addr_ + "\n")
        addr = new_addr_
        #write_check.append(new_addr_)
      write_check = []
      write_check.append(lines[l])
          
    elif "-----------------------------------------" in lines[l]:
      checkNext = True

f.close()
elsef.close()
