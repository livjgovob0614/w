import sys, os, glob
import subprocess

os.chdir("/home/jg/kby/disasm/output/200712_232705/")
file_list = glob.glob("/home/jg/kby/disasm/output/200712_232705/*.so")

f = open("/home/jg/kby/disasm/postproc/20.08~/LP.list", "w")
listf = open("/home/jg/kby/disasm/postproc/20.08~/LPnum.list", "w")
elsef = open("/home/jg/kby/disasm/postproc/20.08~/LP_elseCase.list", "w")
test = False
for lst in file_list:
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
  for l in range(len(lines)):
    if checkNext:
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

          addr_ = str(hex(int(lines[l].split()[0].split(':')[1], 16))) + "\n")
          write_check.append(str(hex(int(lines[l].split()[0].split(':')[1], 16))) + "\n")
# TODO
          #write_check.append(str(l)+"\n")
          write_check.append(lines[l-1]+lines[l])
          """
          f.write("## "+str(l)+"(-1):\n")
          #f.write(lines[l-3]+lines[l-2]+lines[l-1]+lines[l])
          f.write(lines[l-1]+lines[l])
          listf.write("## "+str(l)+"\n")
          """
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
          for w in write_check:
            f.write(w)
          listf.write(write_check[0])
        write_check = []
        continue

      new_addr_ = str(hex(int(lines[l].split()[0].split(':')[1], 16))) + "\n")
      write_check.append(lines[l])
          
    elif "-----------------------------------------" in lines[l]:
      checkNext = True

f.close()
elsef.close()
