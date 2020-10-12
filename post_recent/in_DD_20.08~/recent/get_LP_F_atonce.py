# ............................................ #
import sys, os, glob
import subprocess

os.chdir("/home/jg/kby/disasm/output/200712_232705/")
file_list = glob.glob("/home/jg/kby/disasm/output/200712_232705/*.so")


#listf = open("/home/donghoon/ssd/jg/disasm/input/0922_function_onlySub_and_LP.list", "w")
listf = open("/home/donghoon/ssd/jg/disasm/input/FandLP/FsubAndLP_for_dnfn.list", "w")
#listf = open("test", "w")
elsef = open("/home/jg/kby/disasm/postproc/20.08~/LP_elseCase.list", "w")
test = False
for lst in file_list:
  if test:
    if os.path.basename(lst) != "android.hardware.graphics.mapper@2.0.so":
      continue
    test = False

  print ("\nGet LP & function list from:", os.path.basename(lst))

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
  end_f = True
  write_str = ""
  l = 0
  tt = 0
  f_start = ""
  fakedata_addr = ""
  count = 0
  for l in range(len(lines)):
    # Function
    if "S U B R O U T I N E" in lines[l]:
      if not end_f:
        print("last function end X at line "+str(l))
        sys.exit()

      addr = hex(int(lines[l].split()[0].split(":")[1],16))
      # XXX write_str = str(addr)[2:] + "\n"
      f_start = "f " + str(addr)[2:]
      end_f = False
    elif "End of function " in lines[l]:
      if end_f:
        print ("no function to end at line "+str(l))
        # write into error_log
        sys.exit()
      if not "(" in lines[l].split()[5]:
      #if "sub" in lines[l].split()[5][:3]:
        f_name = lines[l].split()[5]
      else:
        # TODO What if :: not exist ?
        #print (lines[l])
        write2 = False
        if len(lines[l].split("(")[0]) > len(lines[l].split("<")[0]):
          spl = lines[l].split("<")
          write2 = True
        else:
          spl = lines[l].split("(")
        f_name = "".join(spl[1:])
        spl = spl[0].split("::")
        if write2:
          f_name = spl[len(spl)-1]+"<"+f_name
        else:
          f_name = spl[len(spl)-1]+"("+f_name

      if "sub_" in f_name[:4]:
        if first:
          listf.write("*** "+os.path.basename(lst)+" ***\n")
          first = False
        listf.write(f_start + "\n")
        #size += 1
        # XXX f.write("f" + write_str)
      end_f = True


    # Data
    if checkNext:
      len_line = len(lines[l])
      if len_line < 33:
        elsef.write(str(len(write_check)))
        elsef.write(str(l)+"(size):\t"+lines[l]+"\n")

      check = lines[l][22:33]
      if "__unwind" in check:
        continue
      #elif not "loc" in check and not "try" in check and (len(lines) > 34 and not "                         " in lines[l][22:50]):
      elif not "loc" in check and not "try" in check and not "                         " in lines[l][22:50]:
# int, string.. ? TODO
        if "float" in check or (len_line > 40 and (lines[l][39:41] == 'DC' or "word" in check)):

          addr_ = str(hex(int(lines[l].split()[0].split(':')[1], 16)))
          if addr_ == fakedata_addr:
            continue
          else:
            fakedata_addr = ""
          start_addr = addr_
# ???????????
          # XXX if not addr_list or addr_list[len(addr_list)-1] != addr_:
          addr_list.append(addr_)
          write_check.append(str(hex(int(lines[l].split()[0].split(':')[1], 16))) + "\n")
# TODO
          write_check.append(lines[l-1]+lines[l])
          writeNext = True
          checkNext = False
    elif writeNext:
      if "S U B R O U T I N E" in lines[l] or "ALIGN" in lines[l]:
        writeNext = False
      elif "--------------------------------" in lines[l] or "__unwind" in lines[l]:
        writeNext = False
        checkNext = True

      # Checking collecting data whether write or not
      if not writeNext and write_check:
        if not (len(write_check) == 2 and "DCD 0\n" in write_check[1]):
          if first:
            listf.write("*** "+os.path.basename(lst)+" ***\n")
            first = False
          for w in addr_list:
            listf.write("d " + w+"\n")
            #print (w)
        write_check = []
        addr_list = []
        continue

      new_addr_ = str(hex(int(lines[l].split()[0].split(':')[1], 16)))
      # XXXif addr_ != new_addr_:
      #try:

      if addr_list[len(addr_list)-1] != new_addr_:
        if first:
          listf.write("*** "+os.path.basename(lst)+" ***\n")
          first = False
        if addr_list[len(addr_list)-1] != new_addr_:
          addr_list.append(new_addr_)
        #listf.write(addr_ + "\n")
        addr = new_addr_
      else:
        if count == 6:
          addr_list.pop()
          fakedata_addr = start_addr
          count = 0
          checkNext = True
          writeNext = False
        else:
          spl = lines[l].split()
          if spl[1] == "DCQ" and spl[2][:2] == "0x" and spl[2][-1] == ',' and len(spl) == 5:
            count += 1
          else:
            count = 0

      write_check = []
      if writeNext:
        write_check.append(lines[l])
        # ??
      #except Exception:
      #  print (lines[l])
      #  print (len(addr_list))
      #  sys.exit()
      # ??????
      #write_check = []
      #write_check.append(lines[l])
          
    elif "-----------------------------------------" in lines[l]:
      checkNext = True

    #print (addr_list[len(addr_list)-1])
    #print (new_addr_)
  #sys.exit()

elsef.close()
listf.close()
