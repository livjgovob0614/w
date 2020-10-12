import sys, os, glob
import subprocess

os.chdir("/home/jg/kby/disasm/output/200712_232705/")
file_list = glob.glob("/home/jg/kby/disasm/output/200712_232705/*.so")

#f = open("/home/jg/kby/disasm/postproc/20.08~/lst_function_in200712_0825.list", "w")
f = open("/home/jg/kby/disasm/postproc/20.08~/function_all.list", "w")
f_name_list = set()
f_name = ""
size = len(f_name_list)
for lst in file_list:
  #print ("\nGet function list of:", os.path.basename(lst))
  first = True
  #f.write("*** "+os.path.basename(lst)+" ***\n")

  f2 = open(lst+"/funcText", "r")
  lines = f2.readlines()
  f2.close()

  end_f = True
  write_str = ""
  for l in range(len(lines)):
    if "S U B R O U T I N E" in lines[l]:
      if not end_f:
        print("last function end X at line "+str(l))
        sys.exit()
      addr = hex(int(lines[l].split()[0].split(":")[1],16))
      # XXX write_str = str(l)+"\t"+str(addr)[2:]
      write_str = str(addr)[2:] + "\n"
      end_f = False
    if "End of function " in lines[l]:
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

      f_name_list.add(f_name)
      new_size = len(f_name_list)
      if size != new_size or "sub_" in f_name[:4]:
        if first:
          f.write("*** "+os.path.basename(lst)+" ***\n")
          first = False
        if "sub_" in f_name[:4]:
          new_size += 1
        size = new_size
        # XXX write_str = write_str+"\t"+f_name+"\n"
        f.write(write_str)
      elif "sub_" in f_name[:4]:
          print ("sub_dup in "+os.path.basename(lst)+": "+f_name)
      end_f = True
f.close()
print (len(f_name_list))

