import sys, os, glob
import subprocess

os.chdir("/home/donghoon/ssd/jg/coreutils/opt0/")
file_list=glob.glob("/home/donghoon/ssd/jg/coreutils/opt0/*")

#f = open("/home/jg/kby/disasm/postproc/20.08~/lst_function_in200712_0828_onlySub.list", "w")
f = open("/home/donghoon/ssd/jg/coreutils/f_onlySub_opt0.list", 'w')
size = 0
for lst in file_list:
  print ("\nGet function list from:", os.path.basename(lst))
  first = True
  #f.write("*** "+os.path.basename(lst)+" ***\n")

  f2 = open(lst, "r")
  lines = f2.readlines()
  f2.close()

  end_f = True
  write_str = ""
  text_sec = False
  for l in range(len(lines)):
    if not text_sec and lines[l].split(':')[0] != ".text":
      continue
    elif text_sec and lines[l].split(':')[0] == ".text":
      text_sec = True
    elif lines[l].split(':')[0] != ".text":
      break 

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

      if "sub_" in f_name[:4]:
        if first:
          f.write("*** "+os.path.basename(lst)+" ***\n")
          first = False
        size += 1
        # XXX write_str = write_str+"\t"+f_name+"\n"
        f.write(write_str)
      end_f = True
f.close()
print (size)
#print (len(f_name_list))

