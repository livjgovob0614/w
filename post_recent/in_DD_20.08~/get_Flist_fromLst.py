import sys, os, glob
import subprocess

file_list = glob.glob("/home/donghoon/ssd/lst/*.lst")

f = open("LstFileLists", "w")
for lst in file_list:
  print ("\nGet function list of:", os.path.basename(lst))
  f.write("*** "+os.path.basename(lst)+" ***\n")

  f2 = open(lst, "r")
  lines = f2.readlines()
  f2.close()

  end_f = True
  for l in range(len(lines)):
    if "S U B R O U T I N E" in lines[l]:
      if not end_f:
        print("last function end X at line "+str(l))
        sys.exit()
      addr = hex(int(lines[l].split()[0].split(":")[1],16))
      f.write(str(l)+"\t"+str(addr)[2:]+"\n")
      end_f = False
    if "End of function " in lines[l]:
      if end_f:
        print ("no function to end at line "+str(l))
        sys.exit()
      end_f = True
f.close()

