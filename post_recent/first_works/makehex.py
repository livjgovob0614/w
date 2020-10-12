import sys, os, glob

os.chdir("/home/donghoon/ssd/jg/coreutils/opt0_bin")
file_list=glob.glob("/home/donghoon/ssd/jg/coreutils/opt0_bin/*")



for d in dir_list:
  new_list = glob.glob(new_so+d+"/*.so")
  for new in new_list:
    new_n = os.path.basename(new)
    os.system("aobjdump -D "+new+" >> "+new_so+"asm/"+d+"/"+new_n+".asm")
