import sys, os, glob

os.chdir("/home/donghoon/ssd/jg/20/input/200712_232705")
file_list = glob.glob("/home/donghoon/ssd/jg/20/input/200712_232705/*.so")

for i in file_list:
  os.system("rm "+i+"/funcText")
  os.system("rm "+i+"/hexText2")
  os.system("rm "+i+"/adrp_info")
  os.system("rm "+i+"/adrp_data_target")

