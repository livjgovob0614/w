import sys, os, glob
import subprocess

os.chdir("/home/jg/kby/disasm/output/200712_232705/")
file_list = glob.glob("/home/jg/kby/disasm/output/200712_232705/*.so")

fl= []
f = open("/home/donghoon/ssd/jg/20/input/FandLP/large/f20_clfF.list", 'r')
fl.append(f.readlines())
f.close()
f = open("/home/donghoon/ssd/jg/20/input/FandLP/large/f20_clfD.list", 'r')
fl.append(f.readlines())
f.close()

fidx = 0
for f in fl:
  out_f = 0
  if fidx == 0:
    out_f = open("/home/donghoon/ssd/jg/20/input/FandLP/large/f20_clfF_op.list", 'w')
    fidx = 1
  else:
    out_f = open("/home/donghoon/ssd/jg/20/input/FandLP/large/f20_clfD_op.list", 'w')
    fdix = 2
  print (str(fidx)+"th try")

  op_list = []
  l = -1
  while l < len(f)-1:
    l += 1
    spl = f[l].split()
    st = hex(int(spl[2],16))

    fn = spl[0]
    #print (str(fidx)+"th try: "+ "** new fn: " + fn)
    f2 = open(fn+"/funcText", "r")
    lines = f2.readlines()
    f2.close

    i = -1
    while i < len(lines)-1:
      op = []
      i += 1
      spl2 = lines[i].split()
      try: 
        adr = hex(int(spl2[0].split(':')[1], 16))
      except IndexError:
        if len(lines[i]) < 3 or lines[i][:2] != "**":
          print ("error1")
          print (lines[i])
          sys.exit()
        continue

      if st == adr:
        while (True):
          save_addr= 0
          
          try:
            next_adr = hex(int(lines[i+1].split()[0].split(':')[1], 16))
          except IndexError:
            if len(lines[i+1]) < 3 or lines[i+1][:2] != "**":
              print ("error2")
              print (lines[i])
              sys.exit()
            i += 1
            continue

          # ~~~ or same, but one instr function
          if adr != next_adr or "// starts at " in lines[i+1] or "End of function" in lines[i+1]:
            break
          i += 1
          ex = False
          if i == len(lines)-1:
            while (True):
              i -= 1
              spl2 = lines[i].split()
              #if len(spl2) < 2 or lines[i][22:39] != "                 " or lines[i][39:41] == "  " or lines[i][45:51] != "      ":
              if len(spl2) < 2 or lines[i][22:39] != "                 " or lines[i][39:41] == "  ":
                ex = True
                break
          if ex:
            break
            #print ("error3")
            #print ("fn: " + fn)
            #print (hex(int(lines[i-1].split()[0].split(':')[1], 16)))
            #sys.exit()

        # Check Operation Code
        spl2 = lines[i].split()
        k = 0
        while k < 5:
          op.append(spl2[1])
          # test
          if op[-1] == ";" or "loc_" in op[-1]:
            print ("err1")
            print (fn)
            for o in op:
              print (o)
            print (lines[i])
            sys.exit()
          k += 1
          i += 1
          if i == len(lines)-1:
            break
          spl2 = lines[i].split()
          #if len(spl2) < 2 or lines[i][22:39] != "                 " or lines[i][39:41] == "  " or lines[i][45:51] != "      ":
          if len(spl2) < 2 or lines[i][22:39] != "                 " or lines[i][39:40] == " ":
            save_addr = i
            # XXX newF[k-1] = 1
            #while not (i == len(lines)-1 or len(spl2) < 2 or lines[i][22:39] != "                 " or lines[i][39:41] == "  " or lines[i][45:51] != "      "):
            while i+1 < len(lines)-1 and (len(spl2) < 2 or lines[i][22:39] != "                 " or lines[i][39:40] == " "):
              i+=1
              spl2 = lines[i].split()
            #break
            if i < len(lines) -1:
              break
        if not op:
          print ("no function start")
          sys.exit()

        add = True
        for e in op_list:
          if e == op:
            add = False
            break
        if save_addr:
          i = save_addr

        if add:
          op_list.append(op)

        # test
        if len(op) == 1:
          print ("err2")
          print (fn)
          print (op[0])
          print (lines[i])
          sys.exit()

        if l == len(f) - 1 or f[l+1].split()[0] != fn:
          break
        l += 1
        st = hex(int(f[l].split()[2],16))

  op_list.sort()
  for op in op_list:
    for n in op:
      out_f.write("{:8}".format(n))
    #out_f.write('\t\t'.join(op))
    out_f.write("\n")
  print (len(op_list))
  out_f.close()

