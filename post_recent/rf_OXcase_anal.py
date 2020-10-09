import sys, os, glob
import subprocess
os.chdir("/home/jg/kby/disasm/output/200712_232705/")

fl= []
f = open("/home/donghoon/ssd/jg/20/learning/rf/fo", 'r')
fl.append(f.readlines())
f.close()
f = open("/home/donghoon/ssd/jg/20/learning/rf/fx", 'r')
fl.append(f.readlines())
f.close()

hexf=[]
f = open("/home/donghoon/ssd/jg/20/learning/rf/fo_hex", 'r')
hexf.append(f.readlines())
f.close()
f = open("/home/donghoon/ssd/jg/20/learning/rf/fx_hex", 'r')
hexf.append(f.readlines())
f.close()

out_f=[]
out_f.append(open("/home/donghoon/ssd/jg/20/learning/rf/result/fo_op.list", 'w'))
out_f.append(open("/home/donghoon/ssd/jg/20/learning/rf/result/fx_op.list", 'w'))

fidx = 0
for f in fl:
  if fidx == 0:
    fidx = 1
  else:
    fidx = 2
  print (str(fidx)+"th try")

  op_list = []
  l = -1
  while l < len(f)-1:
    l += 1
    spl = f[l].split()
    st = hex(int(spl[2],16))

    fn = spl[0]
    print ("** new fn: " + fn)
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
          #if fn == "libdepthcam3dmodeling_algorithm.arcsoft.so":
          #  print (lines[i])
          if adr != next_adr or "// starts at " in lines[i+1] or "End of function" in lines[i+1]:
            #if fn == "libdepthcam3dmodeling_algorithm.arcsoft.so":
            #  print ("out1")
            #  print (lines[i])
            #  print (lines[i+1])
            break
          i += 1
          ex = False
          ## if last line, -- ##
          if i == len(lines)-1:
            while (True):
              i -= 1
              spl2 = lines[i].split()
              #if len(spl2) < 2 or lines[i][22:39] != "                 " or lines[i][39:41] == "  " or lines[i][45:51] != "      ":
              #### test... "not" correct? ####
              if not (len(spl2) < 2 or lines[i][22:39] != "                 " or ((lines[i][39:41] != "DC" or lines[i][39:44] != "ALIGN") and lines[i][39:40] == " ") or "EXPORT" in lines[i][39:45]) or "WEAK" in lines[i][39:43]:
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
        DCB, DCD, DCQ = 0, 0, 0
        while k < 5:
          op.append(spl2[1])
          # test
          #if op[-1] == ";" or "loc_" in op[-1]:
          '''
          if op[-1] == "ALIGN":
            print ("err1")
            print (fn)
            for o in op:
              print (o)
            print (lines[i])
            sys.exit()
          '''
          k += 1


          # *******************************************************
          # ****************** DC & ALIGN CHECK *******************
          # *******************************************************
          # if previous operand == DCB: 4 to 1, DCD: 1 to 1, DCQ: 1 to 2
          if "ALIGN" in op[-1]:
            diff = int(lines[i+1].split()[0].split(':')[1],16) - int(lines[i].split()[0].split(':')[1],16)
            if diff < 4:
              print("err101")
              sys.exit()
            if diff % 4:
              print("err102")
              sys.exit()
            op.pop()
            k -= 1
            for dd in range(int(diff/4)):
              # XXX op.append(''.join(hexf[fidx-1][l].split()[4*(k-1):4*k]))
              op.append(''.join(hexf[fidx-1][l].split()[4*k:4*(k+1)]))
              k += 1
              if k == 5:
                break

          if "DC" in op[-1]:
            spl3 = lines[i].split(',')
            spl4 = lines[i].split()
            add = 0
            if lines[i].split()[1][:2] != "DC":
              add_ = 1
            if op[-1] == "DCB":
              if DCB:
                assert(len(spl3) == 0, "err6, fn:" + fn + "\nlines: " + lines[i])
                if DCB < 4:
                  DCB_str += hexf[fidx-1][l].split()[4*(k-1) + DCB]
                  if DCB == 4:
                    DCB = 0
                    op[-1] == DCB_str
                    DCB_str = ""
                  else:
                    DCB += 1
                    k -= 1
                else:
                  assert(False, "err7")
                  #print ("err6")
                  #sys.exit()
              elif len(spl3) == 1:
                DCB_str += hexf[fidx-1][l].split()[4*(k-1)]
                DCB += 1
                k -= 1
              elif len(spl3) == 4:
                op[-1] = ''.join(hexf[fidx-1][l].split()[4*(k-1):4*k])
              else:
                print ("err8")
                sys.exit()
            elif op[-1] == "DCD":
              if len(spl3) == 1:
                op[-1] = ''.join(hexf[fidx-1][l].split()[4*(k-1):4*k])
              elif len(spl3) == 4:
                print ("err100")
                sys.exit()
              else:
                print ("err9")
                sys.exit()
            # DCQ #
            else:
              if len(spl3) == 1:
                op[-1] = ''.join(hexf[fidx-1][l].split()[4*(k-1):4*k])
                if k < 5:
                  k += 1
                  op.append(''.join(hexf[fidx-1][l].split()[4*(k-1):4*k]))
              else:
              #elif len(spl3) == 4:
                op.pop()
                k -= 1
                while k < 5:
                  op.append(''.join(hexf[fidx-1][l].split()[4*k:4*(k+1)]))
                  k += 1

          i += 1
          if i == len(lines)-1:
            break



          # ***********************************************************************
          # ****************** if function end, find next instr *******************
          # ***********************************************************************
          spl2 = lines[i].split()
          #if len(spl2) < 2 or lines[i][22:39] != "                 " or lines[i][39:41] == "  " or lines[i][45:51] != "      ":
          if len(spl2) < 2 or lines[i][22:39] != "                 " or ((lines[i][39:41] != "DC" or lines[i][39:44] != "ALIGN") and lines[i][39:40] == " ") or "EXPORT" in lines[i][39:45]  or "WEAK" in lines[i][39:43]:
            if not save_addr:
              save_addr = i
            # XXX newF[k-1] = 1
            #while not (i == len(lines)-1 or len(spl2) < 2 or lines[i][22:39] != "                 " or lines[i][39:41] == "  " or lines[i][45:51] != "      "):
            while i+1 < len(lines)-1 and (len(spl2) < 2 or lines[i][22:39] != "                 " or ((lines[i][39:41] != "DC" or lines[i][39:44] != "ALIGN") and lines[i][39:40] == " ") or "EXPORT" in lines[i][39:45])  or "WEAK" in lines[i][39:43]:
              i+=1
              spl2 = lines[i].split()
            '''
            if "ALIGN" in lines[i]:
              print ("********ALIGN")
              if (lines[i][39:40] != " "):
                print ("33")
                print (lines[i][39:40])
                sys.exit()
              if (lines[i][39:41] != "DC"):
                print ("11")
                sys.exit()
              if (lines[i][39:41] != "DC" and lines[i][39:40] == " "):
                print ("22")
                sys.exit()
            '''
            #break
            if i+1 == len(lines) -1:
              break
        if not op:
          print ("no function start")
          sys.exit()

        # test
        if len(op) == 1 and i+1 < len(lines)-1:
          print ("err2")
          print (fn)
          print (op[0])
          print (lines[i])
          sys.exit()

        if k != 5:
          if i+1 < len(lines)-1:
            print("err102")
            sys.exit()
          while k < 5:
            op.append(''.join(hexf[fidx-1][l].split()[4*k:4*(k+1)]))
            k += 1


        add = True
        for e in op_list:
          if e[:-1] == op:
            add = False
            break
        if save_addr:
          i = save_addr

        if add:
          op.append("( "+fn + ", " + st + " )")
          op_list.append(op)

        if l == len(f) - 1 or f[l+1].split()[0] != fn:
          break
        l += 1
        st = hex(int(f[l].split()[2],16))
        #if fn == "libdepthcam3dmodeling_algorithm.arcsoft.so":
        #  print ("new st:", st)

  op_list.sort()
  for op in op_list:
    for n in op:
      if n == op[-1]:
        out_f[fidx-1].write("     "+n)
      else:
        out_f[fidx-1].write("{:10}".format(n))
    #out_f.write('\t\t'.join(op))
    out_f[fidx-1].write("\n")
  print (len(op_list))
  out_f[fidx-1].close()

