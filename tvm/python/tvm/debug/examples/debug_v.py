from __future__ import print_function
import os
import inspect

def PRINT(txt="", stack_trace=2):
  if False:
    f=open("/var/log/tvm_dgb_log.txt", "a+")
    """if "" == txt:
        f.write("\n"+ str(os.path.basename(inspect.stack()[2][1])) + " : " + str(inspect.stack()[2][3]) +" : " + str(inspect.stack()[2][2]) + "\n")
    else:
        f.write("\n"+ str(os.path.basename(inspect.stack()[2][1])) + " : " + str(inspect.stack()[2][3]) +" : " + str(inspect.stack()[2][2])+" : " + txt + "\n")
    f.write("--->" + str(os.path.basename(inspect.stack()[3][1])) + " : " + str(inspect.stack()[3][3]) +" : " + str(inspect.stack()[3][2]) + "\n")
    f.write("------->" + str(os.path.basename(inspect.stack()[4][1])) + " : " + str(inspect.stack()[4][3]) +" : " + str(inspect.stack()[4][2]) + "\n")"""

    """f.write(str(inspect.stack()[2][1]) + " : " + str(inspect.stack()[2][3]) +" : " + str(inspect.stack()[2][2])+" : " + txt + "\n")"""

    f.write(str(inspect.stack()[2][1]) + " : " + str(inspect.stack()[2][3]) +" : " + str(inspect.stack()[2][2]))

    for i in range (stack_trace-1):
      if (i+1 == stack_trace-1):
        if not "" == txt:
          f.write("                     " + str(os.path.basename(inspect.stack()[3+i][1])) + " : " + str(inspect.stack()[3+i][3]) +" : " + str(inspect.stack()[3+i][2]) + " : " + txt + "\n")
        else:
          f.write("                     " + str(os.path.basename(inspect.stack()[3+i][1])) + " : " + str(inspect.stack()[3+i][3]) +" : " + str(inspect.stack()[3+i][2]) + "\n")
      else:
        f.write("                     " + str(os.path.basename(inspect.stack()[3+i][1])) + " : " + str(inspect.stack()[3+i][3]) +" : " + str(inspect.stack()[3+i][2]))
    f.close()