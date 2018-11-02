def print_(msg):
  print(msg)
  with open('report.txt','a+') as f:
    f.writelines(str(msg)+'\n')
