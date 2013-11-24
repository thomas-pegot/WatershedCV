import csv
import tkFileDialog


import csv
from StringIO import StringIO
import textwrap


def read_csv_file():

  filename =  tkFileDialog.askopenfilename() 
  file = open(filename, "r")
  cr = csv.reader(file)
 
  return cr
 
def extract_D50(cr):
 D50 = []
 for row in cr:
  if row != []:
   if row[0] == '50':
    D50.append(row[1]);
 return D50
 
def ordonner(ordre, liste):
 if len(liste) != len(ordre):
  return -1
 return [D50[i-1] for i in ordre]
 

if __name__=="__main__":
  import pylab as pl
  cr = read_csv_file()
  D50 = extract_D50(cr)
  ordre_gravelometer = \
  [1, 11, 12, 7, 8, 2, 13, 16, 10, 14, 4, 5, 9, 3, 15, 6]
  D50_ordered = ordonner(ordre_gravelometer, D50)
  fig = pl.figure()
  ax1 = fig.add_subplot(111)
  pl.plot(D50_ordered)

  ordre = [1, 15, 6, 11, 12, 7, 8, 2, 13, 16, 10, 14, 4, 5, 9, 3]  
  Test1_D50 = [86, 28, 49, 37, 36, 44, 42, 72, 37, 26, 34, 28, 58, 77, 49, 76]
  # D50_ordered_ = ordonner(ordre, D50)
  pl.plot(Test1_D50)
  pl.title('Liste des D50')
  fig.show() 
  
  