'''
These functions provide data for ROC and confusion matrix plots from results in txt files
It is meant to read data from text files that have been produced by classify.py
'''
from sklearn.metrics import confusion_matrix, roc_curve

def conf_matrix(data_file):
  '''
  Args: data_file path to text file
    tab delimited with a single header row then subsequent rows organized as
    image name, [n class probabilities], predicted class, actual class, correct
  Returns:
    nxn array for confusion matrix
  '''
  f = open(data_file,'r')
  f.readline() #dequeue header line
  
  y_true=[]
  y_pred=[]
  for line in f:
    y_true.append(int(line.split()[-2]))
    y_pred.append(int(line.split()[-3]))
  return confusion_matrix(y_true,y_pred)

def roc_curves(data_file):
  '''
  Args: data_file path to text file (see args for conf_matrix for specifics)
  Returns: list of n roc curves
    each roc curve has 2 elements, the fprs and tprs
  '''

  f = open(data_file, 'r')
  header = f.readline() 
  header_items = header.split()
  num_classes = len(header_items) - 4

  roc_curves = []
  y_trues = [[] for i in range(num_classes)]
  y_scores = [[] for i in range(num_classes)]

  for line in f:
    split=line.split()
    true = int(split[-2])
    for c in range(num_classes):
      y_trues[c].append(1 if true==c else 0)
      y_scores[c].append(float(split[c+1]))
  
  for c in range(num_classes):
    roc_curves.append(roc_curve(y_trues[c], y_scores[c]))
  
  return roc_curves
