# PS2 solution
# Copy right: Chen Shao
# 1/3/2015

import sys
import numpy as np
from scipy.stats.stats import pearsonr
import random
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# set plot configuration
lineweight =  2.5
mpl.rcParams['lines.linewidth'] = lineweight
mpl.rcParams['lines.markeredgewidth'] = lineweight
mpl.rcParams['lines.markersize'] = 12
mpl.rcParams['lines.color'] = 'r'
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['font.size'] = 24
mpl.rcParams['axes.linewidth'] = lineweight
mpl.rcParams['axes.color_cycle'] = 'r, b, m, k, g, c, y'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['figure.dpi'] = 120
mpl.rcParams['figure.figsize'] = (16, 12)
mpl.rcParams['savefig.dpi'] = 120
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.2
mpl.rcParams['legend.numpoints'] = 1
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.default'] = 'regular'

class WineML:
  '''class to predict wine quality using C4.5 algo'''
  def __init__(self, fn, n):
    # define training and validation set
    self._train = [] #training data
    self._valid = []  #validation data
    self._name = [] #attribute names
    self.rules = {} #final rules generated from training set
    self.pred = []  #prediction results applying rules to validation data
    self.grade = []
    self._fn = fn
    self._numTrain = n
    self._n = 0 #number of attributes
    self._attribute = []
    self._indicator = []
    self._ct = [0]*10 #count occurrence of indicator 0~10
    self._selectedAttr = [] #idices of already selected attributes
    self._selectedCond = [] #conditions correspond to selected attribute
    self._skip = [] #attributes to be skipped due to high correlation to other attributes
    
  def readCsv(self):
    f = open(self._fn, 'r')
    tmp = f.readline()
    self._name = [x.split('"')[1] for x in tmp.split(';')]
    self._n = len(self._name)-1
    ct = 0
    for line in f:
      ct += 1
      d = line.split(';')
      r = random.random()
      if r < 0.5:#ct <= self._numTrain:
        self._train.append([float(x) for x in d])
      else:
        self._valid.append([float(x) for x in d])
    f.close()
    return
    
  def getTree(self):
    '''Generate decision tree from training data with C4.5 algo'''
    self.readCsv()  #read data file and get data sets
    
    # get attributes and indicator
    for i in range(self._n):
      self._attribute.append([x[i] for x in self._train])
    #self._indicator = [0 if x[-1]<=5 else 1 for x in self._train]
    self._indicator = [x[-1] for x in self._train]
    
    # update attributes by eliminating highly correlated attributes
    self.updateAttr()
    #print self._skip
    
    while len(self._selectedAttr) < self._n:
      self.selectAttr()
    #print self._selectedAttr, self._selectedCond
    
    # call genRules method to generate rules to be stored in dictionary
    self.genRules('', range(len(self._indicator)))
    #print self.rules
    
    # predict wine grade for validation data
    self.validate()
    self.grade = [x[-1] for x in self._valid]
    
    fn = 'result_'+str(self._numTrain)+'.csv'
    #f = open(fn, 'w+')
    ct = 0
    for i in range(len(self.pred)):
      if self.pred[i] == self.grade[i]:
        same = 1
        ct += 1
      else:
        same = 0
      #s = str(self.pred[i])+','+str(grade[i])+','+str(same)+'\n'
      #f.write(s)
    #f.close()
    return ct/float(len(self.grade))
    
  def validate(self):
    '''predict wine grade for validation data'''
    for d in self._valid:
      key = ''  #generate key
      for i in range(self._n):
        if d[self._selectedAttr[i]] <= self._selectedCond[i]:
          key += '1'
        else:
          key += '0'
      self.pred.append(self.predictGrade(key))
  
  def predictGrade(self, key):
    '''prediction of grade based on generated key'''
    n = len(key)
    if key in self.rules:
      return self.rules[key]
    else:
      if n == 1:
        self.rules[key] = -1
        return self.rules[key]
      self.rules[key] = self.predictGrade(key[:n-1])
      return self.rules[key]
  
  def updateAttr(self):
    tol = 0.9 #any attributes with higher than 0.9 correlation is considered redundent
    for i in range(self._n-1):
      for j in range(i+1,self._n):
        p = pearsonr(self._attribute[i],self._attribute[j])
        if p[0]>tol or p[0]<-tol:
          self._skip.append(j)
    return
  
  def genRules(self, s, idx):
    '''generate rules to predict wine grade from generated attribute selection list'''
    # if length of key string is self._n-1, access last attribute and put prediction to rules
    n = len(s)
    idxAttr = self._selectedAttr[n]
    val = self._selectedCond[n]
    idxLess = [i for i in idx if self._attribute[idxAttr][i] <= val]
    idxMore = [i for i in idx if self._attribute[idxAttr][i] > val]
    vLess = [self._indicator[i] for i in idxLess]
    vMore = [self._indicator[i] for i in idxMore]
    
    #print n, s, vLess, vMore
    #raw_input('------')
    
    #if reaches the last level
    if n == self._n-1:
      if len(vMore) == 0:
        self.rules[s+'1'] = round(sum(vLess)/len(vLess))
      elif len(vLess) == 0:
        self.rules[s+'0'] = round(sum(vMore)/len(vMore))
      else:
        self.rules[s+'1'] = round(sum(vLess)/len(vLess))
        self.rules[s+'0'] = round(sum(vMore)/len(vMore))
      return
    
    #if all cases in vLess is the same, return
    if len(set(vLess)) == 1 and len(set(vMore)) == 1:
      self.rules[s+'1'] = vLess[0]
      self.rules[s+'0'] = vMore[0]
      return
    elif len(set(vLess)) == 1:
      self.rules[s+'1'] = vLess[0]
      if len(vMore) > 0:
        self.genRules(s+'0',idxMore)
      else:
        self.rules[s+'0'] = self.rules[s+'1']
        return
    #if all cases in vMore is the same, return
    elif len(set(vMore)) == 1:
      self.rules[s+'0'] = vMore[0]
      if len(vLess) > 0:
        self.genRules(s+'1',idxLess)
      else:
        self.rules[s+'1'] = self.rules[s+'0']
        return
    
    #if either vLess or vMore is empty
    if len(vLess) == 0:
      self.rules[s+'1'] = round(sum(vMore)/len(vMore))  # take average of vMore
      self.genRules(s+'0',idxMore)  #<= condition as 1
    elif len(vMore) == 0:
      self.rules[s+'0'] = round(sum(vLess)/len(vLess))  # take average of vLess
      self.genRules(s+'1',idxLess)
    else:
      self.genRules(s+'0',idxMore)
      self.genRules(s+'1',idxLess)
    
    
    
  def selectAttr(self):
    '''select attributes that gives max information gain ratio from available attributes'''
    igrTmp = [0]*self._n
    idxMax = 0
    eMaxGlobal = 0
    vMaxGlobal = 0
    for i in range(self._n):
      if i in self._selectedAttr:
        continue
      minVal = min(self._attribute[i])
      maxVal = max(self._attribute[i])
      step = (maxVal-minVal)/10
      val = minVal+step
      eMax = 0
      vMax = val
      while val < maxVal:
        e = self.igr(i, val)
        if e > eMax:
          eMax = e
          vMax = val
        val += step
      if eMax > eMaxGlobal:
        idxMax = i
        eMaxGlobal = eMax
        vMaxGlobal = vMax
    #print idxMax, eMaxGlobal, vMaxGlobal
    self._selectedAttr.append(idxMax)
    self._selectedCond.append(vMaxGlobal)
    return 
  
  def entropy(self, v):
    ct = [0]*9 #wine grade from 1 to 9
    res = 0
    n = len(v)
    #print v
    #raw_input('-----')
    for i in v:
      ct[int(i)-1] += 1
    #print ct, n
    for i in range(len(ct)):
      if ct[i]>0 and ct[i]<n:
        p = ct[i]/float(n)
        res -= p*math.log(p,2)
      #print i,res
    return res
  
  def igr(self, idx, val):
    '''calculate information gain'''
    res = self.entropy(self._indicator) #total entropy
    v = self._attribute[idx]
    n = len(v)
    idxLess = [i for i in range(n) if v[i]<=val] #idices where attribute less than val
    nLess = len(idxLess)
    
    # calc ig for attribute less than val
    p = nLess/float(n)
    vLess = [self._indicator[i] for i in range(n) if i in idxLess]
    vMore = [self._indicator[i] for i in range(n) if i not in idxLess]
    #print res
    res -= p*self.entropy(vLess)
    #print res
    res -= (1-p)*self.entropy(vMore)
    
    iv = self.iv([p,1-p])
    #print res, iv
    return res/iv
    
  def iv(self, p):
    '''calculate intrinsic value'''
    res = 0
    for num in p:
      res -= num*math.log(num)
    return res
  
  def graph(self, fn1, fn2):
    plt.figure() #(num=i,figsize = (8,6))
    plt.hist(self.grade, label='Original Data')
    plt.xlabel('Wine Grade')
    plt.ylabel('Frequency')
    plt.legend(loc = 0, prop={'size':20})
    plt.tight_layout()
    plt.savefig(fn1, pad_inches = 0.2)
    
    plt.figure() #(num=i,figsize = (8,6))
    plt.hist(self.pred, label='Prediction Data')
    plt.xlabel('Wine Grade')
    plt.ylabel('Frequency')
    plt.legend(loc = 0, prop={'size':20})
    plt.tight_layout()
    plt.savefig(fn2, pad_inches = 0.2)
  
if __name__ == '__main__':
  fn_red = 'winequality-red.csv'
  fn_white = 'winequality-white.csv'
  
  rateRed = []
  rateWhite = []
  wineRed = WineML(fn_red, 100)
  rateRed.append(wineRed.getTree())
  wineWhite = WineML(fn_white, 100)
  rateWhite.append(wineWhite.getTree())
  wineRed.graph('red1.tiff','red2.tiff')
  wineWhite.graph('white1.tiff','white2.tiff')
  
    
  