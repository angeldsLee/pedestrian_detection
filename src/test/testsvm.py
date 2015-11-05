'''
Created on 2014-5-18

@author: angelds
'''
from svmutil  import *
import numpy as np

y,x= [1,-1,1,-1], [[1,0,1], [-1,0,-1],[1,1,1],[0,0,1]]
m = svm_train(y[:3], x[:3], '-h 0')
p_label, p_acc, p_val = svm_predict(y[3:], x[3:], m)

