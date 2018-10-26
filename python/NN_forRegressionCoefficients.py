
import os

import pandas as pd
from data_collection import *
import numpy as np

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from keras import Sequential
import xlrd
from pandas import ExcelWriter
from pandas import ExcelFile
cwd = os.getcwd()
print(cwd)
df =pd.read_excel(r'data\raw\17.xlsx', sheetname='X1')
a = np.array(df)
a=a[:,2:]
print(df)
print(a)