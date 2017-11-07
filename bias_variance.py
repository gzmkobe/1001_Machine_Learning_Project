import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn.metrics as skm
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# data cleaning: 
data = pd.read_csv("/bank-additional-full.csv", sep=';', header = 0, index_col = None)
# convert categratical data into dummy
## marriage
data["marr"] = ((data["marital"] == 'married')).astype(int)
data['sgl'] = ((data["marital"] == 'single')).astype(int)
data['m_unk'] = ((data["marital"] == 'unknown')).astype(int)
## occupation -- job
data['blue'] = ((data["job"] == 'blue-collar')).astype(int)
data['tech'] = ((data["job"] == 'technician')).astype(int)
data['j_unk'] = ((data["job"] == 'unknown')).astype(int)
data['svcs'] = ((data["job"] == 'services')).astype(int)
data['mgmt'] = ((data["job"] == 'management')).astype(int)
data['ret'] = ((data["job"] == 'retired')).astype(int)
data['entr'] = ((data["job"] == 'entrepreneur')).astype(int)
data['self'] = ((data["job"] == 'self-employed')).astype(int)
data['maid'] = ((data["job"] == 'housemaid')).astype(int)
data['unemp'] = ((data["job"] == 'unemployed')).astype(int)
data['stud'] = ((data["job"] == 'student')).astype(int)
## Previous outcome of marketing campaign was a success -- poutcome='success'
data['succ'] = ((data["poutcome"] == 'success')).astype(int)
data['nonxst'] = ((data["poutcome"] == 'nonexistent')).astype(int)
## Client has a personal loan -- loan
data['l_unk'] = ((data["loan"] == 'unknown')).astype(int)
data['loans'] = ((data["loan"] == 'yes')).astype(int)
## Client has credit in default -- default 
data['def'] = ((data["default"] == 'yes')).astype(int)
data['l_unk'] = ((data["default"] == 'unknown')).astype(int)
## Client has a housing loan
data['hsng'] = ((data["housing"] == 'yes')).astype(int)
data['h_unk'] = ((data["housing"] == 'unknown')).astype(int)
## Client subscribes for a term deposit -- y
data['yy'] = ((data["y"] == 'yes')).astype(int)
# reomve the original categritcal variables in data
del data['marital']
del data['job']
del data['poutcome']
del data['loan']
del data['default']
del data['housing']
del data['y']

# data: 41188 rows × 36 columns
