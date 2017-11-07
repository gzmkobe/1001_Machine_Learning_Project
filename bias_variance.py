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
## contact method: if telephone--1, otherwise -- 0
data['contact'] = ((data["contact"] == 'telephone')).astype(int)
# reomve the original categritcal variables in data
del data['marital']
del data['job']
del data['poutcome']
del data['loan']
del data['default']
del data['housing']
del data['y']

# Deal with aducation: concert to number: replace basic.4y w/ 4, etc.
data.education.replace(['basic.4y', 'high.school', 'basic.6y', 'basic.9y','professional.course', 'unknown', 'university.degree', 'illiterate'], [4, 12, 6, 9, 14, 'Nan', 16, 0], inplace=True)
# Deal with Month:
data.month.replace(['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'mar', 'apr', 'sep'], [5, 6, 7, 8, 10, 11, 12, 3, 4, 9], inplace=True)
# Deal with day_of_week:
data.day_of_week.replace(['mon', 'tue', 'wed', 'thu', 'fri'], [1, 2, 3, 4, 5], inplace=True)

# data: 41188 rows × 36 columns

# Deal with aducation: concert to number: replace basic.4y w/ 4, etc.
data.education.replace(['basic.4y', 'high.school', 'basic.6y', 'basic.9y','professional.course', 'unknown', 'university.degree', 'illiterate'], [4, 12, 6, 9, 14, 'Nan', 16, 0], inplace=True)
# Deal with Month:
data.education.replace(['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'mar', 'apr', 'sep'], [5, 6, 7, 8, 10, 11, 12, 3, 4, 9], inplace=True)
# Deal with day_of_week:
data.education.replace(['mon', 'tue', 'wed', 'thu', 'fri'], [1, 2, 3, 4, 5], inplace=True)

## Check data: 
#pd.value_counts(data['education'])
# pd.value_counts(data['pdays'])
