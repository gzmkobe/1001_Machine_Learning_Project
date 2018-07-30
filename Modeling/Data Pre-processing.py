import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn.metrics as skm
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def cleanData_pdays_original(infile_name):
    data = pd.read_csv(infile_name, sep=';', header = 0, index_col = None)
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
    ## education
    data['edu_basic.4y'] = ((data["education"] == 'basic.4y')).astype(int)
    data['edu_basic.6y'] = ((data["education"] == 'basic.6y')).astype(int)
    data['edu_high.school'] = ((data["education"] == 'high.school')).astype(int)
    data['edu_professional.course'] = ((data["education"] == 'professional.course')).astype(int)
    data['edu_unknown'] = ((data["education"] == 'unknown')).astype(int)
    data['edu_basic.9y'] = ((data["education"] == 'basic.9y')).astype(int)
    data['edu_university.degree'] = ((data["education"] == 'university.degree')).astype(int)

    # reomve the original categritcal variables in data
    del data['marital']
    del data['job']
    del data['poutcome']
    del data['loan']
    del data['default']
    del data['housing']
    del data['y']
    del data['education']
    
    # Deal with aducation: concert to number: replace basic.4y w/ 4, etc.
    # data.education.replace(['basic.4y', 'high.school', 'high.school', 'basic.9y','professional.course', 'unknown', 'university.degree', 'illiterate'], [4, 12, 6, 9, 14, 'Nan', 16, 0], inplace=True)
    # Deal with Month:
    data.month.replace(['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'mar', 'apr', 'sep'], [5, 6, 7, 8, 10, 11, 12, 3, 4, 9], inplace=True)
    # Deal with day_of_week:
    data.day_of_week.replace(['mon', 'tue', 'wed', 'thu', 'fri'], [1, 2, 3, 4, 5], inplace=True)
    
    return data   
  
  # check the number of labelled '1' instances
  np.count_nonzero(data.yy)

  
  
# Data process for svm and LG: inlcuding normalizatoin and binary convert
def cleanData_pdays_adjusted(infile_name):
    cleanData_pdays_original(infile_name)
    # missing values 
    ## pdays: create a new variable nanmed 'pdays_999', given '1' if 999 in 'pdays', otherwise, give value '0'; then replace the 999 in 'pdays' with average (when calculating ave, 999 were excluded)
    data['pdays_999'] = ((data["pdays"] == 999)).astype(int)
    data_adj = data.replace(999, data['pdays'][data['pdays'] != 999].mean())
    return data_adj   
  
###################################################################################################################
# data exploration:

# correlation matrics
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")

# Compute the correlation matrix
corr = data.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()

# explore the nonlinear relationship among vairables:
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df_adj_drop.drop('yy',1), df['yy'], test_size=0.2)
import matplotlib
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve

# baseline: Considering logsictis regression model. Simply, fit LG model on training and testing data. Check the results
# A logistic regression using sklearn's linear_model.LogisticRegression() and prediction

# Simple Logistic Regression Model:
M_LR = LogisticRegression()
M_LR.fit(X_train,Y_train)
LR_probs = M_LR.predict_proba(X_test)
LR_preds = LR_probs[:,1]

fpr, tpr, thresholds=roc_curve(Y_test, LR_preds)
roc_auc=(auc(fpr, tpr))
#auc(Y_test,M_LR.predict(X_test))
roc_auc

### 0.76232183962937072
## Simple Decision tress model: 
clf = ExtraTreesClassifier()
clf = clf.fit(X_train,Y_train)
clf_probs = clf.predict_proba(X_test)
clf_preds = clf_probs[:,1]
fpr, tpr, thresholds=roc_curve(Y_test, clf_preds)
roc_auc=(auc(fpr, tpr))
#auc(Y_test,M_LR.predict(X_test))
roc_auc

### 0.7249628687733789
# AUC(LG) > AUC(DT), we conclude that there is no strong non-linear relationship between features in this data set. We are good to try LR in the next step.

## Check Train data columns (Features)
X_train.columns

## Plot bar plot for coef_ from LR:
M_LR.coef_
feature_mi_dict = dict(zip(X_train.columns.tolist(), M_LR.coef_.ravel()))
plt.rcParams['figure.figsize'] = 18, 6
plt.bar(range(len(feature_mi_dict)), feature_mi_dict.values(), align='center',color = 'red')
plt.xticks(range(len(feature_mi_dict)), feature_mi_dict.keys(),rotation=90)
plt.title('Coeffients of Baseline Model-Logistic Regression')
plt.show()


# take a look at the descriptive statistics of features:
print(data.describe())

## Cool interavtive plot of feature importance by DT: 

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

clf = ExtraTreesClassifier()
clf = clf.fit(data.drop(['yy'],axis = 1), data['yy'])
feature_mi  = clf.feature_importances_  

feature_mi_dict = dict(zip(data.drop(['yy'],axis = 1).columns.values, feature_mi)) # zip creates the tuples to put into the dictionary
feature_mi_dict
# #plot

# plt.bar(range(len(feature_mi_dict)), feature_mi_dict.values(), align='center')
# plt.xticks(range(len(feature_mi_dict)), feature_mi_dict.keys())

# plt.show()

import pandas as pd
import pygal as pg
from pygal.style import DarkStyle, DarkSolarizedStyle
from pygal import Config

#cibar_chart = pg.Bar(width=1000,spacing=50,style=DarkStyle)
cibar_chart = pg.Bar(width=800,spacing=20)
cibar_chart.title = 'Feature Importance'
# labels
cibar_chart.x_label_rotation=45 
cibar_chart.truncate_label=8
cibar_chart.human_readable = True
#cibar_chart.x_labels_major = ['Crypto','RayTrace']
cibar_chart.x_labels_major_count=5
# cibar_chart.include_x_axis=False
cibar_chart.x_labels = ['age', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'marr', 'sgl', 'm_unk', 'blue', 'tech', 'j_unk', 'svcs', 'mgmt', 'ret', 'entr', 'self', 'maid', 'unemp', 'stud', 'succ', 'nonxst', 'l_unk', 'loans', 'def', 'hsng', 'h_unk', 'edu_basic.4y', 'edu_basic.6y', 'edu_high.school', 'edu_professional.course', 'edu_unknown', 'edu_basic.9y', 'edu_university.degree', 'pdays_999']
cibar_chart.add('Feature Importance by DTs', feature_mi_dict.values())
cibar_chart.render_to_file('feature_importance_chart.svg')
print(data.describe())
