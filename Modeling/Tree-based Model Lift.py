
# coding: utf-8

# # Data Pre-processing

# In[15]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[16]:

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


# In[18]:

# data['month'].unique()
# # data['contact'].unique()
# # data['campaign'].unique()
# # data['pdays'].unique()
# # data['education'].unique()
# # plt.hist(data['age'])
# # plt.show()
# data['day_of_week'].unique()
# data.describe()
data = cleanData_pdays_original("F:/Study/study/Intro to Data Science/project/bank-additional/bank-additional-full.csv")


# In[19]:

data.columns
droplist = ['duration']
data = data.drop(droplist, axis =1)


# # Decision Tree

# In[20]:

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from mlxtend.evaluate import lift_score

# Decision trees often perform well on imbalanced datasets because their hierarchical structure allows them to learn signals from both classes.

 

X = data.drop('yy',1)
y = data['yy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def testTrees(X_train, y_train, X_test, y_test, dep, leaf):
    clf = DecisionTreeClassifier(criterion='entropy', min_samples_leaf = leaf, max_depth = dep)
    df = clf.fit(X_train, y_train)
    return lift_score(y_test, df.predict_proba(X_test)[:,1])


# ## Baseline decision tree

# In[21]:

clf =DecisionTreeClassifier(criterion = 'entropy')
df = clf.fit(X_train, y_train)
lift_score(y_test, df.predict_proba(X_test)[:,1])


# ## Tuning parameters to have optimal AUC

# In[22]:

########################################## Cross validation and grid search 
from sklearn.cross_validation import KFold
X = data.drop('yy',1)
y = data['yy']
kfolds = KFold(data.shape[0], n_folds = 5, shuffle = True)

param_grid = {'max_depth': [5,10,15], 'min_samples_split': np.arange(800,1300,step = 100), 
              'min_samples_leaf': np.arange(100,170, step = 10)}

lift_scorer = make_scorer(lift_score)  ### used for Gridsearchcv
dt_grid_search = GridSearchCV(DecisionTreeClassifier(criterion = 'entropy'), 
                              param_grid, cv = kfolds, n_jobs = 8, scoring = lift_scorer)
dt_grid_search.fit(X, y)

print(dt_grid_search.best_score_, dt_grid_search.best_params_)


# In[ ]:

dt_grid_search.cv_results_


# ## Choice 1: feature importance made by decision tree with its optimal parameters 

# In[23]:

# Chosse minimum leaf size as 100, max depth = 15, and splits as 1200 where AUC is at its maximum
clf = DecisionTreeClassifier(criterion='entropy', min_samples_split= 900, min_samples_leaf= 100, max_depth =  5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df = clf.fit(X_train, y_train)



fig, ax = plt.subplots(figsize=(10,3.5))
ax.bar(np.arange(40), df.feature_importances_, color = 'r')
ax.set_xticks(np.arange(len(df.feature_importances_)))
ax.set_xticklabels(X.columns.values, rotation = 90)
plt.title('Feature Importance from DT')
ax.set_ylabel('Normalized Gini Importance')
plt.show()


# In[ ]:

#### The most important feature is duration which make sense because more contact duration would potential lead to higher return value, but the econd feature 
#### which is employed which is hard to get during that quarter


# ## Choice 2: feature importance with feature selection whose feature importance > 0 

# In[54]:

# feature seletion for importance > 0

truncated_col = X.columns.values[list(df.feature_importances_ != 0)]
truncated_data= data.drop('yy',1)[truncated_col]

X_trun = truncated_data
y_trun = data['yy']



# In[55]:

param_grid = {'max_depth': [5,10,15,20], 'min_samples_split': np.arange(800,1300,step = 100), 
              'min_samples_leaf': np.arange(100,170, step = 10)}

kfolds2 = KFold(truncated_data.shape[0], n_folds = 5, shuffle = True)

dt_grid_search2= GridSearchCV(DecisionTreeClassifier(criterion = 'entropy'), param_grid, cv = kfolds2, n_jobs = 8, scoring = lift_scorer)
dt_grid_search2.fit(X_trun, y_trun)

dt_grid_search2.best_score_, dt_grid_search2.best_params_


# AUC Above increases a little bit

# ## Choice 3: Adding one feature at a time from 0 until 10 features added 

# In[56]:

# Feature selection with top 10 features selected

a = df.feature_importances_
b = sorted(range(len(a)), key=lambda i:a[i], reverse=True)[0:1]

truncated_col3 = X.columns.values[b]
truncated_data3= data.drop('yy',1)[truncated_col3]

X_trun3 = truncated_data3
y_trun3 = data['yy']


kfolds3 = KFold(truncated_data3.shape[0], n_folds = 5, shuffle = True)

dt_grid_search3= GridSearchCV(DecisionTreeClassifier(criterion = 'entropy'), param_grid, cv = kfolds3, n_jobs = 8, scoring = lift_scorer)
dt_grid_search3.fit(X_trun3, y_trun3)

dt_grid_search3.best_score_, dt_grid_search3.best_params_


# In[57]:

a = df.feature_importances_
b = sorted(range(len(a)), key=lambda i:a[i], reverse=True)[0:2]

truncated_col3 = X.columns.values[b]
truncated_data3= data.drop('yy',1)[truncated_col3]

X_trun3 = truncated_data3
y_trun3 = data['yy']


kfolds3 = KFold(truncated_data3.shape[0], n_folds = 5, shuffle = True)

dt_grid_search3= GridSearchCV(DecisionTreeClassifier(criterion = 'entropy'), param_grid, cv = kfolds3, n_jobs = 8, scoring = lift_scorer)
dt_grid_search3.fit(X_trun3, y_trun3)

dt_grid_search3.best_score_, dt_grid_search3.best_params_


# In[58]:

a = df.feature_importances_
b = sorted(range(len(a)), key=lambda i:a[i], reverse=True)[0:3]

truncated_col3 = X.columns.values[b]
truncated_data3= data.drop('yy',1)[truncated_col3]

X_trun3 = truncated_data3
y_trun3 = data['yy']


kfolds3 = KFold(truncated_data3.shape[0], n_folds = 5, shuffle = True)

dt_grid_search3= GridSearchCV(DecisionTreeClassifier(criterion = 'entropy'), param_grid, cv = kfolds3, n_jobs = 8, scoring = lift_scorer)
dt_grid_search3.fit(X_trun3, y_trun3)

dt_grid_search3.best_score_, dt_grid_search3.best_params_


# In[59]:

a = df.feature_importances_
b = sorted(range(len(a)), key=lambda i:a[i], reverse=True)[0:4]

truncated_col3 = X.columns.values[b]
truncated_data3= data.drop('yy',1)[truncated_col3]

X_trun3 = truncated_data3
y_trun3 = data['yy']


kfolds3 = KFold(truncated_data3.shape[0], n_folds = 5, shuffle = True)

dt_grid_search3= GridSearchCV(DecisionTreeClassifier(criterion = 'entropy'), param_grid, cv = kfolds3, n_jobs = 8, scoring = lift_scorer)
dt_grid_search3.fit(X_trun3, y_trun3)

dt_grid_search3.best_score_, dt_grid_search3.best_params_


# In[60]:

a = df.feature_importances_
b = sorted(range(len(a)), key=lambda i:a[i], reverse=True)[0:5]

truncated_col3 = X.columns.values[b]
truncated_data3= data.drop('yy',1)[truncated_col3]

X_trun3 = truncated_data3
y_trun3 = data['yy']


kfolds3 = KFold(truncated_data3.shape[0], n_folds = 5, shuffle = True)

dt_grid_search3= GridSearchCV(DecisionTreeClassifier(criterion = 'entropy'), param_grid, cv = kfolds3, n_jobs = 8, scoring = lift_scorer)
dt_grid_search3.fit(X_trun3, y_trun3)

dt_grid_search3.best_score_, dt_grid_search3.best_params_


# In[61]:

a = df.feature_importances_
b = sorted(range(len(a)), key=lambda i:a[i], reverse=True)[0:6]

truncated_col3 = X.columns.values[b]
truncated_data3= data.drop('yy',1)[truncated_col3]

X_trun3 = truncated_data3
y_trun3 = data['yy']


kfolds3 = KFold(truncated_data3.shape[0], n_folds = 5, shuffle = True)

dt_grid_search3= GridSearchCV(DecisionTreeClassifier(criterion = 'entropy'), param_grid, cv = kfolds3, n_jobs = 8, scoring = lift_scorer)
dt_grid_search3.fit(X_trun3, y_trun3)

dt_grid_search3.best_score_, dt_grid_search3.best_params_


# In[62]:

a = df.feature_importances_
b = sorted(range(len(a)), key=lambda i:a[i], reverse=True)[0:7]

truncated_col3 = X.columns.values[b]
truncated_data3= data.drop('yy',1)[truncated_col3]

X_trun3 = truncated_data3
y_trun3 = data['yy']


kfolds3 = KFold(truncated_data3.shape[0], n_folds = 5, shuffle = True)

dt_grid_search3= GridSearchCV(DecisionTreeClassifier(criterion = 'entropy'), param_grid, cv = kfolds3, n_jobs = 8, scoring = lift_scorer)
dt_grid_search3.fit(X_trun3, y_trun3)

dt_grid_search3.best_score_, dt_grid_search3.best_params_


# In[63]:

a = df.feature_importances_
b = sorted(range(len(a)), key=lambda i:a[i], reverse=True)[0:8]

truncated_col3 = X.columns.values[b]
truncated_data3= data.drop('yy',1)[truncated_col3]

X_trun3 = truncated_data3
y_trun3 = data['yy']


kfolds3 = KFold(truncated_data3.shape[0], n_folds = 5, shuffle = True)

dt_grid_search3= GridSearchCV(DecisionTreeClassifier(criterion = 'entropy'), param_grid, cv = kfolds3, n_jobs = 8, scoring = lift_scorer)
dt_grid_search3.fit(X_trun3, y_trun3)

dt_grid_search3.best_score_, dt_grid_search3.best_params_


# ## Optimal Lift

# In[24]:

df.feature_importances_


# In[26]:

a = df.feature_importances_
b = sorted(range(len(a)), key=lambda i:a[i], reverse=True)[0:9]

truncated_col3 = X.columns.values[b]
truncated_data3= data.drop('yy',1)[truncated_col3]

X_trun3 = truncated_data3
y_trun3 = data['yy']


kfolds3 = KFold(truncated_data3.shape[0], n_folds = 5, shuffle = True)

dt_grid_search3= GridSearchCV(DecisionTreeClassifier(criterion = 'entropy'), param_grid, cv = kfolds3, n_jobs = 8, scoring = lift_scorer)
dt_grid_search3.fit(X_trun3, y_trun3)

print(truncated_col3)
dt_grid_search3.best_score_, dt_grid_search3.best_params_


# In[65]:

a = df.feature_importances_
b = sorted(range(len(a)), key=lambda i:a[i], reverse=True)[0:10]

truncated_col3 = X.columns.values[b]
truncated_data3= data.drop('yy',1)[truncated_col3]

X_trun3 = truncated_data3
y_trun3 = data['yy']


kfolds3 = KFold(truncated_data3.shape[0], n_folds = 5, shuffle = True)

dt_grid_search3= GridSearchCV(DecisionTreeClassifier(criterion = 'entropy'), param_grid, cv = kfolds3, n_jobs = 8, scoring = lift_scorer)
dt_grid_search3.fit(X_trun3, y_trun3)

dt_grid_search3.best_score_, dt_grid_search3.best_params_


# # SPACE
# # SPACE
# # SPACE

# ## Export decision tree graph

# In[66]:

from sklearn.tree import export_graphviz
from sklearn import tree
import pydotplus
from sklearn.externals.six import StringIO

X_train, X_test, y_train, y_test = train_test_split(X_trun3, y_trun3, test_size=0.2, random_state=42)
clf_last = DecisionTreeClassifier(criterion='entropy', min_samples_split= 900, min_samples_leaf= 100, max_depth =  5)
clf_last.fit(X_train, y_train)

dot_data = StringIO()
tree.export_graphviz(clf_last,out_file=dot_data,feature_names = X_trun3.columns)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue()).write_pdf("dtree7.pdf")


# # Random Forest

# ## Base model

# In[27]:

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

rclf = RandomForestClassifier(n_estimators=500,criterion='entropy') # default max_features is the square root of the number of features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rclf.fit(X_train, y_train)
lift_score(y_test, rclf.predict_proba(X_test)[:,1])


# In[69]:

importances = rclf.feature_importances_
indices = np.argsort(importances)
cols = list(X.columns)
cols = [cols[x] for x in indices]
plt.figure(figsize=(10,6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), cols)
plt.xlabel('Relative Importance')
plt.show()


# ## Tuning parameters with 5 features selected (Best lift)

# In[29]:

### Add max_feature = 5 to match the decison tree result to see if the AUC can be improved
parameters = {
     'n_estimators':[500],
     'max_depth':(5,10, 15, 20),
     'min_samples_split':[2],
     'min_samples_leaf':(60, 90,120,150,180)
}


k_folds = KFold(data.shape[0], n_folds = 5,shuffle = True)
clf = GridSearchCV(RandomForestClassifier(max_features=5,criterion='entropy'), parameters, cv=kfolds, n_jobs=8, scoring = lift_scorer)
clf.fit(X, y)
clf.best_score_, clf.best_params_


# In[30]:

## fitting the optimal model and check the relative feature importance

rclf = RandomForestClassifier(n_estimators=500,criterion='entropy', max_depth=5, min_samples_leaf= 180, min_samples_split= 2, max_features=5) 
# default max_features is the square root of the number of features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rclf.fit(X_train, y_train)


probs_rclf = rclf.predict_proba(X_test)
preds_rclf = probs_rclf[:,1]
fpr_rclf, tpr_rclf, threshold_rclf = metrics.roc_curve(y_test, preds_rclf)
AUC_rclf = metrics.auc(fpr_rclf, tpr_rclf)
AUC_rclf

importances = rclf.feature_importances_
indices = np.argsort(importances)
cols = list(X.columns)
cols = [cols[x] for x in indices]
plt.figure(figsize=(10,6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), cols)
plt.xlabel('Relative Importance')
plt.show()


a = rclf.feature_importances_
b = sorted(range(len(a)), key=lambda i:a[i], reverse=True)[0:5]

truncated_col3 = X.columns.values[b]
truncated_col3


# In[32]:

importances


# In[49]:

X.columns


# In[52]:

print(truncated_col3)
print(sorted(importances,reverse = True)[0:10])


# ## Tuning parameters with 6 features selected

# In[73]:

### First time to tune parameters with 6 features selected 

parameters = {
     'n_estimators':[500],
     'max_depth':(5,10, 15, 20),
     'min_samples_split':[2],
     'min_samples_leaf':(60, 90,120,150,180)
}


k_folds = KFold(data.shape[0], n_folds = 5,shuffle = True)
clf = GridSearchCV(RandomForestClassifier(max_features=6, criterion='entropy'), parameters, cv=kfolds, n_jobs=8, scoring = lift_scorer, random_state =42)
clf.fit(X, y)
clf.best_score_, clf.best_params_


# ## Tuning parameters with 7 features selected

# In[74]:

### Add max_feature = 7 to match the decison tree result to see if the AUC can be improved
parameters = {
     'n_estimators':[500],
     'max_depth':(5, 10, 15, 20),
     'min_samples_split':[2],
     'min_samples_leaf':(60, 90,120,150,180)
}


k_folds = KFold(data.shape[0], n_folds = 5,shuffle = True)
clf = GridSearchCV(RandomForestClassifier(criterion='entropy'), parameters, cv=kfolds, n_jobs=8, scoring = lift_scorer, random_state =42)
clf.fit(X, y)
clf.best_score_, clf.best_params_


# ## Tuning parameters with 8 features selected

# In[75]:

### Add max_feature = 8 to match the decison tree result to see if the AUC can be improved
parameters = {
     'n_estimators':[500],
     'max_depth':(5, 10, 15, 20),
     'min_samples_split':[2],
     'min_samples_leaf':(60, 90,120,150,180)
}


k_folds = KFold(data.shape[0], n_folds = 5,shuffle = True)
clf = GridSearchCV(RandomForestClassifier(max_features=8,criterion='entropy'), parameters, cv=kfolds, n_jobs=8, scoring = lift_scorer, random_state =42)
clf.fit(X, y)
clf.best_score_, clf.best_params_


# In[76]:

### Add max_feature = 9 
parameters = {
     'n_estimators':[500],
     'max_depth':(5, 10, 15, 20),
     'min_samples_split':[2],
     'min_samples_leaf':(60, 90,120,150,180)
}


k_folds = KFold(data.shape[0], n_folds = 5,shuffle = True)
clf = GridSearchCV(RandomForestClassifier(max_features=9,criterion='entropy'), parameters, cv=kfolds, n_jobs=8, scoring = lift_scorer, random_state =42)
clf.fit(X, y)
clf.best_score_, clf.best_params_

### Increase a lot!


# ## Tuning parameters with 10 features selected

# In[77]:

### Add max_feature = 10
parameters = {
     'n_estimators':[500],
     'max_depth':(5, 10, 15, 20),
     'min_samples_split':[2],
     'min_samples_leaf':(60, 90,120,150,180)
}


k_folds = KFold(data.shape[0], n_folds = 5,shuffle = True)
clf = GridSearchCV(RandomForestClassifier(max_features=10,criterion='entropy'), parameters, cv=kfolds, n_jobs=8, scoring = lift_scorer, random_state =42)
clf.fit(X, y)
clf.best_score_, clf.best_params_

### Increase a lot!

