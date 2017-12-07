
# coding: utf-8

# ## Predict Term Deposit

# First, we need to read data from csv  into a dataframe. 

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
cwd=os.getcwd()+"/bank-additional/"
data = pd.read_csv(cwd+"bank-additional-full.csv", sep=';', header = 0, index_col = None)
data.head(5)


# ## Data Cleaning

# In[2]:

data.head(5)


# In[3]:

def cleanData_pdays_adjusted(raw_data):
    data=raw_data.copy()
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
    # missing values 
    ## pdays: create a new variable nanmed 'pdays_999', given '1' if 999 in 'pdays', otherwise, give value '0'; then replace the 999 in 'pdays' with average (when calculating ave, 999 were excluded)
    #data['pdays_999'] = ((data["pdays"] == 999)).astype(int)
    #data = data.replace(999, data['pdays'][data['pdays'] != 999].mean())
    return data   


# In[4]:

def cleanData_pdays_original(raw_data):
    data = raw_data.copy()
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


# In[5]:

data_clean_= cleanData_pdays_original(data)


# In[6]:

data_clean_.head(5)
# data_clean_['month'].value_counts()


# ## Feature Selection

# Choose features that used to predict the value 

# In[8]:

features=['age', 'contact', 'month', 'day_of_week',  'campaign',
       'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
       'euribor3m', 'nr.employed', 'marr', 'sgl', 'm_unk', 'blue', 'tech',
       'j_unk', 'svcs', 'mgmt', 'ret', 'entr', 'self', 'maid', 'unemp', 'stud',
       'succ', 'nonxst', 'l_unk', 'loans', 'def', 'hsng', 'h_unk', 'yy',
       'edu_basic.4y', 'edu_basic.6y', 'edu_high.school',
       'edu_professional.course', 'edu_unknown', 'edu_basic.9y',
       'edu_university.degree']
#'duration',
data_clean_=data_clean_[features]


# ## Rescale

# In[9]:

from sklearn.preprocessing import StandardScaler
def rescale(Dataset):
    #rescale those feature columns
    features=list(Dataset.drop("yy", axis=1, inplace=False).columns)
    scaler=StandardScaler()
    scaler.fit(Dataset[features])
    df_feature=pd.DataFrame(scaler.transform(Dataset[features]), index=Dataset[features].index, columns=features)
    df_dummy_and_target=Dataset["yy"]
    data_clean=pd.concat([df_feature, df_dummy_and_target], axis=1)
    return data_clean


# In[10]:

data_clean=rescale(data_clean_)


# In[11]:

data_clean.head(5)


# In[22]:

from sklearn.model_selection import train_test_split
X=data_clean.drop(["yy"], axis=1)
y=data_clean["yy"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[ ]:

from sklearn.svm import SVC
SVM_classifier=SVC(C=1, kernel="rbf", probability=True)
SVM_classifier.fit(X_train, y_train)


# In[21]:

type(y_test)


# In[76]:

from sklearn.metrics import roc_curve, auc
def plot_ROC(predictions, truth, label_string):
    fpr, tpr, thresholds=roc_curve(truth, predictions)
    roc_auc=(auc(fpr, tpr))
    c = (np.random.rand(), np.random.rand(), np.random.rand())
    plt.plot(fpr, tpr, color = c, label=label_string+" "+ " (AUC = %0.4f)" % roc_auc)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC")
    plt.legend(loc= "lower right")
    return [fpr, tpr, thresholds]

get_ipython().magic('matplotlib inline')
fig = plt.figure(figsize= (12, 6))
ax = plt.subplot(111)
fpr_tpr_thresholds_dict={} 

for model in Prob_estimate:
     fpr_tpr_thresholds_dict[model] = plot_ROC(Prob_estimate[model], y_test, model)
ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.15), fancybox = True, 
              shadow = True, ncol =4,  prop = {'size':10})


# ## KFold

# In[40]:

from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc

def xValSVM(dataset, label_name, k, cs):
    kfolds = KFold(n_splits = k, shuffle=True)
    aucs={}
    target_index=dataset.columns.get_loc(label_name)
    for fold in range(k):
        train_index, test_index = list(kfolds.split(dataset))[fold]
        cv_train = dataset.iloc[train_index]
        cv_validate = dataset.iloc[test_index]
        for c in cs:
            SVM_classifier=SVC(C=c, kernel="linear", probability=True)
            SVM_classifier.fit(cv_train.drop(cv_train.columns[target_index:], axis=1), cv_train[label_name])
            truth=cv_validate.iloc[:, target_index]
            predictions=SVM_classifier.predict_proba(cv_validate.drop(cv_validate.columns[target_index:], axis=1))[:, 1]
            fpr, tpr, thresholds=roc_curve(truth, predictions)
            AUC_c_k = auc(fpr, tpr)
            if (fold==0):
                aucs[c]=[AUC_c_k]
            else:
                aucs[c].append(AUC_c_k)
    return aucs


# In[41]:

from sklearn.svm import SVC
cs=[10**x for x in range(-8, 2) ]
aucs=xValSVM(data_clean, "yy", 5, cs)


# In[47]:

get_ipython().magic('matplotlib inline')
mean={}
StdErr={}
Minus_2StdErr={}
Add_2StdErr={}
max_1std=0
for c in aucs:
    mean[c]= np.mean(aucs[c])
    StdErr[c]= np.sqrt(np.var(aucs[c])/len(aucs[c]))
    Minus_2StdErr[c] = mean[c] - 2* StdErr[c]
    Add_2StdErr[c] = mean[c] + 2* StdErr[c]
max_mean_AUC =max(mean, key=mean.get)
max_1std=mean[max_mean_AUC]-StdErr[max_mean_AUC]
Log_C=list(range(-8, 2))
fig = plt.figure(figsize= (10, 6))
ax=plt.subplot(1,1,1)
plt.plot(Log_C, list(mean.values()), label="mean(AUC)")
plt.plot(Log_C, list(Minus_2StdErr.values()), "k+", label="mean(AUC)-2*stderr(AUC)")
plt.plot(Log_C, list(Add_2StdErr.values()), "k--", label="mean(AUC)+2*stderr(AUC)")
plt.axhline(y=max_1std, color="r", label="max(AUC)-1stderr(AUC)")
plt.xticks(Log_C)
plt.legend(loc= "lower right")
plt.ylim([0.90, 0.940])
plt.xlabel("Log_C")
plt.ylabel("AUC")
ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.15), fancybox = True, 
              shadow = True, ncol =4,  prop = {'size':10})


# In[43]:

aucs


# In[ ]:

from sklearn.svm import SVC
SVM_classifier=SVC(C=0.01, kernel="poly", probability=True, gamma=0.1, n_jobs = 8, verbose=100)
SVM_classifier.fit(X_data_clean, y_data_clean)


# In[18]:

from sklearn.metrics import roc_curve, auc
def plot_ROC(predictions, truth, label_string):
    fpr, tpr, thresholds=roc_curve(truth, predictions)
    roc_auc=(auc(fpr, tpr))
    c = (np.random.rand(), np.random.rand(), np.random.rand())
    plt.plot(fpr, tpr, color = c, label=label_string+" "+ " (AUC = %0.4f)" % roc_auc)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC")
    plt.legend(loc= "lower right")
    return [fpr, tpr, thresholds]

get_ipython().magic('matplotlib inline')
fig = plt.figure(figsize= (12, 6))
ax = plt.subplot(111)
fpr_tpr_thresholds_dict={} 
Prob_estimate={}
Prob_estimate["SVM C = 0.01"]=SVM_classifier.predict_proba(X_test)[:, 1]
for model in Prob_estimate:
     fpr_tpr_thresholds_dict[model] = plot_ROC(Prob_estimate[model], y_test, model)
ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.15), fancybox = True, 
              shadow = True, ncol =4,  prop = {'size':10})


# ## Grid search 

# In order to reduce the probablity of overfiting, we used gridsearchcv to find the best parameters in terms of generalization for the SVM model. 

# In[29]:

len(X_data_clean.columns)


# In[27]:

#X_train2, X_test2, y_train2, y_test2 = train_test_split(X_trun, y_trun, test_size=0.1, random_state=42)
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

X_data_clean=data_clean.drop("yy", axis=1)
y_data_clean=data_clean["yy"]

Kfold = KFold(n_splits = 5, shuffle=True)

param_grid = {'kernel': ["rbf"], "C":[10**x for x in range(-8, 2) ], "gamma":[10**x for x in range(-2, 2) ]}
grid_search= GridSearchCV(SVC(), param_grid, cv = Kfold, n_jobs = 8, scoring = 'roc_auc', verbose=100)
grid_search.fit(X_data_clean, y_data_clean)


# In[ ]:

from sklearn.metrics import make_scorer
make_scorer()


# In[28]:

grid_search.best_score_, grid_search.best_params_


# In[15]:

from mlxtend.evaluate import lift_score
from sklearn.metrics import make_scorer
lift_scorer = make_scorer(lift_score)


# In[17]:

#X_train2, X_test2, y_train2, y_test2 = train_test_split(X_trun, y_trun, test_size=0.1, random_state=42)
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

X_data_clean=data_clean.drop("yy", axis=1)
y_data_clean=data_clean["yy"]

Kfold = KFold(n_splits = 5, shuffle=True)

param_grid = {'kernel': ["rbf"], "C":[10**x for x in range(-8, 2) ], "gamma":[10**x for x in range(-2, 2) ]}
grid_search_alift= GridSearchCV(SVC(), param_grid, cv = Kfold, n_jobs = 8, scoring = lift_scorer, verbose=100)
grid_search_alift.fit(X_data_clean, y_data_clean)


# In[19]:

grid_search_alift.best_score_, grid_search_alift.best_params_

