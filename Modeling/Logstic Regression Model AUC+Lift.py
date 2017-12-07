
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.preprocessing import label_binarize,StandardScaler
from mlxtend.evaluate import lift_score
import math


# In[2]:

nowpath=os.path.abspath('.')
bankdt=pd.read_csv(nowpath+'/bankout.csv',sep=',',header=None,index_col=None)
bank_df=pd.DataFrame(data=bankdt.loc[1:,1:])
bank_df.index=bankdt.loc[1:,0]
bank_df.columns=bankdt.loc[0,1:]


# In[3]:

bank_df.head()


# In[3]:

bank_df=bank_df.drop('duration',1)
#bank_df=bank_df.drop('duration',1).drop('emp.var.rate',1).drop('cons.price.idx',1).drop('cons.conf.idx',1).drop('euribor3m',1).drop('nr.employed',1).drop('pdays_999',1)


# In[4]:

#divide data into train and test randomly
train_np=np.array(bank_df)
cter=len(bank_df)
cttd=int(0.2*cter)
for k in range(cttd):
    r=np.random.randint(len(train_np))
    if 'test_np' not in dir():
        test_np=([train_np[r]])
    test_np=np.vstack((test_np,train_np[r]))
    train_np=np.delete(train_np,r,0)


# In[5]:

train_df=pd.DataFrame(data=train_np,columns=bank_df.columns)
test_df=pd.DataFrame(data=test_np,columns=bank_df.columns)


# In[6]:

X_train=train_df.drop('yy',1)
Y_train=train_df.yy.astype('int64')
LR=LogisticRegression(C=1e30,n_jobs=-1)


# In[7]:

import time
start = time.time()
LR.fit(X_train,Y_train)
end = time.time()
print(end - start)


# In[8]:

feature_lr=LR.coef_
feature_lr_pd=pd.DataFrame(index=X_train.columns.values,data=feature_lr.transpose())


# In[9]:

X_test=test_df.drop('yy',1)
Y_test=test_df.yy.astype('int64')
AcXt_Yt=LR.score(X_test,Y_test)
AcXt_Yt


# In[9]:

get_ipython().magic('matplotlib inline')
fig,f1=plt.subplots(figsize=(20,5))
xtks=[x for x in range(0,len(feature_lr[0]),1)]
y=np.array(feature_lr_pd)
f1.set_xticks(xtks)
f1.set_xticklabels(feature_lr_pd.index,rotation=60)
f1.bar(xtks,y)
plt.xlabel('feature')
plt.ylabel('coef')
plt.title('Coef of each feature with y')
plt.show()


# In[11]:

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize,StandardScaler


def xValLR(dataset, label_name, k, cs):
    X=dataset.drop(label_name,1)
    Y=dataset[label_name]
    #print(Y)
    k_fold = KFold(dataset.shape[0], n_folds = k)
    aucs={}
    for j, (train, test) in enumerate(k_fold):
        #print(X.loc[100000+train,:])
        #if j>0:
        #    continue #data is with bias, for the recent data is easier to be predicted
        for c in cs:
            LRcs=LogisticRegression(C=c)
            #sc = StandardScaler()
            tx=X.loc[train,:]
            ty=Y.loc[train].astype('int64') #Y is 1-dimensional!!
            #sc.fit(tx)
            #tx = sc.transform(tx)
            rx=X.loc[test,:]
            #rx = sc.transform(rx)
            ry=Y.loc[test].astype('int64')
            #print('tx'.format(j)+':'+str(len(tx)))
            LRcs.fit(tx,ty)
            fpr, tpr, thresholds = roc_curve(ry, LRcs.predict_proba(rx)[:,1])
            roc_auc = auc(fpr, tpr)
            new_auc='AUC_{}_{}'.format(c,j)#name of auc
            aucs[new_auc]=roc_auc
    #lr_grid_search = GridSearchCV(LogisticRegression(), param_grid_lr, cv = kfolds, scoring = 'roc_auc') 
    #lr_grid_search.fit(X, Y)
    return aucs


# In[14]:

cs=np.array([10**(i) for i in range(-10,31,5)])


# In[13]:

a=xValLR(bank_df,'yy',10,cs)


# In[25]:

def xValLR_shuffle(dataset, label_name, k, cs, rs):
    X=dataset.drop(label_name,1)
    Y=dataset[label_name]
    #print(Y)
    k_fold = KFold(dataset.shape[0], shuffle=True, n_folds = k, random_state=rs)
    aucs={}
    for j, (train, test) in enumerate(k_fold):
        #print(X.loc[100000+train,:])
        if j==9:
        #    continue #data is with bias, for the recent data is easier to be predicted
            for c in cs:
                LRcs=LogisticRegression(C=c)
                #sc = StandardScaler()
                tx=X.loc[train,:]
                ty=Y.loc[train].astype('int64') #Y is 1-dimensional!!
                #sc.fit(tx)
                #tx = sc.transform(tx)
                rx=X.loc[test,:]
                #rx = sc.transform(rx)
                ry=Y.loc[test].astype('int64')
                #print('tx'.format(j)+':'+str(len(tx)))
                LRcs.fit(tx,ty)
                fpr, tpr, thresholds = roc_curve(ry, LRcs.predict_proba(rx)[:,1])
                roc_auc = auc(fpr, tpr)
                new_auc='AUC_{}_{}'.format(c,j)#name of auc
                aucs[new_auc]=roc_auc
    
    #lr_grid_search = GridSearchCV(LogisticRegression(), param_grid_lr, cv = kfolds, scoring = 'roc_auc') 
    #lr_grid_search.fit(X, Y)
    return aucs


# In[26]:

b=xValLR_shuffle(bank_df,'yy',10,cs,5)


# In[27]:

b


# In[47]:

listaa=b.items()
pa,pb=zip(*listaa)
pb_use=np.array([pb[i].astype('float64') for i in range(0,89,10)])
mean_pb=np.mean(pb_use)
stderr_pb=math.sqrt(np.var(pb_use)/9)



fig,ax=plt.subplots(figsize=(12,6))
xplt=[i for i in range(9)]
xplt_out=np.zeros(9)

for i in range(9):
    xplt_out[i]=math.log10(cs[i])-5
ax.plot(xplt, pb_use, label='Mean(AUC)')
ax.plot(xplt,pb_use-2*stderr_pb,'k+',label='Mean(AUC)-2*stderr')
ax.plot(xplt,pb_use+2*stderr_pb,'k--',label='Mean(AUC)+2*stderr')

ax.set_xticks(xplt)
ax.set_xticklabels(xplt_out)
ax.set_ylabel('AUC')
ax.set_xlabel('Log10(c) ')
ax.set_title('Mean AUC in terms of c')
plt.legend()
plt.show()


# In[16]:

def plotUnivariateROC(dataset, label_name, k, cs, rs):
    '''
    preds is an nx1 array of predictions
    truth is an nx1 array of truth labels
    label_string is text to go into the plotting label
    '''
    X=dataset.drop(label_name,1)
    Y=dataset[label_name]
    #print(Y)
    k_fold = KFold(dataset.shape[0], shuffle=True, n_folds = k, random_state=rs)
    aucs={}
    for j, (train, test) in enumerate(k_fold):
        #print(X.loc[100000+train,:])
        if j==9:
        #    continue #data is with bias, for the recent data is easier to be predicted
            for c in cs:
                LRcs=LogisticRegression(C=c)
                #sc = StandardScaler()
                tx=X.loc[train,:]
                ty=Y.loc[train].astype('int64') #Y is 1-dimensional!!
                #sc.fit(tx)
                #tx = sc.transform(tx)
                rx=X.loc[test,:]
                #rx = sc.transform(rx)
                ry=Y.loc[test].astype('int64')
                #print('tx'.format(j)+':'+str(len(tx)))
                LRcs.fit(tx,ty)
                
                fpr, tpr, thresholds = roc_curve(ry, LRcs.predict_proba(rx)[:,1])
                roc_auc = auc(fpr, tpr)
                new_auc='AUC_{}_{}'.format(c,j)#name of auc
                aucs[new_auc]=roc_auc
                cl = (np.random.rand(), np.random.rand(), np.random.rand())
                #print Lift score
                LS=lift_score(ry, LRcs.predict(rx))
                print('Lift Score for c={} is :{}'.format(c,LS))
                #create a plot and set some options
                plt.plot(fpr, tpr, color = cl, label = 'AUC_{}'.format(c) + ' (AUC = %0.3f)' % roc_auc)


                plt.plot([0, 1], [0, 1], color='navy', lw=2,linestyle= '--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.0])
                plt.xlabel('FPR')
                plt.ylabel('TPR')
                plt.title('ROC',fontsize=25)
                plt.legend(loc="lower right")
    
    #lr_grid_search = GridSearchCV(LogisticRegression(), param_grid_lr, cv = kfolds, scoring = 'roc_auc') 
    #lr_grid_search.fit(X, Y)
    return aucs


# In[17]:

cs_score=np.array([10**(i) for i in range(-3,10,1)])
fig = plt.figure(figsize = (36, 24))
pc = plt.subplot(111)
for item in ([pc.title, pc.xaxis.label, pc.yaxis.label] +
             pc.get_xticklabels() + pc.get_yticklabels()):
    item.set_fontsize(25)

c=plotUnivariateROC(bank_df,'yy',10,cs_score,5)
box = pc.get_position()
pc.set_position([box.x0, box.y0 + box.height * 0.0 , box.width, box.height * 1])
pc.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.15), fancybox = True,  #set the legend position and font size
              shadow = True, ncol = 4, prop = {'size':20})


# In[18]:

c


# In[35]:

listaa=c.items()
pa,pc=zip(*listaa)
pc_use=np.array([pc[i].astype('float64') for i in range(13)])
mean_pc=np.mean(pc_use)
stderr_pc=math.sqrt(np.var(pc_use)/13)



fig,ax=plt.subplots(figsize=(12,6))
xplt=[i for i in range(13)]
xplt_out=np.zeros(13)

for i in range(13):
    xplt_out[i]=math.log10(cs_score[i])
ax.plot(xplt, pc_use, label='Mean(AUC)')
ax.plot(xplt,pc_use-2*stderr_pc,'k+',label='Mean(AUC)-2*stderr')
ax.plot(xplt,pc_use+2*stderr_pc,'k--',label='Mean(AUC)+2*stderr')

ax.set_xticks(xplt)
ax.set_xticklabels(xplt_out)
ax.set_ylabel('AUC')
ax.set_xlabel('Log10(c) ')
ax.set_title('Mean AUC in terms of c')
plt.legend()
plt.show()


# In[12]:

Lift_Score=np.array([5.890336935791481,6.00583373845406,5.748827339736432,5.7110268312011785,
                     5.7110268312011785,6.059843034663179,6.059843034663179,6.059843034663179,
                     5.821444106133102,5.821444106133102,5.821444106133102,5.821444106133102,5.821444106133102])


# In[15]:

get_ipython().magic('matplotlib inline')
cs_score=np.array([10**(i) for i in range(-3,10,1)])
fig,pc=plt.subplots(figsize=(12,6))
xplt=[i for i in range(13)]
xplt_label=np.zeros(13)
for i in range(13):
    xplt_label[i]=math.log10(cs_score[i])
pc.plot(xplt,Lift_Score,label='Mean(Lift Score)')
pc.set_xticks(xplt)
pc.set_xticklabels(xplt_label)
pc.set_xlabel('Log10(c)')
pc.set_ylabel('Lift Score')
pc.set_title('Lift Score in terms of c')
plt.legend()
plt.show()


# In[ ]:



