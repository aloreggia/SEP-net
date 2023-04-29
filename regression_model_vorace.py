#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd



# data visualization
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style
from sklearn.preprocessing import LabelEncoder
import numpy as np

import scipy.stats as stats
from scipy.stats import wilcoxon, shapiro


import random
random.seed(42)

import networkx as nx

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

def compareEq(data1,data2):
    n=len(data1)
    if len(data1)>len(data2):
        n=len(data2)
    temp_data1=data1.sample(n=n)
    temp_data2=data2.sample(n=n)
    #print(len(data1), " ", len(data2))
    stat, p = wilcoxon(temp_data1, temp_data2, correction=True)
    return stat, p

# CARICA IL DATASET
prefix=("DF","BF","AF")
suffix={"DF":["SP","WA","SO","CA","CO","SU","BR","HS","TP","MA","FA","SW"],         "BF":["HA","CL","TU","JA","FR","EL","BR"],	         "AF":["MN","BA","JA","CA","BR","HR"]}

listIndex=("1_1","2_1","3_1","4_1","5_1","6_1","7_1","8_1","8_2")

mapping={}
list=('DFO_SP','DFO_WA','DFO_SO','DFO_CA','DFO_CO','DFO_SU','DFO_BR','DFO_HS','DFO_TP','DFO_MA','DFO_FA','DFO_SW','BFO_HA','BFO_CL','BFO_TU','BFO_JA','BFO_FR','BFO_EL','BFO_BR','AFO_MN','AFO_BA','AFO_JA','AFO_CA','AFO_BR','AFO_HR')
#len(list)

listNine=[]

for p in prefix:
    for s in suffix[p]:
        string=p+"O_"+s
        mapping[string]=list.index(string)
        item=[]
        item.append(string)
        for i in listIndex:
            item.append(p+"U_"+s+"_"+i)

        listNine.append(item)
        #print(string, end="")

#print(listNine)
new_listNine=['pref','q_1_1', 'q_2_1', 'q_3_1', 'q_4_1', 'q_5_1', 'q_6_1', 'q_7_1', 'q_8_1', 'q_8_2']
new_listNine=['pref','everyone','first_person','middle_person','last_person','cutter','univers.','likelihood','delay_min','delay_sec']
new_listNine=['Judgement',"Global Welfare", "First Person Welfare", "Middle Person Welfare", "Last Person Welfare", "Line Cutter Welfare", "Universalization", "Likelihood",'delay_min','delay_sec']

train_df=pd.read_csv("data/pref_eval_test.csv", sep=";")
train_df=train_df[train_df.Q276=="I am paying attention"]

new_train=pd.DataFrame()

for l in listNine:
    #crea dizionario per la mappatura con i nuovi campi tutti con lo stesso nome
    #l=[list[j]]+listNine[j]
    dictTemp={l[i]: new_listNine[i] for i in range(0, len(new_listNine))}
    l.append('Duration')
    #estrae domande per scenario
    temp=train_df[train_df.columns.intersection(l)]
    #print(temp)
    #cambia nome
    temp=temp.rename(columns=dictTemp)
    temp=temp.dropna()
    temp["type"]=listNine.index(l)
    print(len(listNine))
    #exit()
    if listNine.index(l)<12:
        temp["family"]=0
        temp.Duration /= 12
    elif listNine.index(l)<19:
        temp["family"]=1
        temp.Duration /= 7
    else:
        temp["family"]=2
        temp.Duration /= 6
    #print(temp)
    #temp["value"]=np.nan
    new_train=pd.concat([new_train,temp], axis=0, ignore_index=True)

print(f"DELI TIME: {np.mean(new_train.Duration[new_train.family==0])}")
print(f"BATH TIME: {np.mean(new_train.Duration[new_train.family==1])}")
print(f"AIR TIME: {np.mean(new_train.Duration[new_train.family==2])}")


# In[2]:


new_train


# In[3]:


from sklearn import linear_model

y = new_train['Judgement']
x = new_train[["Global Welfare", "First Person Welfare", "Middle Person Welfare", "Last Person Welfare", "Line Cutter Welfare", "Universalization", "Likelihood",'delay_min','delay_sec']]


# In[4]:


# with sklearn
regr = linear_model.LinearRegression()
regr.fit(x, y)


# In[5]:


y = y-1


# In[6]:


y


# In[7]:


from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3) # 70% training and 30% test


# In[8]:


#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=1000)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)


# In[9]:


y_pred


# In[10]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[11]:


# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[12]:



# evaluate predictions
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[13]:


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

plt.hist(y, weights=np.ones(len(y)) / len(y))
plt.show()


# In[14]:


from Vorace import Vorace


# In[15]:


from keras.utils import to_categorical
y_oneHot=to_categorical(y,num_classes=2)


# In[16]:


y_oneHot=to_categorical(y_train,num_classes=2)
y_train
 
y_oneHot_test=to_categorical(y_test,num_classes=2)


# In[23]:


vorace = Vorace(n_models=10, profile_type=3, nInput=9, nClasses=2, batch_size=32)


# In[ ]:


vorace.fit(X_train, y_train, y_oneHot, bagging=False)


# In[ ]:


y_pred_vorace,_ = vorace.predict(voting="Plurality",x=X_test, nClasses=2, argMax=True, tiebreak="best")
#print(y_pred_vorace)
#print(y_test)

accuracy = accuracy_score(y_test, y_pred_vorace)
print(accuracy)


# In[ ]:





# In[ ]:




