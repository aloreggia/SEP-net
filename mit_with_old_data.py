import pandas as pd

# data visualization
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style
from sklearn.preprocessing import LabelEncoder
import numpy as np

import statsmodels.api as sm
from statsmodels.formula.api import ols

from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
import scipy.stats as stats
from scipy.stats import wilcoxon


import random
random.seed(42)

import networkx as nx

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
suffix={"DF":["SP","WA","SO","CA","CO","SU","BR","HS","TP","MA","FA","SW"],
		"BF":["HA","CL","TU","JA","FR","EL","BR"],	
		"AF":["MN","BA","JA","CA","BR","HR"]}

listIndex=("1_1","2_1","3_1","4_1","5_1","6_1","7_1","8_1","8_2")

mapping={}
list=('DFO_SP','DFO_WA','DFO_SO','DFO_CA','DFO_CO','DFO_SU','DFO_BR','DFO_HS','DFO_TP','DFO_MA','DFO_FA','DFO_SW','BFO_HA','BFO_CL','BFO_TU','BFO_JA','BFO_FR','BFO_EL','BFO_BR','AFO_MN','AFO_BA','AFO_JA','AFO_CA','AFO_BR','AFO_HR')

listNine=[]

for p in prefix:
	for s in suffix[p]:
		string=p+"O_"+s
		mapping[string]=list.index(string)
		item=[]
		for i in listIndex:
			item.append(p+"U_"+s+"_"+i)
			
		listNine.append(item)
		print(string, end="")

new_listNine=['q_1_1', 'q_2_1', 'q_3_1', 'q_4_1', 'q_5_1', 'q_6_1', 'q_7_1', 'q_8_1', 'q_8_2']

train_df=pd.read_csv("data/all.csv", sep=";")

temp_train=pd.melt(train_df,value_vars=list, var_name="type")
temp_train["type"]=temp_train["type"].map(mapping)
temp_train=temp_train.dropna()

new_train=pd.DataFrame()

'''for i in new_listNine:
	temp_train[i]=np.nan'''

for l in listNine:
	#crea dizionario per la mappatura con i nuovi campi tutti con lo stesso nome
	dictTemp={l[i]: new_listNine[i] for i in range(0, len(new_listNine))}
	#estrae domande per scenario
	temp=train_df[train_df.columns.intersection(l)]
	#cambia nome
	temp=temp.rename(columns=dictTemp)
	temp=temp.dropna()
	temp["type"]=listNine.index(l)
	#print(temp)
	#temp["value"]=np.nan
	new_train=pd.concat([new_train,temp], axis=0, ignore_index=True)
	
'''results = ols('q_1_1 ~ C(type)', data=new_train).fit()
results.summary()

results = ols('q_2_1 ~ C(type)', data=new_train).fit()
results.summary()

results = ols('q_6_1 ~ C(type)', data=new_train).fit()
results.summary()

results = ols('q_8_1 ~ C(type)', data=new_train).fit()
results.summary()

mc = MultiComparison(new_train['q_1_1'], new_train['type'])
mc_results = mc.tukeyhsd()
print(mc_results)

mc = MultiComparison(new_train['q_2_1'], new_train['type'])
mc_results = mc.tukeyhsd()
print(mc_results)

mc = MultiComparison(new_train['q_6_1'], new_train['type'])
mc_results = mc.tukeyhsd()
print(mc_results)

mc = MultiComparison(new_train['q_8_1'], new_train['type'])
mc_results = mc.tukeyhsd()
print(mc_results)'''

threshold=0.01

for variable in new_listNine:
	data_tot=[]
	for t1 in range(0,25):
		for t2 in range(t1+1,25):
			data1=new_train[variable][new_train['type']==t1]
			data2=new_train[variable][new_train['type']==t2]
			data_tot.append(data1)
			data_tot.append(data2)
			p1=compareEq(data1,data2)[1]
			
			#reject = p1<0.01
			#if(reject):
			#	print('Var: %s \tT1: %d \tT2: %d \t p=%.3E \tcreject:%r' % (variable,t1,t2,p1,reject))
	print("")
	f_value, p_value = stats.kruskal(*data_tot)
	print("ONE_WAY _> Var %s \t mean p3 %.3E \t reject: %r" % (variable,p_value, p_value<threshold))
	print("")
	
stats.ttest_ind(data1, data2)

for variable in new_listNine:
	data1=new_train[variable][(new_train['type']==5) | (new_train['type']==6) | (new_train['type']==10) | (new_train['type']==11)] #scenario deli
	
	data2=new_train[variable][(new_train['type']==13) | (new_train['type']==14) | (new_train['type']==16) | (new_train['type']==17)] #scenario aereoporto
	data3=new_train[variable][(new_train['type']==19) | (new_train['type']==20) | (new_train['type']==21) | (new_train['type']==24)] #scenario bagno
	
	p1=compareEq(data1,data2)[1]
	p2=compareEq(data1,data3)[1]
	p3=compareEq(data3,data2)[1]
	
	threshold=0.01
	
	#print("Var %s \t mean p1 %.3E \t reject: %r" % (variable,p1, p1<threshold))
	#print("Var %s \t mean p2 %.3E \t reject: %r" % (variable,p2, p2<threshold))
	#print("Var %s \t mean p3 %.3E \t reject: %r" % (variable,p3, p3<threshold))
	print("")
	f_value, p_value = stats.f_oneway(data1, data2, data3)
	print("ONE_WAY _> Var %s \t mean p3 %.3E \t reject: %r" % (variable,p_value, p_value<threshold))


data1=temp_train['value'][(temp_train['type']==5) | (temp_train['type']==6) | (temp_train['type']==10) | (temp_train['type']==11)] #scenario deli

print("Lenght data1: ",len(data1))
print("Lenght data2: ",len(data2))
print("Lenght data3: ",len(data3))

data2=temp_train['value'][(temp_train['type']==13) | (temp_train['type']==14) | (temp_train['type']==16) | (temp_train['type']==17)] #scenario aereoporto
data3=temp_train['value'][(temp_train['type']==19) | (temp_train['type']==20) | (temp_train['type']==21) | (temp_train['type']==24)] #scenario bagno

p1=compareEq(data1,data2)[1]
p2=compareEq(data1,data3)[1]
p3=compareEq(data3,data2)[1]

threshold=0.05

print("Var %s \t mean p1 %.3E \t reject: %r" % ('value',p1, p1<threshold))
print("Var %s \t mean p2 %.3E \t reject: %r" % ('value',p2, p2<threshold))
print("Var %s \t mean p3 %.3E \t reject: %r" % ('value',p3, p3<threshold))
print("")

print(data3.describe())

f_value, p_value = stats.f_oneway(data1, data2, data3)
print("OneWay %s \t mean p %.3E \t reject: %r" % ('value',p_value, p_value<threshold))
print("")

f_value, p_value = stats.kruskal(data1, data2, data3)
print("Kruskal %s \t mean p %.3E \t reject: %r" % ('value',p_value, p_value<threshold))
print("")

'''mc = MultiComparison(temp_train['value'], new_train['type'])
mc_results = mc.tukeyhsd()
print(mc_results)'''

mc = MultiComparison(new_train['q_1_1'], new_train['type'])
mc_results = mc.tukeyhsd()
print(mc_results)