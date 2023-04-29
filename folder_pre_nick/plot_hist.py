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
		item.append(string)
		for i in listIndex:
			item.append(p+"U_"+s+"_"+i)
			
		listNine.append(item)
		print(string, end="")

new_listNine=['pref','q_1_1', 'q_2_1', 'q_3_1', 'q_4_1', 'q_5_1', 'q_6_1', 'q_7_1', 'q_8_1', 'q_8_2']

train_df=pd.read_csv("data/pre_eval.csv", sep=";")

temp_train=pd.melt(train_df,value_vars=list, var_name="type")
temp_train["type"]=temp_train["type"].map(mapping)
temp_train=temp_train.dropna()

new_train=pd.DataFrame()

'''for i in new_listNine:
	temp_train[i]=np.nan'''
	
for l in listNine:
	#crea dizionario per la mappatura con i nuovi campi tutti con lo stesso nome
	#l=[list[j]]+listNine[j]
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

for v in new_listNine:
	if v not in ("pref","type"):
		data_set=new_train[v]
		#data_set.agg(['min', 'max', 'mean', 'std']).round(decimals=2)
		fig, ax = plt.subplots()
		#data_set.plot.kde(ax=ax, legend=False, title='Histogram: '+v)
		#data_set.plot.hist(density=True, ax=ax)
		ax.boxplot(data_set)
		ax.set_ylabel('Density')
		ax.grid(axis='y')
		ax.set_facecolor('#d8dcd6')
		plt.show()
		#plt.savefig(""+v+".png")