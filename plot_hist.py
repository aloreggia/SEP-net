import pandas as pd

# data visualization
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style
from sklearn.preprocessing import LabelEncoder
import numpy as np

#odels.api as sm
#from statsmodels.formula.api import ols

#from statsmodels.stats.multicomp import pairwise_tukeyhsd
#from statsmodels.stats.multicomp import MultiComparison
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
#new_listNine=['pref','everyone','first_person','middle_person','last_person','cutter','univers.','likelihood','delay_min','delay_sec']
new_listNine=['pref',"Global Welfare", "First Person Welfare", "Middle Person Welfare", "Last Person Welfare", "Line Cutter Welfare", "Universalization", "Likelihood",'delay_min','delay_sec']


train_df=pd.read_csv("data/pref_eval_test.csv", sep=";")
train_df=train_df[train_df.Q276=="I am paying attention"]
print(train_df)

temp_train=pd.melt(train_df,value_vars=list, var_name="type")
temp_train["type"]=temp_train["type"].map(mapping)
temp_train=temp_train.dropna()

new_train=pd.DataFrame()

'''for i in new_listNine:
	temp_train[i]=np.nan'''

print(listNine)
for l in listNine:
	#crea dizionario per la mappatura con i nuovi campi tutti con lo stesso nome
	#l=[list[j]]+listNine[j]
	dictTemp={l[i]: new_listNine[i] for i in range(0, len(new_listNine))}
	#estrae domande per scenario
	temp=train_df[train_df.columns.intersection(l)]
	#print(temp)
	#cambia nome
	temp=temp.rename(columns=dictTemp)
	print(temp)
	temp=temp.drop(['pref'], axis=1)
	temp=temp.dropna()
	print(temp)
	temp["type"]=listNine.index(l)
	#print(temp)
	#temp["value"]=np.nan
	new_train=pd.concat([new_train,temp], axis=0, ignore_index=True)

dropped_dataset = new_train.drop(['delay_min','delay_sec','type'], axis=1)
#title = 'BOX PLOTS of EVALUATION VARIABLES \n'
#plt.title(title, loc='left', fontsize=16)
#fig, ax = plt.subplots()
#xtickNames = plt.setp(ax, xticklabels=["Global Welfare", "First Person Welfare", "Middle Person Welfare", \
#	"Last Person Welfare", "Line Cutter Welfare", "Universalization", "Likelihood"])

fig, ax = plt.subplots()
#sns.boxplot(data=dropped_dataset)
sns.violinplot(data=dropped_dataset)
plt.tight_layout()
#plt.setp(xtickNames, rotation=30, fontsize=12)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(16)

plt.xticks(rotation=10)
plt.show()
exit()
for v in new_listNine:
	if v not in ("pref","type"):
		data_set=new_train[v]
		'''#data_set.agg(['min', 'max', 'mean', 'std']).round(decimals=2)
		fig, ax = plt.subplots()
		#data_set.plot.kde(ax=ax, legend=False, title='Histogram: '+v)
		#data_set.plot.hist(density=True, ax=ax)
		ax.boxplot(data_set)
		ax.set_ylabel('Density')
		ax.grid(axis='y')
		ax.set_facecolor('#d8dcd6')'''
		sns.boxplot(data=data_set)
		plt.show()
		#plt.savefig(""+v+".png")
