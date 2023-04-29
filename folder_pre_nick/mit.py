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


import networkx as nx

# CARICA IL DATASETA
'''train_df1=pd.read_csv("deli_underlying_few.csv", sep=",")
train_df1=train_df1.iloc[:,:13]
train_df1.sort_values(by=['subjectcode'])
train_df2=pd.read_csv("deli_delay_few.csv", sep=",")
train_df2=train_df2.iloc[:,:13]
train_df2.sort_values(by=['subjectcode'])

train_df = pd.concat([train_df1, train_df2], axis=1, sort=False)'''

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

#new_train=pd.melt(train_df,value_vars=list, var_name="type")
#new_train["type"]=new_train["type"].map(mapping)

new_train=pd.DataFrame()

'''for i in new_listNine:
	new_train[i]=np.nan'''

for l in listNine:
	#crea dizionario per la mappatura con i nuovi campi tutti con lo stesso nome
	dictTemp={l[i]: new_listNine[i] for i in range(0, len(new_listNine))}
	#estrae domande per scenario
	temp=train_df[train_df.columns.intersection(l)]
	#cambia nome
	temp=temp.rename(columns=dictTemp)
	temp=temp.dropna()
	temp["type"]=listNine.index(l)
	print(temp)
	#temp["value"]=np.nan
	new_train=pd.concat([new_train,temp], axis=0, ignore_index=True)
	
results = ols('q_1_1 ~ C(type)', data=new_train).fit()
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
print(mc_results)

for i in range(0,25):
	scenario=new_train[new_train.type==i]
	correlation = scenario.astype(float).corr()
	sns.heatmap(correlation, annot=True, cbar=True, cmap="RdYlGn")
	plt.show()

correlation = new_train.astype(float).corr(method="spearman")
sns.heatmap(correlation, annot=True, cbar=True, cmap="RdYlGn")
plt.show()
	
# CALCOLA LA PERCENTUALE DEI VALORI MANCANTI NEL DATASET
total = train_df.isnull().sum().sort_values(ascending=False)
percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(5)

# SELEZIONO SOLO LE COLONNE CON I DATI CHE MI INTERESSANO RIMUOVENDO I RECORD CON VALORI MANCANTI
#train_df= train_df[['Survived', 'Pclass','Sex', 'Fare', 'Age']][train_df.Fare!=0].dropna()

train_df.info()

train_df.describe()



'''
STAMPO IL GRAFICO DIVISO PER SESSO EVIDENZIANDO LA PERCENTUALE DI SOPRAVVISSUTI IN BASE ALL'ETà
'''
survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = train_df[train_df['Sex']=='female']
men = train_df[train_df['Sex']=='male']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')

plt.show()

'''
PLOTTO QUANTI DEI SOPRAVVISSUTI ERANO IN PRIMA, SECONDA O TERZA CLASSE
'''
sns.barplot(x='Pclass', y='Survived', data=train_df)
plt.show()

'''
PLOTTO LA SUDDIVISIONE DEI SOPRAVVISSUTI IN BASE A ETà E CLASSE
'''
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
plt.show()

'''
PLOTTO LA SUDDIVISIONE DEI SOPRAVVISSUTI IN BASE A SESSO E CLASSE
'''
sns.barplot(x='Pclass', y='Survived', hue="Sex", data=train_df)
plt.show()

sns.set(style='whitegrid', color_codes=True)
sns.pairplot(train_df, 
             kind='scatter', 
             hue='Survived', 
             size=1.6, 
             plot_kws={'s': 30, 'alpha': 0.2}, 
             palette={0:'#EB434A', 1:'#61A98F'});
plt.show()

'''
CAMBIO ETICHETTA MASCHIO/FEMMINA IN 0/1
'''
genders = {"male": 0, "female": 1}
train_df['Sex'] = train_df['Sex'].map(genders)


'''
CAMBIO ETICHETTA FARE IN CHEAP/EXPENSIVE 0 se <30, 1>30
'''
train_df.loc[:,'Fare'] = pd.cut(train_df.Fare, [train_df.Fare.min(),30, train_df.Fare.max()], labels =[0,1])


#Sex: Prior probability
print("------------")
print ("P(Sex)")
print ("Sex = 0 (female), 1 (male)")
sex_0=train_df[train_df.Sex==0].shape[0]/float(train_df.Sex.shape[0])
sex_1=train_df[train_df.Sex==1].shape[0]/float(train_df.Sex.shape[0])
print (sex_0,sex_1)

#Fare: Prior probability
print("------------")
print ("P(Fare)")
print ("Fare = 0 (cheap), 1 (expensive)")
fare_0=train_df[train_df.Fare==0].shape[0]/float(train_df.Fare.shape[0])
fare_1=train_df[train_df.Fare==1].shape[0]/float(train_df.Fare.shape[0])
print (fare_0, fare_1)


print ("P(Class|Fare)")
print ("Class= 1, 2 ,3  ; Fare = 0 (cheap):")
class1_0=train_df[(train_df.Fare==0) & (train_df.Pclass==1)].shape[0] /(1.0*train_df[train_df.Fare==0].shape[0])
class2_0=train_df[(train_df.Fare==0) & (train_df.Pclass==2)].shape[0] /(1.0*train_df[train_df.Fare==0].shape[0])
class3_0=train_df[(train_df.Fare==0) & (train_df.Pclass==3)].shape[0]/(1.0*train_df[train_df.Fare==0].shape[0])
print(class1_0,class2_0,class3_0)

print ("Class= 1, 2 ,3  ; Fare = 1 (expensive):")
class1_1=train_df[(train_df.Fare==1) & (train_df.Pclass==1)].shape[0] /(1.0*train_df[train_df.Fare==1].shape[0])
class2_1=train_df[(train_df.Fare==1) & (train_df.Pclass==2)].shape[0] /(1.0*train_df[train_df.Fare==1].shape[0])
class3_1=train_df[(train_df.Fare==1) & (train_df.Pclass==3)].shape[0]/(1.0*train_df[train_df.Fare==1].shape[0])
print(class1_1,class2_1,class3_1)



def get_probs_surv_cond(df, Surv, Pclass, Sex):
    # Return survival probability conditioned on Class and Sex
    # P(Surv | Sex, Pclass)
    return (df[ (df.Survived==Surv) & (df.Pclass==Pclass) & (df.Sex==Sex)].shape[0]
            /(1.0*df[(df.Pclass==Pclass) & (df.Sex==Sex)].shape[0]))


# Surv Probability
print("------------")
print("P(Surv|Class,Sex)")
print("Surv = 0 ,1 , Class = 1 , Sex = 0")
print(get_probs_surv_cond(train_df, 0, 1, 0),
get_probs_surv_cond(train_df,  1, 1, 0))
print("Surv = 0 ,1 , Class = 2 , Sex = 0")
print(get_probs_surv_cond(train_df,  0, 2, 0),
get_probs_surv_cond(train_df,  1, 2, 0))
print("Surv = 0 ,1 , Class = 3 , Sex = 0")
print(get_probs_surv_cond(train_df,  0, 3, 0),
get_probs_surv_cond(train_df,  1, 3, 0))
print("Surv = 0 ,1 , Class = 1 , Sex = 1")
print(get_probs_surv_cond(train_df,  0, 1, 1),
get_probs_surv_cond(train_df,  1, 1, 1))
print("Surv = 0 ,1 , Class = 2 , Sex = 1")
print(get_probs_surv_cond(train_df,  0, 2, 1),
get_probs_surv_cond(train_df,  1, 2, 1))
print("Surv = 0 ,1 , Class = 3 , Sex = 1")
print(get_probs_surv_cond(train_df,  0, 3, 1),
get_probs_surv_cond(train_df,  1, 3, 1))

surv_prob=[]
for classn in range(1,4):
	temp_class=[]
	for sex in range(2):
		temp_sex=[]
		for surv in range(2):
			temp_sex.append(get_probs_surv_cond(train_df,  surv, classn, sex))
		temp_class.append(temp_sex)
	surv_prob.append(temp_class)

bn=nx.DiGraph()

sex_prior = np.array([sex_0,sex_1])
bn.add_node('SEX', dtype='Discrete', prob=sex_prior, pos=(2, 4))

fare_prior = np.array([fare_0,fare_1])
bn.add_node('FARE', dtype='Discrete', prob=fare_prior, pos=(1, 4))

class_array = np.array([[class1_0,class1_1],[class2_0,class2_1],[class3_0,class3_1]])
bn.add_node('CLASS', dtype='Discrete', prob=class_array, pos=(1, 3))
bn.add_node('SURV', dtype='Discrete', prob=surv_prob, pos=(1.5,2))

bn.add_edges_from([('FARE', 'CLASS'),('CLASS', 'SURV'),('SEX', 'SURV')])



nx.draw_networkx(bn, pos=nx.get_node_attributes(bn,'pos'))
plt.show()

