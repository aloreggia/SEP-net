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
suffix={"DF":["SP","WA","SO","CA","CO","SU","BR","HS","TP","MA","FA","SW"],
		"BF":["HA","CL","TU","JA","FR","EL","BR"],	
		"AF":["MN","BA","JA","CA","BR","HR"]}

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
#print(train_df)

temp_train=pd.melt(train_df,value_vars=list, var_name="type")
temp_train["type"]=temp_train["type"].map(mapping)
temp_train=temp_train.dropna()

#print(temp_train)
#exit()
new_train=pd.DataFrame()

'''for i in new_listNine:
	temp_train[i]=np.nan'''
#print(listNine)
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
# get correlations

df_corr = new_train.corr()
#print(df_corr)
# irrelevant fields
fields = ['Duration','family','delay_min','delay_sec','type']
# drop rows
df_corr.drop(fields, inplace=True)
# drop cols
df_corr.drop(fields, axis=1, inplace=True)
# adjust mask and df
fig, ax = plt.subplots(figsize=(10, 7))
# mask
mask = np.triu(np.ones_like(df_corr, dtype=np.bool))
# adjust mask and df
mask = mask[1:, :-1]
corr = df_corr.iloc[1:,:-1].copy()
#corr = df_corr.copy()

cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)
# plot heatmap
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", 
           linewidths=5, cmap="Blues", vmin=-1, vmax=1, 
           cbar_kws={"shrink": .8}, square=True)
# ticks
yticks = [i for i in corr.index]
xticks = [i for i in corr.columns]
plt.yticks(plt.yticks()[0], labels=yticks, rotation=0, fontsize=11)
plt.xticks(plt.xticks()[0], labels=xticks, rotation=90, fontsize=11)

title = 'CORRELATION MATRIX\nEVALUATION and PREFERENCE VARIABLES \n'
#plt.title(title, loc='left', fontsize=16)

b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values

plt.tight_layout()
plt.show()
exit()

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

threshold=0.05

print("Null-hypothesis: given an evaluation variable the scenario does not affect the preference variable")
#for variable in new_listNine:
data_tot=[]
score_per_var={}
total=0
#print(new_listNine[1:-2])
for variable in new_listNine[1:-2]:
	score_per_var[variable]=0

for t1 in range(0,25):
	for t2 in range(t1+1,25):
		print("%s & %s & "%(list[t1],list[t2]), end='')
		total += 1
		for variable in new_listNine[1:-2]:
			data1=new_train[variable][new_train['type']==t1]
			data2=new_train[variable][new_train['type']==t2]
			data_tot.append(data1)
			data_tot.append(data2)
			p1=compareEq(data1,data2)[1]
			
			reject = p1<0.05
			if(reject): 
				print("\\textbf{", end='')
				score_per_var[variable] += 1
			#print('Var: %s \tT1: %d \tT2: %d \t p=%.3E \tcreject:%r' % (variable,t1,t2,p1,reject), end='')
			print('%.4f' % (p1), end='')
			if(reject): print("}", end='')
			print(" & ", end='')
		print("\\\\")
		
		
		#f_value, p_value = stats.kruskal(*data_tot)
		#if p_value>threshold:
		#	print("")
		#	print("ONE_WAY scenario %d against %d -> Var %s \t mean p3 %.3E \t reject: %r" % (t1, t2,variable,p_value, p_value<threshold))
		#print("ONE_WAY scenario %d against %d -> Var %s \t mean p3 %.3E \t reject: %r" % (t1, t2,variable,p_value, p_value<threshold))
		#	print("")
#exit()
print("& & ")
for variable in new_listNine[1:-2]:
	print(f"{score_per_var[variable]/total:.4f}&", end="")

print()
print(score_per_var)
#exit()
#stats.ttest_ind(data1, data2)

print("#############################################################################################################################")
print("null-hypotesis: the location does not affect the evaluation variable")
i=0
threshold=0.05
for variable in new_listNine:
	#data1=new_train[variable][(new_train['type']==5) | (new_train['type']==6) | (new_train['type']==10) | (new_train['type']==11)] #scenario deli
	data1=new_train[variable][(new_train['type']<12)] #scenario deli
	#print(data1)
	
	#data2=new_train[variable][(new_train['type']==13) | (new_train['type']==14) | (new_train['type']==16) | (new_train['type']==17)] #scenario bath
	data2=new_train[variable][(new_train['type']>11) & (new_train['type']<19)] #scenario bath
	#data3=new_train[variable][(new_train['type']==19) | (new_train['type']==20) | (new_train['type']==21) | (new_train['type']==24)] #scenario air
	data3=new_train[variable][(new_train['type']>18)] #scenario air
	
	print(f"{new_listNine[i]} &", end="")
	p=compareEq(data1,data2)[1]
	if p<threshold : print("\\textbf{", end="")
	print(f"{p:.4f}", end="")
	if p<threshold : print("}", end="")
	print(" &", end="")
	p=compareEq(data1,data3)[1]
	if p<threshold : print("\\textbf{", end="")
	print(f"{p:.4f}", end="")
	if p<threshold : print("}", end="")
	print(" &", end="")
	p=compareEq(data3,data2)[1]
	if p<threshold : print("\\textbf{", end="")
	print(f"{p:.4f}", end="")
	if p<threshold : print("}", end="")
	print(" \\\\")
	i=i+1
	
	
	#print("Var %s \t mean p1 %.3E \t reject: %r" % (variable,p1, p1<threshold))
	#print("Var %s \t mean p2 %.3E \t reject: %r" % (variable,p2, p2<threshold))
	#print("Var %s \t mean p3 %.3E \t reject: %r" % (variable,p3, p3<threshold))
	#print("")
	#f_value, p_value = stats.f_oneway(data1, data2, data3)
	#print("ONE_WAY _> Var %s \t mean p3 %.3E \t reject: %r" % (variable,p_value, p_value<threshold))
exit()
print("#############################################################################################################################")
	
'''
Check whether location variable influence the preference
null-hypotesis: the location does not affect the preference variable
'''
print("null-hypotesis: the location does not affect the preference variable")
data1=temp_train['value'][(temp_train['type']==5) | (temp_train['type']==6) | (temp_train['type']==10) | (temp_train['type']==11)] #scenario deli

print("Lenght data1: ",len(data1))
print("Lenght data2: ",len(data2))
print("Lenght data3: ",len(data3))

data2=temp_train['value'][(temp_train['type']==13) | (temp_train['type']==14) | (temp_train['type']==16) | (temp_train['type']==17)] #scenario bathroom
data3=temp_train['value'][(temp_train['type']==19) | (temp_train['type']==20) | (temp_train['type']==21) | (temp_train['type']==24)] #scenario airport

p1=compareEq(data1,data2)[1]
p2=compareEq(data1,data3)[1]
p3=compareEq(data3,data2)[1]

threshold=0.05

print("%s \t mean p1 %.3E \t reject: %r" % ('Deli-Bath',p1, p1<threshold))
print("%s \t mean p2 %.3E \t reject: %r" % ('Deli-Air',p2, p2<threshold))
print("%s \t mean p3 %.3E \t reject: %r" % ('Air-Bath',p3, p3<threshold))
print("")

'''print(data3.describe())

f_value, p_value = stats.f_oneway(data1, data2, data3)
print("OneWay %s \t mean p %.3E \t reject: %r" % ('value',p_value, p_value<threshold))
print("")

f_value, p_value = stats.kruskal(data1, data2, data3)
print("Kruskal %s \t mean p %.3E \t reject: %r" % ('value',p_value, p_value<threshold))
print("")'''

'''mc = MultiComparison(temp_train['value'], new_train['type'])
mc_results = mc.tukeyhsd()
print(mc_results)'''

'''mc = MultiComparison(new_train['q_1_1'], new_train['type'])
mc_results = mc.tukeyhsd()
print(mc_results)'''

'''
Check whether showing first preference and then evaluation influences the decision
null-hypotesis: changing the order of questions does not affect the preference
'''
print("#############################################################################################################################")
print("null-hypotesis: changing the order of the question does not affect the preference")
for i in range(0,len(list)):
	#estrae subset nei quali sono state visualizzati agli individui prima le variabili di preferenza e poi quelle di valutazione per tutti gli scenari
	index_pref=1
	index_eval=4
	if i>11:
		if i<19:
			index_pref=2
			index_eval=5
		else:
			index_pref=3
			index_eval=6
			
	train_pref=train_df[list[i]][train_df['CONDITION']==index_pref]
	train_pref = train_pref - 1
	train_eval=train_df[list[i]][train_df['CONDITION']==index_eval]
	train_eval = train_eval - 1
	p=compareEq(train_pref,train_eval)[1]
	#print("%2d.ORDER CHECKING Scenario %s \t mean p1 %.3E \t reject: %r" % (i,list[i],p, p<threshold))
	print(" %s & %.4f (%.4f) & %.4f (%.4f) & %.4f & %s\\\\" % (list[i],np.mean(train_pref), np.std(train_pref), np.mean(train_eval), np.std(train_eval), p, p<threshold))


print("#############################################################################################################################")
#exit()

	
'''train_pref1=train_df[list[i]][train_df['CONDITION']==index_pref]
train_pref2=train_df[list[i+1]][train_df['CONDITION']==index_pref]
p=compareEq(train_pref1,train_pref1)[1]
print("%2d. Scenario %s \t mean p1 %.3E \t reject: %r" % (i,list[i],p, p<threshold))
'''

'''
forse conviene dividere le variabili, i.e. q_1_1, in 3 o 4 gruppi distinti
verificare quindi se la preferenza è influenzata o no dai gruppi, quindi da qui quale
variabile di valutazione influenza la preferenza
group id:
0: value < -25
1: -25 < value < 0
2: 0< value < 25
3: 25 > value

null-hypotesis: given a scenario, the evaluation variable does not affect the preference variable
'''
#creo i 4 gruppi diversi
# Group responses into 4 different groups based on the distribution of the responses. Look at the q_x_y.png files in the folder
l='q_1_1'
new_train.loc[(new_train[l] > 0) &(new_train[l] < 18),l]=2
new_train.loc[(new_train[l] > -20) &(new_train[l] <= 0),l]=1
new_train.loc[(new_train[l] <= -20),l]=0
new_train.loc[(new_train[l] >= 18),l]=3

l='q_2_1'
new_train.loc[(new_train[l] > -3) &(new_train[l] < 20),l]=2
new_train.loc[(new_train[l] <= -12),l]=0
new_train.loc[(new_train[l] > -12) &(new_train[l] <= -3),l]=1
new_train.loc[(new_train[l] >= 20),l]=3

l='q_3_1'
new_train.loc[(new_train[l] > -5) &(new_train[l] < 17),l]=2
new_train.loc[(new_train[l] <= -18),l]=0
new_train.loc[(new_train[l] > -18) &(new_train[l] <= -5),l]=1
new_train.loc[(new_train[l] >= 17),l]=3

l='q_4_1'
new_train.loc[(new_train[l] > 0) &(new_train[l] < 18),l]=2
new_train.loc[(new_train[l] > -20) &(new_train[l] <= 0),l]=1
new_train.loc[(new_train[l] <= -20),l]=0
new_train.loc[(new_train[l] >= 18),l]=3

l='q_5_1'
new_train.loc[(new_train[l] <= 9),l]=0
new_train.loc[(new_train[l] > 9) &(new_train[l] <= 25),l]=1
new_train.loc[(new_train[l] > 25) &(new_train[l] < 40),l]=2
new_train.loc[(new_train[l] >= 40),l]=3

l='q_6_1'
new_train.loc[(new_train[l] > -5) &(new_train[l] < 21),l]=2
new_train.loc[(new_train[l] <= -25),l]=0
new_train.loc[(new_train[l] > -25) &(new_train[l] <= -5),l]=1
new_train.loc[(new_train[l] >= 21),l]=3

l='q_7_1'
new_train.loc[(new_train[l] <= 38),l]=0
new_train.loc[(new_train[l] > 38) &(new_train[l] <= 63),l]=1
new_train.loc[(new_train[l] > 63) &(new_train[l] < 79),l]=2
new_train.loc[(new_train[l] >= 79),l]=3

l='q_8_1'
new_train.loc[(new_train[l] <= 2),l]=0
new_train.loc[(new_train[l] > 2) &(new_train[l] <= 5),l]=1
new_train.loc[(new_train[l] > 5) &(new_train[l] < 10),l]=2
new_train.loc[(new_train[l] >= 10),l]=3

'''l='q_8_2'
new_train.loc[(new_train[l] <= 0),l]=0
new_train.loc[(new_train[l] > 0) &(new_train[l] < 20),l]=1
#new_train.loc[(new_train[l] > 25) &(new_train[l] < 40),l]=2
new_train.loc[(new_train[l] >= 40),l]=3
'''
threshold=0.05
print("#############################################################################################################################")
for i in range(1,8):
	for s in range(0,25):
		l=new_listNine[i]
		#print(l)
		'''#creo i 4 gruppi diversi
		new_train.loc[(new_train[l]< -25),l]=0
		new_train.loc[(new_train[l] < 0),l]=1
		new_train.loc[(new_train[l] > 24),l]=3
		new_train.loc[(new_train[l] > 3),l]=2'''
		
		#ANOVA sui 4 gruppi per ogni variabile
		
		temp=new_train[new_train['type']==s]
		#stat,p=stats.f_oneway(temp['pref'][temp[l] == 0], temp['pref'][temp[l] == 1], temp['pref'][temp[l] == 2], temp['pref'][temp[l] == 3])
		#_, normality = shapiro(temp['pref'][temp[l] == 0])
		#print("%2d. Evaluating %s in scenario %s \t pvalue p1 %.3E \t reject: %r " % (s,l,list[s],p, p<threshold))

		print(f"{list[s]} & {new_listNine_new[i]} &", end="")
		p=compareEq(temp['pref'][temp[l] == 0],temp['pref'][temp[l] == 1])[1]
		if p<threshold : print("\\textbf{", end="")
		print(f"{p:.4f}", end="")
		if p<threshold : print("}", end="")
		print(" &", end="")
		
		p=compareEq(temp['pref'][temp[l] == 0],temp['pref'][temp[l] == 2])[1]
		if p<threshold : print("\\textbf{", end="")
		print(f"{p:.4f}", end="")
		if p<threshold : print("}", end="")
		print(" &", end="")
		p=compareEq(temp['pref'][temp[l] == 0],temp['pref'][temp[l] == 3])[1]
		if p<threshold : print("\\textbf{", end="")
		print(f"{p:.4f}", end="")
		if p<threshold : print("}", end="")
		print(" &", end="")
		p=compareEq(temp['pref'][temp[l] == 1],temp['pref'][temp[l] == 2])[1]
		if p<threshold : print("\\textbf{", end="")
		print(f"{p:.4f}", end="")
		if p<threshold : print("}", end="")
		print(" &", end="")
		p=compareEq(temp['pref'][temp[l] == 1],temp['pref'][temp[l] == 3])[1]
		if p<threshold : print("\\textbf{", end="")
		print(f"{p:.4f}", end="")
		if p<threshold : print("}", end="")
		print(" &", end="")
		p=compareEq(temp['pref'][temp[l] == 2],temp['pref'][temp[l] == 3])[1]
		if p<threshold : print("\\textbf{", end="")
		print(f"{p:.4f}", end="")
		if p<threshold : print("}", end="")
		print(" \\\\")

#exit()	
print("#############################################################################################################################")
'''l=new_listNine[7]
print(l)
new_train.loc[(new_train[l]< 25),l]=0
new_train.loc[(new_train[l] > 75),l]=3
new_train.loc[(new_train[l] > 50),l]=2
new_train.loc[(new_train[l] >= 25),l]=1

for s in range(0,25):
	temp=new_train[new_train['type']==s]
	stat,p=stats.f_oneway(temp['pref'][temp[l] == 0], temp['pref'][temp[l] == 1], temp['pref'][temp[l] == 2], temp['pref'][temp[l] == 3])
	print("%2d. Evaluating %s in scenario %s \t pvalue p1 %.3E \t reject: %r" % (s,l,list[s],p, p<threshold))'''
	
	
''''''
#Grouping values for minutes
'''
l=new_listNine[8]
print(l)
new_train.loc[(new_train[l]< 25),l]=0
new_train.loc[(new_train[l] > 75),l]=3
new_train.loc[(new_train[l] > 50),l]=2
new_train.loc[(new_train[l] >= 25),l]=1

'''''
#Grouping values for seconds
'''
l=new_listNine[9]
print(l)
new_train.loc[(new_train[l]< 25),l]=0
new_train.loc[(new_train[l] > 75),l]=3
new_train.loc[(new_train[l] > 50),l]=2
new_train.loc[(new_train[l] >= 25),l]=1

for i in range(1,len(new_listNine)-2):
	print("******************** " + new_listNine[i] +  " ************************")
	print()
	mc = MultiComparison( new_train['pref'],new_train[new_listNine[i]])
	mc_results = mc.tukeyhsd()
	print(mc_results)
	'''
	
print("null-hypothesis: given a scenario Deli-Airport the evaluation variable does not affet the preference")

'''for i in range(1,7):
	l=new_listNine[i]
	print(l)
	#creo i 4 gruppi diversi
	new_train.loc[(new_train[l]< -25),l]=0
	new_train.loc[(new_train[l] < 0),l]=1
	new_train.loc[(new_train[l] > 24),l]=3
	new_train.loc[(new_train[l] > 3),l]=2'''

deli_data=new_train[(new_train['type']==5) | (new_train['type']==6) | (new_train['type']==10) | (new_train['type']==11)]

bath_data=new_train[(new_train['type']==13) | (new_train['type']==14) | (new_train['type']==16) | (new_train['type']==17)] #scenario bathroom
air_data=new_train[(new_train['type']==19) | (new_train['type']==20) | (new_train['type']==21) | (new_train['type']==24)] #scenario airpor

for i in range(1,8):
	l=new_listNine[i]
	#print(l)
	stat,p=stats.f_oneway(deli_data['pref'][deli_data[l] == 0], air_data['pref'][air_data[l] == 0])
	#print(f"============= VARIABLE {l} ===============")
	print(f"{new_listNine_new[i]} &", end="")
	#print("Pvalue p1 %.3E \t reject: %r" % (p, p<threshold))
	if p<threshold : print("\\textbf{", end="")
	print(f"{p:.4f}", end="")
	if p<threshold : print("}", end="")
	print(" &", end="")
	stat,p=stats.f_oneway(deli_data['pref'][deli_data[l] == 1], air_data['pref'][air_data[l] == 1])
	if p<threshold : print("\\textbf{", end="")
	print(f"{p:.4f}", end="")
	if p<threshold : print("}", end="")
	print(" &", end="")
	#print("Pvalue p1 %.3E \t reject: %r" % (p, p<threshold))
	stat,p=stats.f_oneway(deli_data['pref'][deli_data[l] == 2], air_data['pref'][air_data[l] == 2])
	if p<threshold : print("\\textbf{", end="")
	print(f"{p:.4f}", end="")
	if p<threshold : print("}", end="")
	print(" &", end="")
	#print("Pvalue p1 %.3E \t reject: %r" % (p, p<threshold))
	stat,p=stats.f_oneway(deli_data['pref'][deli_data[l] == 3], air_data['pref'][air_data[l] == 3])
	if p<threshold : print("\\textbf{", end="")
	print(f"{p:.4f}", end="")
	if p<threshold : print("}", end="")
	print(" \\\\")
	#print("Pvalue p1 %.3E \t reject: %r" % (p, p<threshold))

'''for i in range(1,8):
	l=new_listNine[i]
	print(l)
	stat,p=stats.f_oneway(bath_data['pref'][bath_data[l] == 0], air_data['pref'][air_data[l] == 0])
	print(f"============= VARIABLE {l} ===============")
	print("Pvalue p1 %.3E \t reject: %r" % (p, p<threshold))
	stat,p=stats.f_oneway(bath_data['pref'][bath_data[l] == 1], air_data['pref'][air_data[l] == 1])
	print("Pvalue p1 %.3E \t reject: %r" % (p, p<threshold))
	stat,p=stats.f_oneway(bath_data['pref'][bath_data[l] == 2], air_data['pref'][air_data[l] == 2])
	print("Pvalue p1 %.3E \t reject: %r" % (p, p<threshold))
	stat,p=stats.f_oneway(bath_data['pref'][bath_data[l] == 3], air_data['pref'][air_data[l] == 3])
	print("Pvalue p1 %.3E \t reject: %r" % (p, p<threshold))'''


	
'''mc = MultiComparison( new_train['pref'],new_train[new_listNine[i]])
mc_results = mc.tukeyhsd()
print(mc_results)'''


