from sklearn import svm
import numpy as np
import random
import copy
from xgboost import XGBClassifier

class empty_pred():
	def __init__(self, value):
		self.value = value

	def predict(self,x_samples):
		return self.value
		

class SEP_svm():
	def __init__(self) -> None:
		self.septables={}
		self.random = 0
		pass

	def group_var(self, dataset):
		new_listNine=['Judgement',"Global Welfare", "First Person Welfare", "Middle Person Welfare", "Last Person Welfare", "Line Cutter Welfare", "Universalization", "Likelihood"]
		l=new_listNine[1] #'q_1_1'
		dataset.loc[(dataset[l] > 0) &(dataset[l] < 18),l]=2
		dataset.loc[(dataset[l] > -20) &(dataset[l] <= 0),l]=1
		dataset.loc[(dataset[l] <= -20),l]=0
		dataset.loc[(dataset[l] >= 18),l]=3

		l=new_listNine[2] #'q_2_1'
		dataset.loc[(dataset[l] > -3) &(dataset[l] < 20),l]=2
		dataset.loc[(dataset[l] <= -12),l]=0
		dataset.loc[(dataset[l] > -12) &(dataset[l] <= -3),l]=1
		dataset.loc[(dataset[l] >= 20),l]=3

		l=new_listNine[3] #'q_3_1'
		dataset.loc[(dataset[l] > -5) &(dataset[l] < 17),l]=2
		dataset.loc[(dataset[l] <= -18),l]=0
		dataset.loc[(dataset[l] > -18) &(dataset[l] <= -5),l]=1
		dataset.loc[(dataset[l] >= 17),l]=3

		l=new_listNine[4] #'q_4_1'
		dataset.loc[(dataset[l] > 0) &(dataset[l] < 18),l]=2
		dataset.loc[(dataset[l] > -20) &(dataset[l] <= 0),l]=1
		dataset.loc[(dataset[l] <= -20),l]=0
		dataset.loc[(dataset[l] >= 18),l]=3

		l=new_listNine[5] #'q_5_1'
		dataset.loc[(dataset[l] <= 9),l]=0
		dataset.loc[(dataset[l] > 9) &(dataset[l] <= 25),l]=1
		dataset.loc[(dataset[l] > 25) &(dataset[l] < 40),l]=2
		dataset.loc[(dataset[l] >= 40),l]=3

		l=new_listNine[6] #'q_6_1'
		dataset.loc[(dataset[l] > -5) &(dataset[l] < 21),l]=2
		dataset.loc[(dataset[l] <= -25),l]=0
		dataset.loc[(dataset[l] > -25) &(dataset[l] <= -5),l]=1
		dataset.loc[(dataset[l] >= 21),l]=3

		l=new_listNine[7] #'q_7_1'
		dataset.loc[(dataset[l] <= 38),l]=0
		dataset.loc[(dataset[l] > 38) &(dataset[l] <= 63),l]=1
		dataset.loc[(dataset[l] > 63) &(dataset[l] < 79),l]=2
		dataset.loc[(dataset[l] >= 79),l]=3

		return dataset
		'''l=new_listNine[8] #'q_8_1'
		dataset.loc[(dataset[l] <= 2),l]=0
		dataset.loc[(dataset[l] > 2) &(dataset[l] <= 5),l]=1
		dataset.loc[(dataset[l] > 5) &(dataset[l] < 10),l]=2
		dataset.loc[(dataset[l] >= 10),l]=3'''

	def fit(self, x_samples, labels):
		self.dataset = copy.deepcopy(x_samples)
		#self.group_var(self.dataset)
		self.dataset['Judgement'] = labels

		self.loc_pref = self.dataset.groupby(['family','Judgement'])['Global Welfare'].count()
		self.loc_reas_pref = self.dataset.groupby(['family','type','Judgement'])['Global Welfare'].count()

		self.dataset = copy.deepcopy(x_samples)

		self.eval_pref={}
		new_listNine=["Global Welfare", "First Person Welfare", "Middle Person Welfare", "Last Person Welfare", "Line Cutter Welfare", "Universalization", "Likelihood"]
		for v in new_listNine:
			#TRAIN FOR EACH SCENARIO
			for scenario in range(0,25):
				
				#print(f"variable {v} \t scenario {scenario}")
				#print(x_samples.loc[x_samples['type']==scenario][v])
				#Each SVM is trained based on (variable,scenario)
				if len(x_samples.loc[x_samples['type']==scenario])>0:
					if len(np.unique(labels.loc[x_samples['type']==scenario]))>1:
						self.eval_pref[(v,scenario)] = svm.SVC(kernel='rbf')
						#self.eval_pref[(v,scenario)] = XGBClassifier()
						self.eval_pref[(v,scenario)].fit(np.array(x_samples.loc[x_samples['type']==scenario][v]).reshape((-1,1)),labels.loc[x_samples['type']==scenario])
					else:
						self.eval_pref[(v,scenario)] = empty_pred(np.unique(labels.loc[x_samples['type']==scenario])[0])
				else:
					print(f"variable {v} \t scenario {scenario}")
					self.eval_pref[(v,scenario)] = None


	def predict(self,x_samples):
		prediction = []
		temp_df = x_samples

		for index, r in temp_df.iterrows():

			#print(r)
			zeros = 0
			ones = 0
			if self.loc_pref[(r['family'],0.0)] > self.loc_pref[(r['family'],1.0)]:
				zeros = zeros + 1
			else:
				ones = ones + 1

			new_listNine=["Global Welfare", "First Person Welfare", "Middle Person Welfare", "Last Person Welfare", "Universalization", "Likelihood"]
			for v in new_listNine:

				if self.eval_pref[(v,r['type'])] :
					pred = self.eval_pref[(v,r['type'])].predict(np.array(r[v]).reshape((1,-1)))
				else: 
					pred = random.randint(0,1)
					self.random += 1
				#print(pred)
				if pred == 0:
					zeros = zeros + 1
				else:
					ones = ones + 1
				
			if zeros > ones:
				prediction.append(0)
			else:
				prediction.append(1)
		return prediction