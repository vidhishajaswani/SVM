#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 19:10:07 2018

@author: vidhishajaswani
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D
import math




# read the data
df=pd.read_csv("vjaswan.csv",header=None)
df.columns=['X1','X2','L']
X1=df.iloc[:,0].values
X2=df.iloc[:,1].values
labels=df.iloc[:,2].values


print("-------------Task 1: Scatter Plot---------------------------")
plt.scatter(df.iloc[:,0],df.iloc[:,1],c=df.iloc[:,2])
plt.xlabel('X1')
plt.ylabel('X2')
plt.title("Scatter plot of Original Data")
plt.show()

print("-------------Task 2: Normalize to [0,1]---------------------")
normalize=preprocessing.MinMaxScaler()
df[['X1','X2']]=normalize.fit_transform(df[['X1','X2']])


#scatter plot for normalized data
plt.scatter(df.iloc[:,0],df.iloc[:,1],c=df.iloc[:,2])
plt.xlabel('X1')
plt.ylabel('X2')
plt.title("Scatter plot of Normalized Data")
plt.show()


print("-------------Task 3: Apply SVM------------------------------")
#generate values for C and Gamma

c=[]
for i in range(-5,16,2):
    c.append(1*pow(2,i))
    
gamma=[]
for i in range(-15,5,2):
    gamma.append(1*pow(2,i))


'''apply to SVM using grid search that takes as parameters C, Gamma, 
Type of Kernel, and n for n-cross validation'''

params = {'C':c,'gamma':gamma, 'kernel':['rbf']}
model = GridSearchCV(SVC(),params,cv=5)
model.fit(df[['X1','X2']],labels)
print("Best parameters are",model.best_params_)
print("Highest Accuracy is",model.cv_results_['mean_test_score'].max()*100)

#storing the values of c, gamma and their accuracies
all_c=[]
all_gamma=[]    
all_accuracies=[]
for i in range(len(model.cv_results_['params'])):
    all_c.append((model.cv_results_['params'][i].get('C')))
    all_gamma.append((model.cv_results_['params'][i].get('gamma')))
    all_accuracies.append(model.cv_results_['mean_test_score'][i]*100)

#3-D plot for C, Gamma, and Accuracy
fig = plt.figure()
ax = Axes3D(fig)
a=ax.plot_trisurf(all_c,all_gamma,all_accuracies,cmap='viridis')
fig.colorbar(a)
plt.title("3D Scatter Plot") 
ax.set_xlabel('X where C=2^X');
ax.set_ylabel('Y where gamma=2^Y');
ax.set_zlabel('Accuracy in %');
plt.show()

print("-----------Test for best parameters in small range-----------")
#uncomment below line 92 and 93 and comment line 97 and 98 to test for below ranges

#updated_c=[0.03125,0.125,0.5,2]
#updated_gamma=[0.125,0.5,2,4,8]

#updating the model
updated_c=[0.95,0.99,0.10,0.110,0.125]
updated_gamma=[1.85,1.9,1.95,2,2.1]

updated_params = {'C':updated_c,'gamma':updated_gamma, 'kernel':['rbf']}
updated_model = GridSearchCV(SVC(),updated_params,cv=5)
updated_model.fit(df[['X1','X2']],labels)
print("Best Parameters for Updated Model",updated_model.best_params_)
print("Accuracy for Updated Model",updated_model.cv_results_['mean_test_score'].max()*100)

#recording updated values of C, Gamma, and Accuracies
updated_all_c=[]
updated_all_gamma=[]
updated_all_accuracies=[]
for i in range(len(updated_model.cv_results_['params'])):
    updated_all_c.append((updated_model.cv_results_['params'][i].get('C')))
    updated_all_gamma.append((updated_model.cv_results_['params'][i].get('gamma')))
    updated_all_accuracies.append(updated_model.cv_results_['mean_test_score'][i]*100)

#3-D plot for the updated model
fig = plt.figure()
ax = Axes3D(fig)
a1=ax.plot_trisurf(updated_all_c,updated_all_gamma,updated_all_accuracies,cmap='rainbow')
fig.colorbar(a1)
plt.title("3D Scatter Plot for the updated model")  
ax.set_xlabel('C');
ax.set_ylabel('Gamma');
ax.set_zlabel('Accuracy');
plt.show()




