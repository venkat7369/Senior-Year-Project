from __future__ import print_function
import numpy as np    
import csv
import copy
import pandas as pd
import random
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, ExpSineSquared
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared
from sklearn.pipeline import Pipeline
from keras import Sequential
from keras.layers import Dense
from keras.layers import Dropout


import tensorflow as tf
import tensorflow 

from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

# Read Data

ifile  = open('Calculated_data_25.csv', "rt")
reader = csv.reader(ifile)
csvdata=[]
for row in reader:
        csvdata.append(row)
ifile.close()
numrow=len(csvdata)
numcol=len(csvdata[0])
csvdata = np.array(csvdata).reshape(numrow,numcol)
dopant = csvdata[1:,0]
E_form  = csvdata[1:,29]
X = csvdata[1:,1:28]

   
 # Read Outside Data
ifile  = open('Predicted-data.csv', "rt")
reader = csv.reader(ifile)
csvdata=[]
for row in reader:
        csvdata.append(row)
ifile.close()
numrow=len(csvdata)
numcol=len(csvdata[0])
csvdata = np.array(csvdata).reshape(numrow,numcol)
dopant_out = csvdata[1:,0]
X_out = csvdata[1:,1:17]

n_out = dopant_out.size

data_out = copy.deepcopy(csvdata);



XX = copy.deepcopy(X)
prop = copy.deepcopy(E_form)
n = dopant.size
m = int(X.size/n)
m_y = 1

rmse_train_form  =  [[[0.0 for a in range(10)] for b in range(10)] for c in range(100)]
rmse_test_form  =  [[[0.0 for a in range(10)] for a in range(10)] for c in range(100)]

t = 0.2

X_train, X_test, E_form_train, E_form_test  = train_test_split(XX, prop, test_size=t)
n_tr = E_form_train.size
n_te = E_form_test.size
# print(X_train)

X_train_fl = [[0.0 for a in range(m)] for b in range(n_tr)]
for i in range(0,n_tr):
    for j in range(0,m):
        # print(X_train[i][j])
        try:
            X_train_fl[i][j] = float(X_train[i][j])
        except:
            X_train_fl[i][j] = 0
            continue

X_test_fl = [[0.0 for a in range(m)] for b in range(n_te)]
for i in range(0,n_te):
    for j in range(0,m):
        # print(X_test[i][j])
        try:
            X_test_fl[i][j] = float(X_test[i][j])
        except:
            X_test_fl[i][j] = 0

## NN Optimizers and Model Definition


pipelines = []

dp = [0.00, 0.10, 0.20]
n1 = [50, 100, 150]
n2 = [50, 100, 150]
lr = [0.001, 0.01, 0.1]
ep = [200, 400, 600]
bs = [50, 100, 200]

dp = [0.20]
n1 = [50, 100]
n2 = [50, 100]
lr = [0.001]
ep = [200]
bs = [100]





for a in range(0,len(lr)):
    for b in range(0,len(n1)):
        for c in range(0,len(dp)):
            for d in range(0,len(n2)):
                for e in range(0,len(ep)):
                    for f in range(0,len(bs)):
                        
                        keras.optimizers.Adam(learning_rate=lr[a], beta_1=0.9, beta_2=0.999, amsgrad=False)

                        # define base model
                        def baseline_model():
                            model = Sequential()
                            model.add(Dense(m, input_dim=m, kernel_initializer='normal', activation='relu'))
                            model.add(Dense(n1[b], kernel_initializer='normal', activation='relu'))
                            model.add(Dropout(dp[c], input_shape=(m,)))
                            model.add(Dense(n2[d], kernel_initializer='normal', activation='relu'))
                            model.add(Dense(m_y, kernel_initializer='normal'))
                            model.compile(loss='mean_squared_error', optimizer='Adam')
                            return model

                        # evaluate model with standardized dataset
                        estimators = []
                        estimators.append(('standardize', sklearn.preprocessing.StandardScaler()))
                        estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=ep[e], batch_size=bs[f], verbose=0)))
                        pipelines.append ( Pipeline(estimators) )





 ##  Train E_form Model  ##

Prop_train = copy.deepcopy(E_form_train)
Prop_test  = copy.deepcopy(E_form_test)

Prop_train_fl = np.zeros(n_tr)
for i in range(0,n_tr):
    try:
        Prop_train_fl[i] = copy.deepcopy(float(Prop_train[i]))
    except:
        Prop_train_fl[i] = 0

Prop_test_fl = np.zeros(n_te)
for i in range(0,n_te):
    try:
        Prop_test_fl[i] = copy.deepcopy(float(Prop_test[i]))
    except:
        Prop_test_fl[i] = 0

pipeline = pipelines[np.random.randint(0,4)]

pipeline.fit(X_train_fl, Prop_train_fl)
Pred_train = pipeline.predict(X_train_fl)
Pred_test  = pipeline.predict(X_test_fl)
Pred_train_fl = [0.0]*n_tr
Pred_test_fl = [0.0]*n_te
for i in range(0,n_tr):
    Pred_train_fl[i] = float(Pred_train[i])
for i in range(0,n_te):
    Pred_test_fl[i] = float(Pred_test[i])

Prop_train_form = copy.deepcopy(Prop_train_fl)
Pred_train_form = copy.deepcopy(Pred_train_fl)
Prop_test_form  = copy.deepcopy(Prop_test_fl)
Pred_test_form  = copy.deepcopy(Pred_test_fl)

## Outside Predictions

X_out_fl = [[0.0 for a in range(m)] for b in range(n_out)]
for i in range(0,n_out):
    for j in range(0,m):
        if i < len(X_out) and j < len(X_out[i]):
            X_out_fl[i][j] = float(X_out[i][j])


Pred_out_fl  =  [[0.0 for a in range(1)] for b in range(n_out)]
Pred_out_str  =  [[0.0 for a in range(1)] for b in range(n_out)]
err_up_out   =  [[0.0 for a in range(1)] for b in range(n_out)]
err_down_out =  [[0.0 for a in range(1)] for b in range(n_out)]


Pred_out = pipeline.predict(X_out_fl)
for i in range(0,n_out):
    Pred_out_fl[i] = float(Pred_out[i])

for i in range(0,n_out):
    Pred_out_str[i] = str(Pred_out[i])

dopant_out_array = np.array(dopant_out)
Pred_out_array = np.array(Pred_out)


#Pred_dopant_out = [[dopant_out_array[i], Pred_out_array[i]] for i in range(len(dopant_out_array))]
Pred_dopant_out = np.column_stack((dopant_out_array, Pred_out_array))
np.savetxt('Pred_out.csv', Pred_dopant_out, fmt='%s')


mse_test_prop  = sklearn.metrics.mean_squared_error(Prop_test_form, Pred_test_form)
mse_train_prop = sklearn.metrics.mean_squared_error(Prop_train_form, Pred_train_form)
rmse_test_form  = np.sqrt(mse_test_prop)
rmse_train_form = np.sqrt(mse_train_prop)
'''print('rmse_test_form  = ', np.sqrt(mse_test_prop))
print('rmse_train_form = ', np.sqrt(mse_test_prop))
print('      ')'''
 


## ML Parity Plots ##


#fig, ( [ax1, ax2], [ax3, ax4], [ax5, ax6] ) = plt.subplots( nrows=3, ncols=2, figsize=(6,6) )

#fig, ( [ax1, ax2], [ax3, ax4] ) = plt.subplots( nrows=2, ncols=2, sharex=True, sharey=True, figsize=(8,8) )

fig, ( ax ) = plt.subplots( nrows=1, ncols=1, figsize=(8,10) )

fig.text(0.5, 0.02, 'DFT Calculation', ha='center', fontsize=20)
fig.text(0.02, 0.5, 'ML Prediction', va='center', rotation='vertical', fontsize=20)


plt.subplots_adjust(left=0.12, bottom=0.10, right=0.97, top=0.95, wspace=0.25, hspace=0.4)
plt.rc('font', family='sans-serif')


Prop_train_temp = copy.deepcopy(Prop_train_form)
Pred_train_temp = copy.deepcopy(Pred_train_form)
Prop_test_temp  = copy.deepcopy(Prop_test_form)
Pred_test_temp  = copy.deepcopy(Pred_test_form)

a = [-175,0,125]
b = [-175,0,125]
ax.plot(b, a, c='k', ls='-')

ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)

ax.scatter(Prop_test_temp[:], Pred_test_temp[:], c='orange', marker='s', s=60, label='Test')
ax.scatter(Prop_train_temp[:], Pred_train_temp[:], c='blue', marker='s', s=60, label='Train')


te = '%.2f' % rmse_test_form
tr = '%.2f' % rmse_train_form

ax.set_ylim([-3, 4])
ax.set_xlim([-3, 4])

ax.text(4.27, 0.65, 'Test_rmse = ', c='r', fontsize=10)
ax.text(6.75, 0.65, te, c='r', fontsize=10)
ax.text(7.67, 0.65, 'eV', c='r', fontsize=10)
ax.text(4.14, -0.18, 'Train_rmse = ', c='r', fontsize=10)
ax.text(6.75, -0.18, tr, c='r', fontsize=10)
ax.text(7.67, -0.18, 'eV', c='r', fontsize=10)

ax.set_xticks([0, 2, 4, 6, 8])
ax.set_yticks([0, 2, 4, 6, 8])

ax.set_title('NN Band Gap', c='k', fontsize=16, pad=10)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles,labels)

plt.savefig('plot.png')  # Save the plot as a file
plt.show()  # Display the plot