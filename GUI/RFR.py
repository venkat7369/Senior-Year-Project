# Modified code

from __future__ import print_function
import numpy as np    
import csv
import copy
import pandas
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


# Read Data

def predict_RFR(csvdata):

    # ifile  = open('Calculated-data.csv', "rt")
    # reader = csv.reader(ifile)
    # csvdata=[]
    # for row in reader:
    #         csvdata.append(row)
    # ifile.close()
    numrow=len(csvdata)
    numcol=len(csvdata[0])
    csvdata = np.array(csvdata).reshape(numrow,numcol)
    dopant = csvdata[1:,0]
    E_form  = csvdata[1:,18]
    X = csvdata[1:,1:17]

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

    t = 0.2

    X_train, X_test, E_form_train, E_form_test  = train_test_split(XX, prop, test_size=t)
    n_tr = E_form_train.size
    n_te = E_form_test.size

    X_train_fl = [[0.0 for a in range(m)] for b in range(n_tr)]
    for i in range(0,n_tr):
        for j in range(0,m):
            X_train_fl[i][j] = float(X_train[i][j])

    X_test_fl = [[0.0 for a in range(m)] for b in range(n_te)]
    for i in range(0,n_te):
        for j in range(0,m):
            X_test_fl[i][j] = float(X_test[i][j])
            


        ####      Define Random Forest Hyperparameter Space     ####


    '''param_grid = {
    "n_estimators": [100, 200, 500],
    "max_features": [10, 15, m],
    "min_samples_leaf": [5,10,20],
    "max_depth": [5,10,15],
    "min_samples_split": [2, 5, 10]
    }'''

    param_grid = {
    "n_estimators": [100],
    "max_features": [15],
    "max_depth": [10]
    }


    ##  Train E_form Model  ##

    Prop_train = copy.deepcopy(E_form_train)
    Prop_test  = copy.deepcopy(E_form_test)

    Prop_train_fl = np.zeros(n_tr)
    for i in range(0,n_tr):
        Prop_train_fl[i] = copy.deepcopy(float(Prop_train[i]))

    Prop_test_fl = np.zeros(n_te)
    for i in range(0,n_te):
        Prop_test_fl[i] = copy.deepcopy(float(Prop_test[i]))
        

    rfreg_opt = GridSearchCV(RandomForestRegressor(), param_grid=param_grid, cv=5)

    rfreg_opt.fit(X_train_fl,Prop_train_fl)
    Pred_train_fl = rfreg_opt.predict(X_train_fl)
    Pred_test_fl  = rfreg_opt.predict(X_test_fl)

    feature_importances = [0.0 for a in range(0,n)]

    for i in range(0,n):
        feature_importances[i] = rfreg_opt.best_estimator_.feature_importances_[i]
        
    print('feature_importances', feature_importances)
    print('      ')


    Prop_train_form = copy.deepcopy(Prop_train_fl)
    Pred_train_form = copy.deepcopy(Pred_train_fl)
    Prop_test_form  = copy.deepcopy(Prop_test_fl)
    Pred_test_form  = copy.deepcopy(Pred_test_fl)


    ## Outside Predictions

    X_out_fl = [[0.0 for a in range(m)] for b in range(n_out)]
    for i in range(0,n_out):
        for j in range(0,m):
            X_out_fl[i][j] = float(X_out[i][j])


    Pred_out_fl  =  [[0.0 for a in range(1)] for b in range(n_out)]
    err_up_out   =  [[0.0 for a in range(1)] for b in range(n_out)]
    err_down_out =  [[0.0 for a in range(1)] for b in range(n_out)]


    Pred_out = rfreg_opt.predict(X_out_fl)
    for i in range(0,n_out):
        Pred_out_fl[i] = float(Pred_out[i])


    np.savetxt('Pred_out.csv', Pred_out_fl)


    mse_test_prop  = sklearn.metrics.mean_squared_error(Prop_test_form, Pred_test_form)
    mse_train_prop = sklearn.metrics.mean_squared_error(Prop_train_form, Pred_train_form)
    rmse_test_form  = np.sqrt(mse_test_prop)
    rmse_train_form = np.sqrt(mse_train_prop)
    print('rmse_test_form  = ', np.sqrt(mse_test_prop))
    print('rmse_train_form = ', np.sqrt(mse_train_prop))
    print('      ')




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

    ax.set_title('Formation Energy', c='k', fontsize=16, pad=10)





    plt.savefig('plot.png', dpi=450)
    plt.show()