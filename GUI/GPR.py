from __future__ import print_function
import numpy as np
import csv
import copy
import pandas as pd
import random
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import train_test_split, cross_val_score, KFold


def predict_GPR(csvdata):

    # Read Data

    # ifile  = open('calcdata.csv', "rt")
    # reader = csv.reader(ifile)
    # csvdata=[]
    # for row in reader:
    #     csvdata.append(row)
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

    data_out = copy.deepcopy(csvdata)

    XX = copy.deepcopy(X)
    prop = copy.deepcopy(E_form)
    n = dopant.size
    m = int(X.size/n)

    t = 0.2

    X_train, X_test, E_form_train, E_form_test = train_test_split(XX, prop, test_size=t)
    n_tr = E_form_train.size
    n_te = E_form_test.size

    X_train_fl = X_train.astype(float)
    X_test_fl = X_test.astype(float)

    # Define Gaussian Process Regression with RBF kernel and white noise
    kernel = 1.0 * RBF() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel)



    # Define Gaussian Process Regression with RBF kernel and white noise
    kernel = 1.0 * RBF() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel)

    # Convert data to float
    XX = X.astype(float)
    prop = E_form.astype(float)

    # Specify the number of folds
    n_folds =  4

    # custom scoring function for RMSE
    def rmse_scorer(model, X, y):
        y_pred, _ = model.predict(X, return_std=True)
        mse = sklearn.metrics.mean_squared_error(y, y_pred)
        return np.sqrt(mse)

    # k-fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)  
    rmse_scores = cross_val_score(gpr, XX, prop, cv=kf, scoring=rmse_scorer)  

    # Mean and standard deviation RMSE scores
    mean_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores)

    print(f"Mean RMSE: {mean_rmse:.3f}")
    print(f"Standard Deviation of RMSE: {std_rmse:.3f}")












    # Train E_form Model
    Prop_train = E_form_train.astype(float)
    Prop_test = E_form_test.astype(float)

    # Fit the Gaussian Process model
    gpr.fit(X_train_fl, Prop_train)
    Pred_train_fl, sigma = gpr.predict(X_train_fl, return_std=True)
    Pred_test_fl, sigma = gpr.predict(X_test_fl, return_std=True)

    Prop_train_form = Prop_train
    Pred_train_form = Pred_train_fl
    Prop_test_form = Prop_test
    Pred_test_form = Pred_test_fl

    # Outside Predictions
    X_out_fl = X_out.astype(float)
    Pred_out_fl, sigma = gpr.predict(X_out_fl, return_std=True)


    np.savetxt('Pred_out.csv', Pred_out_fl)



    output_df = pd.DataFrame({"Dopant": dopant_out, "Formation Energy": Pred_out_fl})
    output_df.to_csv('Output.csv', index=False)

    # Calculate RMSE for test and train data
    mse_test_prop = sklearn.metrics.mean_squared_error(Prop_test_form, Pred_test_form)
    mse_train_prop = sklearn.metrics.mean_squared_error(Prop_train_form, Pred_train_form)
    rmse_test_form = round(np.sqrt(mse_test_prop), 3)
    rmse_train_form = round(np.sqrt(mse_train_prop), 3)
    print('rmse_test_form  = ', rmse_test_form)
    print('rmse_train_form = ', rmse_train_form)




    # ML Parity Plots
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 10))


    fig.text(0.5, 0.02, 'DFT Calculation', ha='center', fontsize=20)
    fig.text(0.02, 0.5, 'ML Prediction', va='center', rotation='vertical', fontsize=20)

    plt.subplots_adjust(left=0.12, bottom=0.10, right=0.97, top=0.95, wspace=0.25, hspace=0.4)
    plt.rc('font', family='sans-serif')

    Prop_train_temp = copy.deepcopy(Prop_train_form)
    Pred_train_temp = copy.deepcopy(Pred_train_form)
    Prop_test_temp = copy.deepcopy(Prop_test_form)
    Pred_test_temp = copy.deepcopy(Pred_test_form)

    a = [-175, 0, 125]
    b = [-175, 0, 125]
    ax.plot(b, a, c='k', ls='-')

    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)

    ax.scatter(Prop_test_temp, Pred_test_temp, c='blue', marker='s', s=60, label='Test')
    ax.scatter(Prop_train_temp, Pred_train_temp, c='orange', marker='s', s=60, label='Train')

    te = '%.2f' % rmse_test_form
    tr = '%.2f' % rmse_train_form

    ax.set_ylim([-3, 4])
    ax.set_xlim([-3, 4])

    ax.text(4.27, 0.65, 'Test_rmse = ', c='b', fontsize=10)
    ax.text(6.65, 0.65, te, c='b', fontsize=10)
    ax.text(7.67, 0.65, 'eV', c='b', fontsize=10)
    ax.text(4.14, -0.18, 'Train_rmse = ', c='b', fontsize=10)
    ax.text(6.65, -0.18, tr, c='b', fontsize=10)
    ax.text(7.67, -0.18, 'eV', c='b', fontsize=10)

    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_yticks([0, 2, 4, 6, 8])

    ax.legend()


    plt.savefig('plot.png', dpi=450)
    plt.show()


    # def load_gpr_model():
        
    #     kernel = 1.0 * RBF() + WhiteKernel()
    #     gpr = GaussianProcessRegressor(kernel=kernel)

    #     
    #     return gpr