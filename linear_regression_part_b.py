# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 08:25:23 2021

@author: ma-ty
"""

from matplotlib.pyplot import figure, plot, xlabel, ylabel, show, subplot, semilogx, title, grid, legend, suptitle, tight_layout
import numpy as np
import pandas as pd
#from scipy.io import loadmat
#from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
#from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import OneHotEncoder
from toolbox_02450 import mcnemar,  train_neural_net, draw_neural_net
import torch
#from scipy import stats

# Gets data
#from read_ecoli_data_2 import load
df = pd.read_excel("Real estate valuation data set.xlsx")
df = df.drop(['No'], axis=1)
raw_data = np.array(df)
cols = range(0,6) 
X = raw_data[:, 0:6]
y = raw_data[:,6]
N, M = X.shape

# Set attribute names
attributeNames = np.asarray(df.columns[cols])
strs = ["" for x in range(M)]
for i in range(M):
    strs[i] = attributeNames[i]
attributeNames = strs


# Standardizes data matrix so each column has mean 0 and std 1
X = (X - np.ones((N,1))*X.mean(0))/X.std(0)

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M = M+1

def inner_loop(k_inner, ):
    pass


if __name__ == '__main__':

    # Parameters for neural network classifier
    n_replicates = 1        # number of networks trained in each k-fold
    max_iter = 10000        # 

   # X, y, C, N, M, attributeNames = load()

    # one hot encoding of cell type
    ohe = OneHotEncoder(sparse=False,categories='auto')
    x = ohe.fit_transform(y.reshape(-1, 1))

    # Set cross-validation parameters
    K1 = 5
    K2 = K1
    CV = model_selection.KFold(n_splits=K1,shuffle=True, random_state = 1)
    CV2 = model_selection.KFold(n_splits=K2,shuffle=True, random_state = 1)


    # Initialize error and complexity control - LOGREG  #(LINREG)
    lambdas = np.logspace(0, 1, 10)
    #lambdas = np.power(10.,range(0,1))
    L_LIN = len(lambdas) 

    error2_LIN = np.zeros((K2,len(lambdas)))
    error_LIN = np.zeros((K1))
    min_error_LIN = np.zeros(K1)
    s_LIN = np.zeros(K1)
    opt_lambda = np.zeros(K1)


    # Initialize error and complexity control - ANN
    L = 6  # Maximum number of hidden units
    L_list = np.arange(1,L+1,1)
    errors_ANN = np.zeros((K1))
    errors2_ANN = np.zeros((K2,L))
    s_ANN = np.zeros(K1)
    x_ANN = [0] * K1

    # Initialize error and complexity control - baseline
    errors_baseline = np.zeros((K1))

    yhat = []
    y_true = []
    n = 0
    ANN_min_errors = []
    for train_index, test_index in CV.split(X):

        # extract training and test set for current CV fold
        X_train_lin = X[train_index,:]
        y_train_lin = y[train_index]
        X_test_lin = X[test_index,:]
        y_test_lin = y[test_index]

        X_train_outer = torch.tensor(X[train_index,1:], dtype=torch.float)
        y_train_outer = torch.tensor(y[train_index], dtype=torch.float)
        X_test_outer = torch.tensor(X[test_index,1:], dtype=torch.float)
        y_test_outer = torch.tensor(y[test_index], dtype=torch.float)
        
        
        i = 0
        w = np.empty((M,K2,len(lambdas))) # Changed from K1 to K2
#        y = y.squeeze()

        
        for train_index2, test_index2 in CV2.split(X_train_lin):
            print('Crossvalidation fold: {0}/{1}'.format(n+1,i+1))    
            
            ANN_errors = []
            # Extract training and test set for current CV fold, convert to tensors
            X_train2 = torch.tensor(X[train_index2,1:], dtype=torch.float)
            y_train2 = torch.tensor(y[train_index2], dtype=torch.float)
            X_test2 = torch.tensor(X[test_index2,1:], dtype=torch.float)
            y_test2 = torch.tensor(y[test_index2], dtype=torch.float)
        
            # ANN
            # Fit classifier and classify the test points (consider 1 to 40 neighbors)
            for h in range(1,L+1):

                n_hidden_units = h
                model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M-1, n_hidden_units), #M features to H hiden units
                        # 1st transfer function, either Tanh or ReLU:
                        #torch.nn.ReLU(), 
                        torch.nn.Tanh(),   
                        torch.nn.Linear(n_hidden_units, 1), # H hidden units to 1 output neuron
                        )
                
                
                loss_fn = torch.nn.MSELoss()

                net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train2,
                                                               y=y_train2,
                                                               n_replicates=1,
                                                               max_iter=max_iter,
                                                               tolerance = 1e-12)
                
                y_test_est = net(X_test2)
                y_test_est = y_test_est.reshape(-1)

                se = (y_test_est.float()-y_test2.float())**2 # squared error
                mse = (sum(se).type(torch.float)/len(y_test2)).data.numpy() #mean
                
                # Set error matrix
                errors2_ANN[i,h-1] = mse
                
            
            
            X_train_lin_in = X[train_index2,:]
            y_train_lin_in = y[train_index2]
            X_test_lin_in = X[test_index2,:]
            y_test_lin_in = y[test_index2]
                
                
            # LINEAR REGRESSION 

            # precompute terms
            Xty = X_train_lin_in.T @ y_train_lin_in
            XtX = X_train_lin_in.T @ X_train_lin_in
        
            for l in range(0,L_LIN):
                # Compute parameters for current value of lambda and current CV fold
                lambdaI = lambdas[l] * np.eye(M)
                lambdaI[0,0] = 0 # remove bias regularization 
                w[:,i,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
                # Evaluate training and test performance
                y_est2_lin = X_test_lin_in @ w[:,i,l].T
                error2_LIN[i,l] = np.power(y_test_lin_in-y_est2_lin,2).mean(axis=0)

            i+=1       
        
        # ANN
        # Find which element corresponds to the smallest error

        minArg = np.argmin(errors2_ANN.mean(0))
        s_ANN[n] = minArg+1
        x_ANN[n] = L_list[minArg]
        
        # Compute the best ANN from the inner fold
        n_hidden_units = L_list[minArg]

        model = lambda: torch.nn.Sequential(
                torch.nn.Linear(M-1, n_hidden_units), #M features to H hiden units
                # 1st transfer function, either Tanh or ReLU:
                #torch.nn.ReLU(), 
                torch.nn.Tanh(),   
                torch.nn.Linear(n_hidden_units, 1), # H hidden units to 1 output neuron
                )
        loss_fn = torch.nn.MSELoss()

        net, final_loss, learning_curve = train_neural_net(model,
                                                        loss_fn,
                                                        X=X_train_outer,
                                                        y=y_train_outer,
                                                        n_replicates=1,
                                                        max_iter=max_iter)
        y_est_ANN = net(X_test_outer)
        y_est_ANN = y_est_ANN.reshape(-1)

        # Determine errors 
        se = (y_est_ANN.float()-y_test_outer.float())**2 # squared error
        mse = (sum(se).type(torch.float)/len(y_test_outer)).data.numpy() #mean
        
        errors_ANN[n] = mse


        # LINEAR MODEL
        
        # precompute terms
        Xty = X_train_lin.T @ y_train_lin
        XtX = X_train_lin.T @ X_train_lin
        
        # Compute lienar regression with best lambda from inner fold        
        minArg = np.argmin(np.mean(error2_LIN,axis=0))
        opt_lambda[n] = lambdas[minArg]
        
        lambdaI = lambdas[minArg] * np.eye(M)
        lambdaI[0,0] = 0 # remove bias regularization 
        w_outer = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
        
        # Evaluate training and test performance
        y_est_lin = X_test_lin @ w_outer.T
        error_LIN[n] = np.power(y_test_lin-y_est_lin,2).mean(axis=0)

#        # BASELINE
        y_est_BASE = np.ones(y_test_lin.shape)*y_test_lin.mean()
        errors_baseline[n] = np.mean((y_est_BASE-y_test_lin)**2)


#        # Combine all predictions in array
        dy = []
        dy.append(y_est_BASE)
        dy.append(y_est_lin)
        dy.append(y_est_ANN.data.numpy())
        dy = np.stack(dy, axis=1)
        yhat.append(dy)
        
        y_true.append(y_test_lin)
        n+=1
    
        
    # combine all predictions and real values
    y_true = np.concatenate(y_true)
    yhat = np.concatenate(yhat)


print('Errors ANN:\tErrors Baseline: \tErrors Linear')
for m in range(K1):   
    print(np.round(errors_ANN[m],3),'\t\t',np.round(errors_baseline[m],3),'\t\t',np.round(error_LIN[m],3))
        
        
   
# PLOTS 
dpi = 75 # Sets dpi for plots
save_plots = False

# Plot results from last inner fold
test_err_vs_lambda = np.mean(error2_LIN,axis=0)
mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))
    
f = figure(dpi = dpi)
subplot(2, 1, 1)
title('Optimal lambda: {0}'.format(np.round(opt_lambda[i-1],3)))
semilogx(lambdas,test_err_vs_lambda.T,'r.-')
xlabel('Regularization factor')
ylabel('Estimated generalization error')
legend(['Validation error'])
grid()
tight_layout()

subplot(2, 1, 2)
title('ANN - inner crossvalidation')
plot(L_list,errors2_ANN.mean(0))
xlabel('Hidden nodes')
ylabel('Estimated generalization error')
legend(['Validation error'])
grid()
tight_layout()


show()
f.savefig('./figures/inner_fold_regression.png', bbox_inches='tight') if save_plots else 0

import seaborn as sns

f1 = figure(dpi=dpi)
boxes = [errors_ANN,errors_baseline,error_LIN,]
boxes_df = pd.DataFrame(boxes).T
boxes_df.columns = ['ANN', 'Baseline', 'Linear Regression']
sns.boxplot(data = boxes_df,palette="Set3")
ylabel('Generalization Error')
f1.savefig('./figures/boxplot_regression.png', bbox_inches='tight') if save_plots else 0

#%% Statistical evaluation Setup I
import numpy as np, scipy.stats as st
    
alpha = 0.05

print('A : Baseline\nB : LIN')
yhatA = yhat[:,0]
yhatB = yhat[:,1]
# compute z with squared error.
zA = np.abs(y_true - yhatA ) ** 2
# Compute confidence interval of z = zA-zB and p-value of Null hypothesis
zB = np.abs(y_true - yhatB ) ** 2
z = zA - zB
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value
print('CI: ',np.round(CI,2),' p: ',p)
print('')

print('A : Baseline\nB : ANN')
yhatA = yhat[:,0]
yhatB = yhat[:,2]
# compute z with squared error.
zA = np.abs(y_true - yhatA ) ** 2
# Compute confidence interval of z = zA-zB and p-value of Null hypothesis
zB = np.abs(y_true - yhatB ) ** 2
z = zA - zB
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value
print('CI: ',np.round(CI,2),' p: ',p)
print('')

print('A : LIN\nB : ANN')
yhatA = yhat[:,1]
yhatB = yhat[:,2]
# compute z with squared error.
zA = np.abs(y_true - yhatA ) ** 2
# Compute confidence interval of z = zA-zB and p-value of Null hypothesis
zB = np.abs(y_true - yhatB ) ** 2
z = zA - zB
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value
print('CI: ',np.round(CI,2),' p: ',p)
        





