from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, tight_layout, 
                           title, subplot, show, grid, plot)
import numpy as np
import pandas as pd
from sklearn import model_selection
from toolbox_02450 import rlr_validate

#read dataset
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

#Set attribute shape
attributeNames = [u'Offset']+attributeNames
M = M+1


## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True,random_state = 1)


# Values of lambda
#lambdas = np.power(10.,range(-2,8))
lambdas = np.logspace(-2, 8, 50)
#lambdas = np.arange(0.1,100,0.5)

# Initialize data
w = np.empty((M,K,len(lambdas)))
train_error = np.empty((K,len(lambdas)))
test_error = np.empty((K,len(lambdas)))
y = y.squeeze()


k = 0
for train_index, test_index in CV.split(X,y):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    
    # precompute terms
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    
    for l in range(0,len(lambdas)):
        # Compute parameters for current value of lambda and current CV fold
        lambdaI = lambdas[l] * np.eye(M)
        lambdaI[0,0] = 0 # remove bias regularization
        w[:,k,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
        # Evaluate training and test performance
        train_error[k,l] = np.power(y_train-X_train @ w[:,k,l].T,2).mean(axis=0)
        test_error[k,l] = np.power(y_test-X_test @ w[:,k,l].T,2).mean(axis=0)

    k=k+1
    
minArg = np.argmin(np.mean(test_error,axis=0))
k
opt_val_err = np.min(np.mean(test_error,axis=0))
opt_lambda = lambdas[minArg]
train_err_vs_lambda = np.mean(train_error,axis=0)
test_err_vs_lambda = np.mean(test_error,axis=0)
mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))


# PLOTS 
dpi = 75 # Sets dpi for plots
save_plots = False

# The difference from last plot here is that opt_lamda is not written as a power of 10
f = figure()
title('Optimal lambda: {0}'.format(np.round(opt_lambda,3)))
semilogx(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
xlabel('Regularization factor')
ylabel('Estimated generalization error')
legend(['Train error','Validation error'])
grid()
tight_layout()
f.savefig('./figures/reg_part_a_error_vs_lambdas.png', bbox_inches='tight') if save_plots else 0

f2 = figure()
semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
xlabel('Regularization factor')
ylabel('Mean Coefficient Values')
grid()
legend(attributeNames[1:], loc='best')


print('Weights for best regularization parameter:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(mean_w_vs_lambda[m,minArg],3)))

