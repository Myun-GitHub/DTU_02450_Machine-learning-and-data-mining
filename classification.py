import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot,xlabel, ylabel, show, subplot, semilogx, title, grid, legend, suptitle, tight_layout, boxplot
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import mcnemar


#Get Data
df = pd.read_excel("Real estate valuation data set.xlsx")
df = df.drop(['No'], axis=1)
columns = df.columns
attr = [x[:2] for x in columns]
attr = attr[0:6]
raw_data = np.array(df)
X = raw_data[:, 0:6]  
N,M = np.shape(X)
labels = ['Low Price', 'Medium Price', 'High Price']
#Classify Y
conditions = [
    df['Y house price of unit area'] <= 27.7,
    (df['Y house price of unit area'] > 27.7) & (df['Y house price of unit area'] <= 46.6),
    df['Y house price of unit area'] > 46.6]
label = np.select(conditions, labels)
classdict = dict(zip(labels, [0,1,2]))
y = np.asarray([classdict[value] for value in label])
# Standardizes data matrix so each column has mean 0 and std 1
X = (X - np.ones((N,1))*X.mean(0))/X.std(0)


# Set parameters
K1 = 5
K2 = 5
lambda_interval = np.logspace(-3, 2, 20)
L = 20
L_list = np.arange(1,L+1,1)

CV1 = model_selection.KFold(n_splits = K1, shuffle = True, random_state = 1)
CV2 = model_selection.KFold(n_splits = K2, shuffle = True, random_state = 1)

error1_logistic = np.zeros((K1))
error2_logistic =  np.zeros((K2,len(lambda_interval)))
min_error_logistic = np.zeros(K1)
opt_lambda = np.zeros(K1)


error1_KNN = np.zeros((K1))
error2_KNN = np.zeros((K2,L))
x_KNN = [0] * K1

error_baseline = np.zeros((K1))

yhat = []
y_true = []
n = 0

for train_index1, test_index1 in CV1.split(X):
    X_train1 = X[train_index1,:]
    y_train1 = y[train_index1]
    X_test1 = X[test_index1,:]
    y_test1 = y[test_index1]
    
    i = 0
    for train_index2, test_index2 in CV2.split(X_train1):
        print('Crossvalidation fold: {0}/{1}'.format(n+1,i+1))
        
        X_train2 = X[train_index2,:]
        y_train2 = y[train_index2]
        X_test2 = X[train_index2,:]
        y_test2 = y[train_index2]
        
        #Logistical Regression
        for k in range(0,len(lambda_interval)):
            mdl = LogisticRegression(penalty='l2',multi_class='ovr', solver='liblinear', C=1/lambda_interval[k] )
            mdl.fit(X_train2, y_train2)
            y_est_log2 = mdl.predict(X_test2).T
            
            error2_logistic[i,k] = np.sum(y_est_log2 !=y_test2)/len(y_test2)
    
        #KNN
        for k in range(1,L+1):
            knclassifier = KNeighborsClassifier(n_neighbors=k);
            knclassifier.fit(X_train2, y_train2);
            
            y_est_KNN2 = knclassifier.predict(X_test2);
            error2_KNN[i,k-1] = np.sum(y_est_KNN2 != y_test2) / len(y_test2)
            
        i+=1
    
    #Logistical Regression
    min_error_logistic[n] = np.min(error2_logistic.mean(0))
    opt_lambda_idx = np.argmin(error2_logistic.mean(0))
    opt_lambda[n] = lambda_interval[opt_lambda_idx]
    
    mdl = LogisticRegression(penalty='l2',multi_class='ovr', solver='liblinear', C=1/lambda_interval[n] )
    mdl.fit(X_train1, y_train1)
    y_est_log1 = mdl.predict(X_test1).T
            
    error1_logistic[n] = np.sum(y_est_log1 !=y_test1)/len(y_test1)
    
    #KNN
    min_idx = np.argmin(error2_KNN.mean(0))
    x_KNN[n] = L_list[min_idx]
    
    knclassifier = KNeighborsClassifier(n_neighbors=x_KNN[n]);
    knclassifier.fit(X_train1, y_train1);
    y_est_KNN1 = knclassifier.predict(X_test1);
    error1_KNN[n] = np.sum(y_est_KNN1 != y_test1) / len(y_test1)
    
    #Baseline
    baseline = np.argmax(np.bincount(y_train1))
    y_est_base = np.ones((y_test1.shape[0]), dtype = int)*baseline
    error_baseline[n] = np.sum(y_est_base != y_test1) / len(y_test1)
    
    dy = []
    dy.append(y_est_base)
    dy.append(y_est_KNN1)
    dy.append(y_est_log1)
    dy = np.stack(dy, axis=1)
    yhat.append(dy)
    
    y_true.append(y_test1)
    n+=1
    

y_true = np.concatenate(y_true)
yhat = np.concatenate(yhat)


print('Errors KNN:\tErrors baseline\tErrors LOGREG')
for m in range(K1):   
    print(' ',np.round(error1_KNN[m],2),'\t\t',np.round(error_baseline[m],2),'\t\t',np.round(error1_logistic[m],2))



fig = plt.figure()
plt.plot(L_list,error2_KNN.mean(0)*100,'-o')
plt.xlabel('Number of neighbors')
plt.ylabel('Classification error rate (%)')
plt.savefig('KNN.png',dpi=300, bbox_inches='tight')

fig = plt.figure()
plt.semilogx(lambda_interval, error2_logistic.mean(0)*100,'-or')
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.ylabel('Classification error rate (%)')
plt.savefig('Logtistic Regression.png',dpi=300, bbox_inches='tight')


fig= plt.figure()
boxes = [error1_logistic, error1_KNN,error_baseline]
boxes_df = pd.DataFrame(boxes).T
x = [1,2,3]
labels = ['Logistic Regression','KNN', 'Baseline']
plt.boxplot(boxes_df)
ylabel('Generalization Error')
plt.xticks(x,labels)
plt.savefig('boxplot_classification.png',dpi=300, bbox_inches='tight') 


alpha = 0.05

print('A : Baseline\nB : KNN')
[thetahat, CI, p] = mcnemar(y_true, yhat[:,0], yhat[:,1], alpha=alpha)
print('theta: ',np.round(thetahat,2),' CI: ',np.round(CI,2),' p: ',np.round(p,3))
print('\n')
print('A : Baseline\nB : Logistical Regression')
[thetahat, CI, p] = mcnemar(y_true, yhat[:,0], yhat[:,2], alpha=alpha)
print('theta: ',np.round(thetahat,2),' CI: ',np.round(CI,2),' p: ',np.round(p,3))
print('\n')
print('A : KNN\nB : Logistical Regression')
[thetahat, CI, p] = mcnemar(y_true, yhat[:,1], yhat[:,2], alpha=alpha)
print('theta: ',np.round(thetahat,2),' CI: ',np.round(CI,2),' p: ',np.round(p,3))



