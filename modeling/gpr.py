'''Import statements'''
# Data
from sklearn.datasets import fetch_california_housing

# ML
import pandas as pd
import numpy as np
import gpflow

# Preprocessing
from sklearn.model_selection import train_test_split

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Metrics
from sklearn.metrics import root_mean_squared_error, r2_score


''' Feature Engineering and Training '''
def read_data_and_preprocess():

    data = fetch_california_housing()

    # Convert data from sklearn bunch to dataframe to sample at random
    X = data.data
    X = pd.DataFrame(data.data, columns=data.feature_names)
    Y = data.target
    Y = pd.Series(data.target, name='target')
    fulldata = pd.concat([X,Y], axis=1)
    fulldata = fulldata.sample(n=1000, replace=True, random_state=42)

    # Re-extract features and labels
    X = fulldata.drop(columns=['target'])
    Y = fulldata['target']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) # 80-20 train-test split
    
    return X_train, X_test, Y_train, Y_test

def feature_encoding(X): 
    # In the future, this will be needed when GPR is tried on a real-life dataset
    # return X
    pass

def label_encoding(Y):
    # In the future, this will be needed when GPR is tried on a real-life dataset
    # return Y
    pass

def gp_training(X,Y):

    kernel = gpflow.kernels.SquaredExponential()
    model = gpflow.models.GPR((X,Y), kernel=kernel)
    
    # Optimize hyperparameters of the kernel
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables) 

    return model

def gp_predict(model, X_test):

    mean, covar = model.predict_f(X_test.values, full_cov=False) # predict_f expects a numpy array 

    # GPR is configured for multioutput regression in GPFlow. Not sure how to change that configuration. The mean dims are (200,800) meaning there are 800 predictions for the same input
    # Selecting only one out of them as of now. The root cause of this is yet to be debugged
    mean = mean.numpy()[:,0] 

    return mean

def gp_judge_model(Y_test, Y_preds):

    rmse = root_mean_squared_error(Y_test.to_numpy(), Y_preds) 
    rsq =  r2_score(Y_test.to_numpy(), Y_preds)

    print(f'RMSE: {rmse}')    
    print(f'Rsq: {rsq}')

''' Helper functions'''
def check_correlation(X):

    print(f'Check correlation between features')
    corr_matrix = pd.DataFrame(X).corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.show()

def path_exists(path):

    import os
    if os.path.exists(path):
        print("Path exists")
    else:
        print("Path does not exist")

''' Main function '''
def main():
    X_train, X_test, Y_train, Y_test = read_data_and_preprocess()
    trained_model = gp_training(X_train, Y_train)
    print('Training complete.')
    Y_predictions = gp_predict(trained_model, X_test)
    gp_judge_model(Y_test, Y_predictions)
    
if __name__=="__main__":
    main()