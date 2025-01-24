'''Import statements'''

# ML
import pandas as pd
import numpy as np
import gpflow
import tensorflow as tf

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Metrics
from sklearn.metrics import root_mean_squared_error, r2_score


''' Feature Engineering and Training '''
def read_data_and_preprocess(csv_file):
    """
    Read training, testing data

    Args:
        csv_file (str): CSV file of training data

    Returns:
        X_train(pd.Dataframe): Training features data
        X_test(pd.Dataframe): Testing features data
        Y_train(pd.Dataframe): Training labels data
        Y_test(pd.Dataframe): Testing labels data
    """

    # X_train, X_test, Y_train, Y_test = basic_data() # Basic hardcoded data to test out prediction
    X_train, X_test, Y_train, Y_test = material_data(csv_file) 

    return X_train, X_test, Y_train, Y_test

def material_data(csv_file):
    """
    Preprocess and split data

    Args:
        csv_file (str): Training data file

    Returns:
        X_train(pd.Dataframe): Training features data
        X_test(pd.Dataframe): Testing features data
        Y_train(pd.Dataframe): Training labels data
        Y_test(pd.Dataframe): Testing labels data
    """

    df = pd.read_csv(csv_file)
    df = feature_encoding(df) # encode features
    df = target_encoding(df) # encode labels
    
    data = df.to_numpy()

    train = data[:1500,:]
    X_train = train[:,:7] # features
    Y_train = train[:,-1] # label

    test = data[1500:,:]
    X_test = test[:,:7] # features
    Y_test = test[:,-1] # label

    return X_train, X_test, Y_train, Y_test

def basic_data():
    """
    Hardcoded basic data

    Returns:
        X_train(pd.Dataframe): Training features data
        X_test(pd.Dataframe): Testing features data
        Y_train(pd.Dataframe): Training labels data
        Y_test(pd.Dataframe): Testing labels data
    """
    X_train = np.array(
    [
        [0.865], [0.666], [0.804], [0.771], [0.147], [0.866], [0.007], [0.026],
        [0.171], [0.889], [0.243]
    ]
    )
    Y_train = np.array(
        [
            [1.57], [3.48], [3.12], [3.91], [3.07], [1.35], [3.80], [3.82], [3.49],
            [1.30], [4.00]
        ]
    )   
    X_test = np.array([[0.028]])
    Y_test = np.array([[3.82]])

    return X_train, X_test, Y_train, Y_test

def feature_encoding(df): 
    """
    Encodes features

    Args:
        df (pd.Dataframe): Feature dataframe

    Returns:
        df (pd.Dataframe): Feature dataframe
    """
    encoder = LabelEncoder()
    df['Material'] = encoder.fit_transform(df['Material'])
    return df
    

def target_encoding(df):
    """
    Converts Boolean labels to 0s and 1s

    Args:
        df (pd.Dataframe): Labels series

    Returns:
        df (pd.Dataframe): Labels series
    """

    df['Use'] = df['Use'].astype(int)
    return df
    

def gp_training(X,Y):
    """
    GPR training 

    Args:
        X (pd.Dataframe): Features dataframe
        Y (pd.Series): Labels dataframe

    Returns:
        model (gpflow.models.gpr.GPR): Trained model
    """
    Y = Y.reshape(-1,1)
    k = gpflow.kernels.SquaredExponential(lengthscales=1) # need to set lengthscales to avoid ill conditioned matrix
    model = gpflow.models.GPR(data=(X,Y), kernel=k, mean_function=None)
    
    # Optimize hyperparameters of the kernel
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables) 

    return model

def gp_predict(model, X_test):
    """
    Prediction function

    Args:
        model (gpflow.models.gpr.GPR): Trained model
        X_test (pd.Dataframe): Labels dataframe

    Returns:
        mean(tensor): Mean of GPR (predictions)
    """
    print(X_test)
    mean, _ = model.predict_f(X_test) # predict_f expects a numpy array 

    return mean

def gp_judge_model(Y_test, Y_preds):
    """
    RMSE and Rsquared value of GPR model

    Args:
        Y_test (ndarray): True predictions
        Y_preds (tensor): GPR predictions

    Returns:
        [type]: [description]
    """
    rmse = root_mean_squared_error(Y_test, Y_preds.numpy()) 
    r2 = r2_score(Y_test, Y_preds.numpy())

    print(f'RMSE: {rmse}')   
    print(f'Rsq: {r2}') 

    return rmse, r2

def gp_save_model(model):
    """
    Save GPR model with tensorflow

    Args:
        model (gpflow.models.gpr.GPR): GPR model to be saved
    """
    checkpoint = tf.train.Checkpoint(model=model)
    model_saved_at = "../optimization/gpr_model_checkpoint" # to use during optimization
    checkpoint.save(model_saved_at)
    print(f"Model saved at {model_saved_at}")


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
def main(csv_file):
    """
    Driver function

    Args:
        csv_file (str): CSV file training data

    Returns:
        rmse, r2 (float, float): metrics
    """
    X_train, X_test, Y_train, Y_test = read_data_and_preprocess(csv_file)    
    trained_model = gp_training(X_train, Y_train)
    print('Training complete.')
    gp_save_model(trained_model)
    Y_predictions = gp_predict(trained_model, X_test)
    print(Y_predictions)
    rmse,r2 = gp_judge_model(Y_test, Y_predictions)

    return rmse,r2
    
if __name__=="__main__":
    main('data/material.csv')