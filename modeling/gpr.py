# Import statements
import pandas as pd
from sklearn.model_selection import train_test_split


def read_data():
    material_df = pd.read_csv("data/material.csv")
    print(material_df.shape)
    X = material_df.iloc[:,:2]# Two features
    Y = material_df.iloc[:,7] # Yes or No
    fulldata = pd.concat([X, Y], axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, Y_train, Y_test

def gp_training() 

def path_exists(path):
    import os
    if os.path.exists(path):
        print("Exists")
    else:
        print("not")

read_data()
