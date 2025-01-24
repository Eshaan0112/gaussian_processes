'''Import statements'''

# ML
import gpflow
import tensorflow as tf
import numpy as np

# General
import sys
import os

# Add root directory to Python Path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def rebuild_gpr_model(model_saved_at):
    """
    Args:
        model_saved_at (str): Directory where the GPR model is saved

    Returns:
        rebuilt_model (gpflow.models.gpr.GPR): Rebuilt model
    """

    # Load saved model with dummy data
    X_placeholder = np.zeros((1, 7))  # number of features in our training data were 7
    Y_placeholder = np.zeros((1, 1)) 
    rebuilt_model = gpflow.models.GPR(data=(X_placeholder, Y_placeholder), kernel=gpflow.kernels.SquaredExponential())

    # Load the saved parameters into the new model
    checkpoint = tf.train.Checkpoint(model=rebuilt_model)
    checkpoint.restore(model_saved_at).assert_consumed()  # ensures all variables are restored
    model_ok = verify_model(rebuilt_model) # just a simple check to see if model was loaded correctly
    if model_ok:
        print("Model restored successfully.")
        return rebuilt_model
    else:
        print("Something wrong with model. Check rebuilding procedure or forward model")
    

def verify_model(rebuilt_model):
    """
    Args:
        rebuilt_model (gpflow.models.gpr.GPR): Rebuilt model

    Returns:
        Boolean: True if model verified else False
    """

    # A random entry expecting prediction of False i.e 0.0
    X = np.array([1.326e+03, 4.800e+02, 3.000e+02, 2.060e+05, 8.000e+04, 3.000e-01, 7.860e+03])
    X = X.reshape(1, -1)  # shape: (1, 7)
    X = tf.convert_to_tensor(X, dtype=tf.float64)  # ensure correct data type

    m,_ = rebuilt_model.predict_f(X)

    if not m:
        return True
    else:
        return False

if __name__=="__main__":
    model_saved_at = "gpr_model_checkpoint-1"
    rebuilt_model = rebuild_gpr_model(model_saved_at)

    #todo
    '''
    pyomo optimization
    '''

