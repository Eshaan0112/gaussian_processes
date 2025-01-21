import gpflow
import tensorflow as tf
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rebuild_gpr_model():
    model_saved_at = "gpr_model_checkpoint-1"
    X_placeholder = np.zeros((1, 7))  # 1 sample, 3 features (adjust shape based on your training data)
    Y_placeholder = np.zeros((1, 1))  # 1 sample, 1 output (binary target 0 or 1)
    # Ensure you use the same kernel and likelihood structure as the original model
    rebuilt_model = gpflow.models.GPR(data=(X_placeholder, Y_placeholder), kernel=gpflow.kernels.SquaredExponential())

    # Load the saved parameters into the new model
    checkpoint = tf.train.Checkpoint(model=rebuilt_model)
    checkpoint.restore(model_saved_at).assert_consumed()  # Ensures all variables are restored
    model_ok = verify_model(rebuilt_model)
    if model_ok:
        print("Model restored successfully")

def verify_model(rebuilt_model):
    X = np.array([1.326e+03, 4.800e+02, 3.000e+02, 2.060e+05, 8.000e+04, 3.000e-01, 7.860e+03])
    X = X.reshape(1, -1)  # Shape: (1, 7)
    X = tf.convert_to_tensor(X, dtype=tf.float64)  # Ensure correct data type
    m,_ = rebuilt_model.predict_f(X)
    if not m:
        import pdb;pdb.set_trace()
        return True

