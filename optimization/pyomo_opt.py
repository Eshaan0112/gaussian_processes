'''Import statements'''
from pyomo.environ import ConcreteModel, Var, Objective, SolverFactory, maximize
import pyomo.environ as pyo
import numpy as np
import tensorflow as tf
import math
from scipy.stats import norm
from opt import main_opt


solver = 'ipopt'
solver_io = 'nl'
stream_solver = False  # True prints solver output to screen
keepfiles = False  # True prints intermediate file names (.nl,.sol,...)

# Initialize solver IPOPT
opt = SolverFactory(solver, solver_io=solver_io)

if opt is None:
    print("")
    print(
        "ERROR: Unable to create solver plugin for %s "
        "using the %s interface" % (solver, solver_io)
    )

def acquisition_function_binary(candidates, gpc_model, f_target = 1.0):
    """
    Calculating Expected Improvement as an acquisition function for Bayesian Optimization

    Args:
        candidates ([type]): [description]
        gpc_model ([type]): [description]
        f_target (float, optional): [description]. Defaults to 1.0.

    Returns:
        [type]: [description]
    """
    x_tensor = tf.convert_to_tensor(candidates, dtype=tf.float64)
    mean, var = gpc_model.predict_f(x_tensor)
    sigma = math.sqrt(var)
    mean = sigmoid(mean)    
    improvement = np.maximum(mean - f_target, 0)
    
    # Compute Expected Improvement
    # Avoid division by zero if sigma is zero
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (mean - f_target) / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Return the negative EI (since we're minimizing)
    return -ei

# Pyomo Optimization Model
def run_optimizer(gpr_model, f_target=1.0):
    # Create a Pyomo model
    model = pyo.ConcreteModel()
    num_properties=7
    
    # Define decision variable(s) - let's assume we are optimizing for a single material property
    # You can expand this to multiple properties if needed (e.g., x1, x2, ..., xn)
    model.x = pyo.Var(range(num_properties), domain=pyo.Reals)
    initial_values = [1.326e+03, 4.800e+02, 3.000e+02, 2.060e+05, 8.000e+04, 3.000e-01, 7.860e+03]
    for i in range(num_properties):
        model.x[i] = initial_values[i]  # Initialize with your data or an educated guess
    # Define the objective function: Expected Improvement (EI)
    def objective_function(model):
        
        candidates = np.array([pyo.value(model.x[i]) for i in range(num_properties)]).reshape(1, -1)
        ei_value = acquisition_function_binary(candidates, gpr_model, f_target)
        return ei_value
    
    model.obj = pyo.Objective(rule=objective_function, sense=pyo.minimize)
    
    
    # Solve the model using the Ipopt solver
    solver = pyo.SolverFactory('ipopt')
    solver.options['tol'] = 1e-6  # Set tolerance for solver precision
    solver.options['max_iter'] = 1000  # Max number of iterations
    
    _ = solver.solve(model, tee=True)
    
    # Get the optimized material properties
    optimized_material_properties = [model.x[i].value for i in range(len(num_properties))]
    
    return optimized_material_properties


''' Helper functions'''
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

if __name__=="__main__":
    gpr_model = main_opt()
    properties = run_optimizer(gpr_model)
    print("Optimized Material Properties:", properties)
