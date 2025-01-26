'''Import statements'''
from pyomo.environ import ConcreteModel, Var, Objective, SolverFactory, maximize, Param
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

def acquisition_function_callback(x_values, gpr_model, f_target=1.0, xi=0.01):
    # Convert the Pyomo variables into a NumPy array for GPR
    x_array = np.array(x_values).reshape(1, -1)

    # Predict mean (mu) and variance (var) using the GPR model
    mu, var = gpr_model.predict_f(x_array)
    sigma = np.sqrt(np.diagonal(var))

    # Compute EI
    best_observed = f_target
    Z = (mu - best_observed) / (sigma + 1e-9)  # Avoid division by zero
    ei_value = (mu - best_observed) * norm.cdf(Z) + sigma * norm.pdf(Z)
    return ei_value[0][0]  # Return the scalar EI value

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def run_optimizer(gpr_model, f_target=1.0):
    import pdb;pdb.set_trace()
    import pyomo.environ as pyo
    import numpy as np

    # Create a Pyomo model
    model = pyo.ConcreteModel()
    num_properties = 7

    # Define decision variables
    model.x = pyo.Var(range(num_properties), domain=pyo.Reals, initialize=0.0)  # Initialize to 0.0


    # Initialize decision variables with some starting values
    initial_values = {0:1.326e+03, 1:4.800e+02, 2:3.000e+02, 3:2.060e+05, 4:8.000e+04, 5:3.000e-01, 6:7.860e+03}
    # for i in range(num_properties):
    #     model.x[i].set_value(initial_values[i])

    # External function to compute the acquisition function (Expected Improvement)
    # Define the objective function
    def objective_function(model):
        # Extract decision variables as concrete values
        x_values = [model.x[i] for i in range(len(model.x))]

        # Use a callback to compute the acquisition function (EI)
        ei_value = acquisition_function_callback(
            x_values=[pyo.value(var) for var in x_values],  # Extract concrete values
            gpr_model=gpr_model,
            f_target=f_target
        )
        return ei_value



    import pdb;pdb.set_trace()
    # Set the objective in the Pyomo model
    model.obj = pyo.Objective(rule=objective_function, sense=pyo.minimize)

    import pdb;pdb.set_trace()

    # Solve the model using the Ipopt solver
    solver = pyo.SolverFactory('ipopt')
    solver.options['tol'] = 1e-6  # Set tolerance for solver precision
    solver.options['max_iter'] = 1000  # Max number of iterations

    results = solver.solve(model, tee=True)

    # Get the optimized material properties
    optimized_material_properties = [model.x[i].value for i in range(num_properties)]

    return optimized_material_properties

if __name__=="__main__":
    gpr_model = main_opt()
    opt_props = run_optimizer(gpr_model)
    print(opt_props)
