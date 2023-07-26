# Downloading packages and files

import numpy as np
import pandas as pd
import gdown
import regex as re
from scipy.optimize import minimize

# File path for equations
file_path_eq    = r"D:\Research\Climate Action Tool\Iron and steel\equations_steel_V2.txt"

# File path for variables
file_path_var   = r"D:\Research\Climate Action Tool\Iron and steel\variables_steel.xlsx"

# File path for parameters
file_path_param = r"D:\Research\Climate Action Tool\Iron and steel\parameters_steel.xlsx"

""" Processing files """

parameters_df = pd.read_excel(file_path_param)
correct_params_df = parameters_df.loc[parameters_df['type'] == 'correct', ['parameters','values']]
variables_df = pd.read_excel(file_path_var)
correct_vars_df = variables_df.loc[variables_df['type'] == 'correct', ['variables','values']]

''' Obtaining scaling factor '''
max_value_df1 = variables_df['values'].max()
max_value_df2 = correct_params_df['values'].max()
scaling_factor = max(max_value_df1, max_value_df2)

'''concatinating the dataframes containing correct variables and parameters'''

correct_params_df_copy = correct_params_df.copy()
correct_params_df_copy = correct_params_df_copy.rename(columns={'parameters': 'variables'})
correct_df = pd.concat([correct_vars_df, correct_params_df_copy])
correct_df = correct_df.reset_index(drop=True)

decision_var_list = []
static_var_list = []

''' The following lines of code checks if the variable is set to fixed, float or correctable.
    fixed = variable value is fixed
    float = variable is part of the decision variable. Value assigned for variable is discarded
    correct = variable is changed by a small margin by making it a decision variable and applying a tolerance (using bounds).
    variable value is not discarded.'''

float_vars = variables_df.loc[variables_df['type'].isin(['float']), 'variables'].tolist()
decision_var_list.extend(float_vars) # float params is added to decision variables list
correct_vars = variables_df.loc[variables_df['type'].isin(['correct']), 'variables'].tolist()
decision_var_list.extend(correct_vars) # correct params are added to decision variables list
variables_df = variables_df[~variables_df['variables'].isin(float_vars + correct_vars)] # deleting parameters that are float and correct from parameters list. This is done so that they don't get substituted


''' The following lines of code checks if the parameter is set to fixed, float or correctable.
    fixed = parameter value is fixed
    float = parameter is part of the decision variable. Value of parameter is discarded
    correct = parameter is changed by a small margin by making it a decision variable and applying a tolerance (using bounds).
    Parameter value is not discarded.'''

float_params = parameters_df.loc[parameters_df['type'].isin(['float']), 'parameters'].tolist()
decision_var_list.extend(float_params) # float params is added to decision variables list
correct_params = parameters_df.loc[parameters_df['type'].isin(['correct']), 'parameters'].tolist()
decision_var_list.extend(correct_params) # correct params are added to decision variables list
parameters_df = parameters_df[~parameters_df['parameters'].isin(float_params + correct_params)] # deleting parameters that are float and correct from parameters list. This is done so that they don't get substituted

with open(file_path_eq, 'r') as f:
    # Read the equations line by line
    eq_lines = f.readlines()

# Create a list to store the equations
eq_list = []

# Loop through the equation lines
for eq_line in eq_lines:
    # Split the line into the equation name and the equation expression
    eq_name, eq_expr = eq_line.strip().split(':')
    # Convert the tuple of symbols to a single expression
    eq_list.append(eq_expr)

# Creating dictionary for parameters and their values
param_fixed_dict = dict(zip(parameters_df['parameters'],parameters_df['values']))

# Creating dictionary for variables and their values
var_fixed_dict = dict(zip(variables_df['variables'],variables_df['values']))

correct_dict = dict(zip(correct_df['variables'],correct_df['values']))

# Substituting parameters in equation with their values
modified_eq_list_param = []
for eq in eq_list:
  for key in param_fixed_dict:
    pattern = r'\b' + re.escape(key) + r'\b'
    if re.search(pattern, eq):
        value = param_fixed_dict.get(key)
        eq = re.sub(pattern, str(value), eq)
  modified_eq_list_param.append(eq)

# Substituting variables in equation with their values
modified_eq_list_var = [] # list after substituting for variables with corresponding values
for eq in modified_eq_list_param:
  for key in var_fixed_dict:
    pattern = r'\b' + re.escape(key) + r'\b'
    if re.search(pattern, eq):
        value = var_fixed_dict.get(key)
        eq = re.sub(pattern, str(value), eq)
  modified_eq_list_var.append(eq)

''' Updating the equations with the equations for data reconsiliation'''
obj_fn_eq = modified_eq_list_var.copy()
for key, value in correct_dict.items():
  eq = f"{key} - {value}"
  obj_fn_eq.append(eq)

""" Solving the optimization problem"""

# Define the objective function
def objective_function(decision_variables):
    for index, variable in enumerate(decision_var_list):
      globals()[variable] = decision_variables[index]

    equations = []
    for equation in obj_fn_eq:
      equations.append(eval(equation))
    squared_errors = [result**2 for result in equations]
    return sum(squared_errors)

def constraints(decision_variables):

    for index, variable in enumerate(decision_var_list):
      globals()[variable] = decision_variables[index]

    constraints = []
    for equation in modified_eq_list_var:
      constraints.append(eval(equation))
    return constraints

# If type of decision variable is correctable the bounds are adjusted to a tolerance
tol = 0.2
bounds = []
initial_guess = []
for var in decision_var_list:
  if var in correct_params:
    value = (correct_params_df.loc[correct_params_df['parameters'] == var, 'values'].values[0])
    lower_bound = value * (1 - tol)
    upper_bound = value * (1 + tol)
    bounds.append((lower_bound, upper_bound))
    initial_guess.append(value)

  elif var in correct_vars:
    value = (correct_vars_df.loc[correct_vars_df['variables'] == var, 'values'].values[0])
    lower_bound = value * (1 - tol)
    upper_bound = value * (1 + tol)
    bounds.append((lower_bound, upper_bound))
    initial_guess.append(value)

  else:
    bounds.append((0, float('inf')))
    initial_guess.append(float(1))

result = minimize(objective_function, initial_guess, bounds = bounds, constraints={'type': 'eq', 'fun': constraints})

# Extract the optimal solution
optimal_solution = result.x
print("Optimal solution:", optimal_solution)

""" Print solution"""

for variable, value in zip(decision_var_list, optimal_solution):
    value = round(value,2)
    print(f"{variable}: {value}")