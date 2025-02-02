# A full-stack implementation of Gaussian Processes

## Data details
The Material Dataset (located `modeling/data`) includes the following mechanical properties:

- Features:
1. Unique Identification code for the Material (ID)
2. Ultimate Tensile Strength (Su) in MPa
3. Yield Strength (Sy) in MPa
4. Elastic Modulus (E) in MPa
5. Shear Modulus (G) in MPa
6. Poisson's Ratio (mu) in Units of Length
7. Density (Ro) in Kg/m3
- Label
Use - Yes/No

You can read more about the dataset [here](https://www.kaggle.com/datasets/purushottamnawale/materials)

## What the project does
This system provides the user with a placeholder to upload a CSV dataset (in this specific case some material properties - details provided below) and displays the RMSE and Rsq values after performing Gaussian Process Regression on the data uploaded as well as printing the optimized material properties in the CLI that have a constraint of choosing "Yes" in the label.
Essentially, the material properties that are returned are the final optimized inputs which have the highest probability to have "Use" as "Yes"

## How to run the project:
### NOTE: poetry basics and ML basics are assumed to run this project
1. Run backend from project root: `/gaussian_processes`
  ``` poetry run uvicorn api.app:app --reload ```
2. Run frontend from `gaussain_processes/frontend/` using VS code extension live server


