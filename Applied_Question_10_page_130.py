from ISLP import load_data
from sklearn.datasets import fetch_openml

import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Load the Carseats dataset from ISLP
carseats = load_data('Carseats')

print(carseats.head())

print(carseats.dtypes)

# (a) Fit a multiple regression model with Sales as the response and Price, Urban, and US as predictors.
model_full=smf.ols('Sales ~ Price + Urban + US', data=carseats).fit()
print(model_full.summary())

# (e)  Fit a smaller model with only significant predictors.

model_reduced=smf.ols('Sales ~ Price + US', data=carseats).fit()
print(model_reduced.summary())

infl = model_reduced.get_influence()
leverage=infl.hat_matrix_diag

fig, ax= plt.subplots(figsize=(8, 6))
ax.scatter(np.arange(len(leverage)), leverage, alpha=0.7)
ax.axhline(y=2*(model_reduced.df_model+1)/model_reduced.nobs,color='red', linestyle='--', label='High Leverage Threshold')
ax.set_xlabel('Observation Index')
ax.set_ylabel('Leverage')
ax.set_title('Leverage Values for Each Observation')
ax.legend()

max_lev_idx= np.argmax(leverage)
print(f'Max Leverage Index: {max_lev_idx}, Leverage Value: {leverage[max_lev_idx]:.4f}')

from statsmodels.api import OLS
residuals=infl.resid_studentized_internal

# Identify large residuals
outliers = np.where(np.abs(residuals) > 2)[0]
print("Potential outliers at indices:", outliers)

