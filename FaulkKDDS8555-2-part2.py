import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor


train_data = pd.read_csv("/Users/kevinfaulk/Documents/DDS-8555/train.csv")
test_data = pd.read_csv("/Users/kevinfaulk/Documents/DDS-8555/test.csv")

# Display the first few rows of the training data
print("Training Data:")
print(train_data.head())
print("\nTest Data:")
print(test_data.head())

# Check for missing values in the training data
print("\nMissing values in training data:")
print(train_data.isnull().sum())
# Check for missing values in the test data
print("\nMissing values in test data:")
print(test_data.isnull().sum())

# Check the data types of the columns in the training data
print("\nData types in training data:")
print(train_data.dtypes)
# Check the data types of the columns in the test data
print("\nData types in test data:")
print(test_data.dtypes)

# One-hot encoding for categorical variables in the training data
train_data = pd.get_dummies(train_data, columns=['Sex'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Sex'], drop_first=True)

# Check the columns after one-hot encoding
print("\nColumns in training data after one-hot encoding:")
print(train_data.columns)
print("\nColumns in test data after one-hot encoding:")
print(test_data.columns)

# Train/Test Split
X = train_data.drop("Rings", axis=1)
y = train_data["Rings"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


scaler= StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test_data)

# Fit the model
lr= LinearRegression()
lr.fit(X_train_scaled, y_train)

# Predict & Evaluate
y_pred_val = lr.predict(X_val_scaled)
print("Linear RMSE:", np.sqrt(mean_squared_error(y_val, y_pred_val)))
print("Linear R^2:", r2_score(y_val, y_pred_val))

# Residuals for linear model
residuals = y_val - y_pred_val

plt.figure()
plt.scatter(y_pred_val, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted - Linearity Check")
plt.show()

# Fit model using statsmodels
X_sm = sm.add_constant(X_train_scaled)
ols_model = sm.OLS(y_train, X_sm).fit()

# Durbin-Watson statistic
from statsmodels.stats.stattools import durbin_watson
dw_stat = durbin_watson(ols_model.resid)
print("Durbin-Watson statistic:", dw_stat)

# Compute VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_scaled, i) for i in range(X_train_scaled.shape[1])]

print(vif_data)

# Histogram
sns.histplot(residuals, kde=True)
plt.title("Histogram of Residuals")
plt.show()

# Q-Q plot
sm.qqplot(residuals, line='45')
plt.title("Q-Q Plot of Residuals")
plt.show()

# Drop variables with high multicollinearity
cols_to_drop = ['Whole weight.1', 'Whole weight.2', 'Length']  # you can experiment with others
X = train_data.drop(columns=cols_to_drop + ['Rings'])
y = train_data['Rings']

# Re-split and scale
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Rebuild model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Evaluate
y_pred_val = lr.predict(X_val_scaled)
print("New Linear RMSE:", np.sqrt(mean_squared_error(y_val, y_pred_val)))
print("New Linear R²:", r2_score(y_val, y_pred_val))

# Check VIF again
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_scaled, i) for i in range(X_train_scaled.shape[1])]

print(vif_data)

# Match test set to training features
test_features = test_data.drop(columns=['Whole weight.1', 'Whole weight.2', 'Length'])

X_test_scaled = scaler.transform(test_features)

test_preds = lr.predict(X_test_scaled)
test_preds_rounded = np.round(test_preds)


from sklearn.preprocessing import PolynomialFeatures

# Generate polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_train = poly.fit_transform(X_train_scaled)
X_poly_val = poly.transform(X_val_scaled)

poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)


y_poly_pred_val = poly_model.predict(X_poly_val)
rmse_poly = np.sqrt(mean_squared_error(y_val, y_poly_pred_val))
r2_poly = r2_score(y_val, y_poly_pred_val)

print("Polynomial RMSE:", rmse_poly)
print("Polynomial R²:", r2_poly)

# Reuse scaled test features from earlier
X_poly_test = poly.transform(X_test_scaled)

# Predict
test_preds_poly = poly_model.predict(X_poly_test)
test_preds_poly_rounded = np.round(test_preds_poly)

print("Linear Model RMSE:", np.sqrt(mean_squared_error(y_val, y_pred_val)))
print("Polynomial Model RMSE:", rmse_poly)

# Save Polynomial Predictions for Kaggle
submission_poly = pd.DataFrame({
    "id": test_data["id"],
    "Rings": test_preds_poly_rounded
})

submission_poly.to_csv("FaulkKDDS8555-2_polynomial.csv", index=False)
print("Polynomial submission file 'submission_polynomial.csv' created!")
