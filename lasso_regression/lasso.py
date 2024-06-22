import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load dataset
from sklearn.datasets import fetch_california_housing
california_housing = fetch_california_housing(as_frame=True)

# split up dataset into train set and test set
X_train, X_test, y_train, y_test = train_test_split(california_housing.data, california_housing.target, test_size=0.2, random_state=42)

# processing data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# load lasso model
model = Lasso(random_state=42)

# set up parameters
parameters = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
    'max_iter': [1000, 2000, 5000],
  }

# model selection
GS = GridSearchCV(model, param_grid=parameters, cv=5, scoring='r2')
GS.fit(X_train, y_train)
train_r2 = GS.best_score_
best_params = GS.best_params_

model = GS.best_estimator_
Y_pred = model.predict(X_test)
test_r2 = r2_score(y_test, Y_pred)

# create a dict to save the result
result = {}
result['best_parameters'] = best_params
result['train_r2'] = train_r2
result['test_r2'] = train_r2
print(result)

coefficients = model.coef_

# Count the number of variables with a coefficient of 0 (unimportant variable)
num_zero_coef = sum(coefficients == 0)
print(f'Number of features with zero coefficient: {num_zero_coef}')

# Coefficients and variable names
lasso_coef = [float(x) for x in np.abs(model.coef_)]
feature_names = california_housing.data.columns.tolist()
coefficients_df = pd.DataFrame()
coefficients_df['features'] = feature_names
coefficients_df['coefficients'] = lasso_coef

# Create a bar plot
plt.figure(figsize=(12, 6))
sns.barplot(x="features", y="coefficients", data=coefficients_df)
plt.title("Coefficients of Lasso models")
plt.xlabel("Features")
plt.ylabel("Coefficients")
plt.tight_layout()

# Create the folder if it doesn't exist
folder_path = "lasso_regression\image"
os.makedirs(folder_path, exist_ok=True)

# Save the plot to the folder
plt.savefig(os.path.join(folder_path, "bar_plot.png"), dpi=300)