import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error





############################################################################################################

# Split the data into train, validation, and test sets
def split_csv(filename, train_size=0.7, validation_size=0.2):
  df = pd.read_csv(filename)
  train, test = train_test_split(df, test_size=1 - train_size)
  train, validation = train_test_split(train, test_size=validation_size / (train_size + validation_size))
  return train, validation, test
filename = 'data_from_kaggle.csv'
train_df, validation_df, test_df = split_csv(filename)
train_df.to_csv('train_data.csv', index=False)
validation_df.to_csv('validation_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)

############################################################################################################
############################################################################################################

#train
train_data_unedited = pd.read_csv("train_data.csv")
#validation
validation_data_unedited = pd.read_csv("validation_data.csv")
#test
test_data_unedited = pd.read_csv("test_data.csv")
train_data_features = train_data_unedited.drop(columns=['Unnamed: 0','target','song_title','artist'])
validation_data_features = validation_data_unedited.drop(columns=['Unnamed: 0','target','song_title','artist'])
test_data_features = test_data_unedited.drop(columns=['Unnamed: 0','target','song_title','artist'])
train_data_conclusion = train_data_unedited['target']
validation_data_conclusion = validation_data_unedited['target']
test_data_conclusion = test_data_unedited['target']


############################################################################################################
############################################################################################################

# Plot the distribution of the target variable in the training set 

#plot for all attributes in the data vs target and save the plot as a png file in scatterplot folder
for column in train_data_features.columns:
    sns.scatterplot(data=train_data_unedited, x=column, y='target', hue='target', palette=['blue', 'orange'])
    plt.xlabel(column)
    plt.ylabel('Target')
    plt.title(f"{column} vs. Target")
    plt.savefig(os.path.join('scatterplot', f"{column}.png"))
    plt.clf()

# Plot the correlation matrix for the training set

temporary = test_data_unedited.drop(columns=['Unnamed: 0','song_title','artist'])
correlation_matrix = temporary.corr()

# Create a heatmap to visualize the correlations
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', linewidths=0.5)

# Set plot title and labels
plt.title('Correlation Matrix')
plt.xlabel('Features')
plt.ylabel('Features')

# Show the plot
plt.show()

############################################################################################################
############################################################################################################

best_features = ['instrumentalness', 'duration_ms', 'loudness', 'speechiness', 'danceability', 'energy', 'acousticness', 'valence', 'tempo', 'liveness', 'key', 'mode', 'time_signature']
X_train_best = train_data_unedited[best_features]
X_val_best = validation_data_unedited[best_features]
X_test_best = test_data_unedited[best_features]
scaler = StandardScaler()
X_train_best = scaler.fit_transform(X_train_best)
X_val_best = scaler.transform(X_val_best)
X_test_best = scaler.transform(X_test_best)
# Random Forest
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train_best, train_data_conclusion)

# Make predictions
y_pred_rf_val = rf_classifier.predict(X_val_best)
y_pred_rf = rf_classifier.predict(X_test_best)

# Print classification metrics
print("Random Forest Best Model Accuracy (Validation):", accuracy_score(validation_data_conclusion, y_pred_rf_val))
print("Random Forest Best Model Accuracy (Test):", accuracy_score(test_data_conclusion, y_pred_rf))
# with 10
# Random Forest Best Model Accuracy (Validation): 0.7738853503184714
# Random Forest Best Model Accuracy (Test): 0.764026402640264
# with all 13
# Random Forest Best Model Accuracy (Validation): 0.767515923566879
# Random Forest Best Model Accuracy (Test): 0.7821782178217822




# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Initialize the random forest model
rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=5)
grid_search.fit(X_train_best, train_data_conclusion)
print("Best parameters found: ", grid_search.best_params_)
best_rf = grid_search.best_estimator_
y_pred_rf_val = best_rf.predict(X_val_best)
y_pred_rf = best_rf.predict(X_test_best)
# print("Random Forest Best Model Accuracy (Validation):", accuracy_score(validation_data_conclusion, y_pred_rf_val))
# print("Random Forest Best Model Accuracy (Test):", accuracy_score(test_data_conclusion, y_pred_rf))
# Best parameters found:  {'max_depth': 20, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 200}
# Random Forest Best Model Accuracy (Validation): 0.7707006369426752
# Random Forest Best Model Accuracy (Test): 0.7739273927392739


gb = GradientBoostingClassifier()
gb.fit(X_train_best, train_data_conclusion)
y_pred_gb_val = gb.predict(X_val_best)
y_pred_gb = gb.predict(X_test_best)
print("Gradient Boosting Accuracy (Validation):", accuracy_score(validation_data_conclusion, y_pred_gb_val))
print("Gradient Boosting Accuracy (Test):", accuracy_score(test_data_conclusion, y_pred_gb))
# Gradient Boosting Accuracy (Validation): 0.7770700636942676
# Gradient Boosting Accuracy (Test): 0.7656765676567657
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 5, 7, 9, 11],
    'min_samples_split': [2, 5, 10, 15, 100],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'max_features': ['sqrt', 'log2', None, 0.5, 0.8]
}

grid_search = GridSearchCV(estimator=gb, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=5)
grid_search.fit(X_train_best, train_data_conclusion)
print("Best parameters found: ", grid_search.best_params_)
best_gb = grid_search.best_estimator_
y_pred_gb_val = best_gb.predict(X_val_best)
y_pred_gb = best_gb.predict(X_test_best)
print("Gradient Boosting Best Model Accuracy (Validation):", accuracy_score(validation_data_conclusion, y_pred_gb_val))
print("Gradient Boosting Best Model Accuracy (Test):", accuracy_score(test_data_conclusion, y_pred_gb))
# Best parameters found:  {'max_depth': 11, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 500}
# Gradient Boosting Best Model Accuracy (Validation): 0.7770700636942676
# Gradient Boosting Best Model Accuracy (Test): 0.7656765676567657

############################################################################################################
############################################################################################################

lr = LinearRegression()
lr.fit(train_data_features, train_data_conclusion)
# Predict on the test set
y_pred_lr_val = lr.predict(validation_data_features)
y_pred_lr = lr.predict(test_data_features)
print("Linear Regression R² validation :", r2_score(validation_data_conclusion, y_pred_lr_val))
print("Linear Regression MSE validation :", mean_squared_error(validation_data_conclusion, y_pred_lr_val))
print("Linear Regression R²:", r2_score(test_data_conclusion, y_pred_lr))
print("Linear Regression MSE:", mean_squared_error(test_data_conclusion, y_pred_lr))
# Linear Regression R² validation : 0.09894698212749553
# Linear Regression MSE validation : 0.2251170330363646
# Linear Regression R²: 0.09757846008611182
# Linear Regression MSE: 0.22551692095142434



ridge = Ridge(alpha=1.0)
ridge.fit(train_data_features, train_data_conclusion)
y_pred_ridge_val = ridge.predict(validation_data_features)
y_pred_ridge = ridge.predict(test_data_features)
print("Ridge Regression R² validation :", r2_score(validation_data_conclusion, y_pred_ridge_val))
print("Ridge Regression MSE validation :", mean_squared_error(validation_data_conclusion, y_pred_ridge_val))
print("Ridge Regression R²:", r2_score(test_data_conclusion, y_pred_ridge))
print("Ridge Regression MSE:", mean_squared_error(test_data_conclusion, y_pred_ridge))
# Ridge Regression R² validation : 0.09896145697562575
# Ridge Regression MSE validation : 0.22511341667328708
# Ridge Regression R²: 0.09763806772811867
# Ridge Regression MSE: 0.2255020248842362



lasso = Lasso(alpha=0.01)
lasso.fit(train_data_features, train_data_conclusion)
y_pred_lasso = lasso.predict(test_data_features)
y_pred_lasso_val = lasso.predict(validation_data_features)
print("Lasso Regression R² validation :", r2_score(validation_data_conclusion, y_pred_lasso_val))
print("Lasso Regression MSE validation :", mean_squared_error(validation_data_conclusion, y_pred_lasso_val))
print("Lasso Regression R²:", r2_score(test_data_conclusion, y_pred_lasso))
print("Lasso Regression MSE:", mean_squared_error(test_data_conclusion, y_pred_lasso))
X = train_data_unedited.drop(columns=['Unnamed: 0','target','song_title','artist'])
lasso_coef = pd.Series(lasso.coef_, index=X.columns)
print(lasso_coef[lasso_coef != 0])  
# Lasso Regression R² validation : 0.10118583798538072
# Lasso Regression MSE validation : 0.2245576823898142
# Lasso Regression R²: 0.10239399072535393
# Lasso Regression MSE: 0.22431351035617997
# acousticness       -0.086154
# danceability        0.054891
# duration_ms         0.047758
# instrumentalness    0.070211
# liveness            0.018209
# loudness           -0.050131
# mode               -0.008989
# speechiness         0.065702
# tempo               0.019653
# valence             0.052024


rf = RandomForestRegressor(n_estimators=100)
rf.fit(train_data_features, train_data_conclusion)
y_pred_rf = rf.predict(test_data_features)
y_pred_rf_val = rf.predict(validation_data_features)
print("Random Forest R² validation :", r2_score(validation_data_conclusion, y_pred_rf_val))
print("Random Forest MSE validation :", mean_squared_error(validation_data_conclusion, y_pred_rf_val))
print("Random Forest R²:", r2_score(test_data_conclusion, y_pred_rf))
print("Random Forest MSE:", mean_squared_error(test_data_conclusion, y_pred_rf))
importances = rf.feature_importances_
X = train_data_unedited.drop(columns=['Unnamed: 0','target','song_title','artist'])
feature_importance = pd.Series(importances, index=X.columns)
feature_importance.sort_values(ascending=False, inplace=True)
print(feature_importance)
# Random Forest R² validation : 0.3648993119370132
# Random Forest MSE validation : 0.15867210889950462
# Random Forest R²: 0.31245394251395653
# Random Forest MSE: 0.171819114503117
# speechiness         0.126530
# instrumentalness    0.122276
# loudness            0.109353
# duration_ms         0.104678
# energy              0.103670
# danceability        0.095080
# acousticness        0.092643
# valence             0.088084
# tempo               0.060844
# liveness            0.058636
# key                 0.028300
# mode                0.006978
# time_signature      0.002927


# Parameter tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf = RandomForestRegressor()
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(train_data_features, train_data_conclusion)
best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(test_data_features)
y_pred_rf_val = best_rf.predict(validation_data_features)
print("Random Forest Best Model R² validation :", r2_score(validation_data_conclusion, y_pred_rf_val))
print("Random Forest Best Model MSE validation :", mean_squared_error(validation_data_conclusion, y_pred_rf_val))
print("Random Forest Best Model R²:", r2_score(test_data_conclusion, y_pred_rf))
print("Random Forest Best Model MSE:", mean_squared_error(test_data_conclusion, y_pred_rf))
importances = best_rf.feature_importances_
X = train_data_unedited.drop(columns=['Unnamed: 0','target','song_title','artist'])
feature_importance = pd.Series(importances, index=X.columns)
feature_importance.sort_values(ascending=False, inplace=True)
print(feature_importance)
# Random Forest Best Model R² validation : 0.35355449982187204
# Random Forest Best Model MSE validation : 0.16150647091046116
# Random Forest Best Model R²: 0.32650498166058983
# Random Forest Best Model MSE: 0.16830773213427522
# instrumentalness    0.128488
# duration_ms         0.116569
# loudness            0.114548
# speechiness         0.106845
# danceability        0.100134
# energy              0.098066
# acousticness        0.097126
# valence             0.075859
# tempo               0.065333
# liveness            0.058094
# key                 0.027153
# mode                0.007876
# time_signature      0.003909