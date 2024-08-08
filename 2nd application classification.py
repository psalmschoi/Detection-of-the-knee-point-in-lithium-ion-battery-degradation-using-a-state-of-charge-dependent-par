# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 10:17:04 2024

@author: KIMHYUNJAE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


file_path = '~2nd_lifetime.xlsx' 
df = pd.read_excel(file_path)


X = df['Var (ΔQ0.2C-1C(V))'].values.reshape(-1, 1) # [Var (ΔQ0.2C-1C(V)), ΔQ_Δcycle, SOH, DCIR at low SOC(5%)]
y = df.iloc[:, 6].values


train_accuracies = []
test_accuracies = []
best_test_accuracy = 0
best_X_train = None
best_y_train_pred = None
best_y_train_true = None
best_X_test = None
best_y_test_pred = None
best_y_test_true = None

for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i, shuffle=True)
    
  
    pipe = Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression())])

  
    param_grid = {
        'logreg__C': np.logspace(-4, 4, 20),
        'logreg__penalty': ['l1', 'l2'],  # L1 = Lasso, L2 = Ridge
        'logreg__solver': ['liblinear']   # L1 규제에 필요한 solver
    }

    grid_search = GridSearchCV(pipe, param_grid, cv=StratifiedKFold(5), n_jobs=-1)
    grid_search.fit(X_train, y_train)

 
    best_model = grid_search.best_estimator_
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        best_train_accuracy = train_accuracy
        best_X_train = X_train
        best_y_train_pred = y_train_pred
        best_y_train_true = y_train
        best_X_test = X_test
        best_y_test_pred = y_test_pred
        best_y_test_true = y_test


train_accuracy_mean = sum(train_accuracies) / len(train_accuracies)
test_accuracy_mean = sum(test_accuracies) / len(test_accuracies)


decision_boundary = None
sorted_indices = np.argsort(best_X_train.flatten())
sorted_X_train = best_X_train[sorted_indices].flatten()
sorted_y_train_pred = best_y_train_pred[sorted_indices]

for i in range(1, len(sorted_X_train)):
    if sorted_y_train_pred[i] != sorted_y_train_pred[i - 1]:
        decision_boundary = (sorted_X_train[i] + sorted_X_train[i - 1]) / 2
        decision_boundary = round(decision_boundary, 4)
        break


misclassified_train = best_X_train[best_y_train_pred != best_y_train_true]
misclassified_train_pred = best_y_train_pred[best_y_train_pred != best_y_train_true]


misclassified_test = best_X_test[best_y_test_pred != best_y_test_true]
misclassified_test_pred = best_y_test_pred[best_y_test_pred != best_y_test_true]


print("Misclassified Train Points:")
print(misclassified_train, misclassified_train_pred)

print("Misclassified Test Points:")
print(misclassified_test, misclassified_test_pred)


plt.figure(figsize=(10, 6))


plt.scatter(best_X_train, best_y_train_pred, color='blue', marker='o', label='Train')


plt.scatter(best_X_test, best_y_test_pred, color='red', marker='^', label='Test')


plt.scatter(misclassified_train, misclassified_train_pred, color='orange', marker='x', label='Misclassified Train')
plt.scatter(misclassified_test, misclassified_test_pred, color='purple', marker='x', label='Misclassified Test')


if decision_boundary is not None:
    plt.axvline(x=decision_boundary, color='green', linestyle='--', label=f'Decision Boundary: {decision_boundary:.4f}')

plt.xlabel('Feature Value')
plt.ylabel('Predicted Class')
plt.title('Training and Test Set Predictions with Decision Boundary and Misclassifications')
plt.legend()

plt.show()

print(f"Train Accuracy (mean): {train_accuracy_mean:.4f}")
print(f"Test Accuracy (mean): {test_accuracy_mean:.4f}")
if decision_boundary is not None:
    print(f"Decision Boundary at Feature Value: {decision_boundary:.4f}")
else:
    print("No clear decision boundary found.")
