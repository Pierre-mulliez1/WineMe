---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Product recommender system

```python
#import packages 
import sklearn
import pandas as pd
import numpy as np
import os
```

## Get the data

```python
#Load dataset

DIRECTORY_WHERE_THIS_FILE_IS = os.path.dirname(os.path.abspath("Product_recommender.md"))
DATA_PATH = os.path.join(DIRECTORY_WHERE_THIS_FILE_IS, "data/kaggle_wine2.csv")
df1 = pd.read_csv(DATA_PATH)
```

```python
from sklearn.model_selection import train_test_split
y = df1["points"]
X = df1.loc[:,df1.columns != "points"]
```

```python
print(X.head())

#onhotencode the country
#province ?
```

```python
#scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
X = pd.DataFrame(X)

#remove correlated features
def get_correlation(data, threshold):
    corr_col = set()
    corrmat = data.corr()
    for i in range(len(corrmat.columns)):
        for j in range(i):
            if abs(corrmat.iloc[i, j]) > threshold:
                colname = corrmat.columns[i]
                corr_col.add(colname)
    return corr_col

corr_features = get_correlation(X, 0.70)
print('correlated features: ', len(set(corr_features)) )
print(corr_features)

X = X.drop(labels=corr_features, axis = 1)

#train test
X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.20, random_state=42)
```

## Linear models


### Ridge

```python
from sklearn.linear_model import LogisticRegression
```

```python
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred = clf.predict(X_test)
print()
  
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
```

```python

```

## Non linear model 

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

random_grid = {'n_estimators': [200,300,500,800,1300,1500],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [10,30,50,80],
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4]}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 5, cv = 2, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X, y)
rf_random.best_params_
```

```python
# creating a RF classifier
clf = RandomForestClassifier(n_estimators = 500,min_samples_split = 10,
 min_samples_leaf= 1,
 max_features = 'sqrt',
 max_depth = 80)  
  
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)
  
# performing predictions on the test dataset
y_pred = clf.predict(X_test)
  
# metrics are used to find accuracy or error
from sklearn import metrics  
print()
  
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
```
