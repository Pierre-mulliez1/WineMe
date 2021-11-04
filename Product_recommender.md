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
from sklearn.datasets import load_wine
data1 = load_wine(as_frame = True)
```

```python
#Load dataset
DIRECTORY_WHERE_THIS_FILE_IS = os.path.dirname(os.path.abspath("Product_recommender.md"))
DATA_PATH = os.path.join(DIRECTORY_WHERE_THIS_FILE_IS, "data/kaggle_wine2.csv")
df1 = pd.read_csv(DATA_PATH)
```

```python
df1['description'].head(2)
```

```python
#extract the wine year
import re
df1['year'] = 0
count = 0
for el in df1['title']: 
    res = [int(i) for i in el.split() if i.isdigit()]
    if len(res) == 1:
        df1.loc[count,'year'] = res[0]
    elif len(res) == 2 and res[0] > 1900:
        df1.loc[count,'year'] = res[0]
    elif len(res) == 2 and res[1] > 1900:
         df1.loc[count,'year'] = res[1]
    count += 1
```

```python
#onehotencode the country qnd yeqr
categorical_columns = ['country', 'variety']
for column in categorical_columns:
    tempdf = pd.get_dummies(df1[column], prefix=column)
    df1 = pd.merge(
        left=df1,
        right=tempdf,
        left_index=True,
        right_index=True,
    )
    df1 = df1.drop(columns=column)
```

```python
#drop the other non numericql collumns
df2 = df1.drop(['Unnamed: 0','designation','description','province','region_1','region_2','taster_name','taster_twitter_handle','winery'], axis = 1)
```

```python
#Take the most represented collumns
for col in df2.columns:
    if col not in ('points', 'price','title'):
        if sum(df2[col]) < 5000 :
            df2 = df2.drop(columns= col)
```

```python
df2.head(5)
```

```python
#null
#drop for now
df2 = df2.dropna()
```

```python
from sklearn.model_selection import train_test_split
y = df2["points"]
X = df2.loc[:,df2.columns != ["points",'title']]
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

#X = X.drop(labels=corr_features, axis = 1)


```

```python
#train test
X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.30, random_state=42)
```

### Baseline

```python
average = sum(y_train) / len(y_train)
```

```python
arr_avg = []
for el in y_test:
    arr_avg.append(average)
```

```python
print('Mean absolute error: {}'.format(mean_absolute_error(y_test,arr_avg)))
```

### Random forest regressor 

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
```

```python
grid = {
    "max_depth": [35,40], 
    "min_samples_leaf": [2,3],
    "min_samples_split": [4,5,8,10]
}

"Sklearn"
"-----------------------"
grid_search = GridSearchCV(RandomForestRegressor(), param_grid = grid)
grid_search.fit(X_train, y_train)
optimal_model = grid_search.best_estimator_
"-----------------------"

print("Fine Tuned Model: {0}".format(optimal_model))
```

```python
model = RandomForestRegressor(max_depth = optimal_model.max_depth, min_samples_leaf = optimal_model.min_samples_leaf, min_samples_split = optimal_model.min_samples_split)
model.fit(X_train,y_train)
adjusted_pred = model.predict(X_test)
```

```python
print('Mean absolute error: {}'.format(mean_absolute_error(y_test,adjusted_pred)))
```

```python

```
