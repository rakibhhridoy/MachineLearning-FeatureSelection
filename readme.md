# *Machine Learning Feature Selection*
>Step By step
1. *importing libraries & functions*
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
import os


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
```

2. *loading datasets*
```python
file = os.getcwd()+"/datasets_228_482_diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = pd.read_csv(file, names = names)

array = df.values

X = array[:, 0:8]
y = array[:,8]
```

3. Different feature selection techniques:
> SelectKBest
```python
test = SelectKBest(score_func = f_classif, k=4) 
fit = test.fit(X,y)

features = fit.transform(X)

corr_p = df['skin'].corr(df['class'])
print(corr_p)

print(features[0:5,:])


model = LogisticRegression(solver = 'lbfgs')
rfe = RFE(model, 3)
fit = rfe.fit(X,y)


print('Num features: %d' % fit.n_features_)
print('Selected features: %s' % fit.support_)
print('feature ranking: %s' % fit.ranking_)
```
>ExtraTreeClasssifier
```python
model = ExtraTreesClassifier(n_estimators=10)
model.fit(X,y)

print(model.feature_importances_)
```
>Dimensionality Reduction- PCA
```python
pca = PCA(n_components = 3)
fit = pca.fit(X,y)

print('Explained Variance: %s'% fit.explained_variance_ratio_)
print(fit.components_)
```
> best params and score findings
```python
lasso = Lasso()

parameters = {'alpha': [1e-15,1e-10, 1e-8, 1e-4, 1e-3,1e-2,1,5,10,20]}

lasso_regressor = GridSearchCV(lasso, parameters, scoring = 'neg_mean_squared_error', cv=5)
lasso_regressor.fit(X,y)

print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)
```

#### *Get Touch With Me*
Connect- [Linkedin](https://linkedin.com/in/rakibhhridoy) <br>
Website- [RakibHHridoy](https://rakibhhridoy.github.io)
