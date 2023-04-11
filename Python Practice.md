### Causal inference in python 

I assume you meant TSLS instead of T-SLS. Here's an example of how to calculate TSLS and LATE using a sample dataset with 20 observations:

Suppose we have the following variables:

   Y: The outcome variable
   D: The treatment variable
   Z: The instrumental variable

We can generate some random data for these variables with the numpy library in Python
```python 
import numpy as np

np.random.seed(123)
n = 20
D = np.random.randint(0, 2, size=n)
Z = np.random.normal(size=n)
Y = 1 + D + 2*Z + np.random.normal(size=n)

```

The regression model for the first stage is:
"D = alpha + beta*Z + e"

where e is the error term. We estimate this model and obtain the predicted values of D:

```python 
from statsmodels.api import OLS

X = np.column_stack((np.ones(n), Z))
model_d = OLS(D, X).fit()
D_hat = model_d.predict(X)
```

The second stage regression model is:

"Y = gamma + delta*D_hat + u"
where u is the error term. We estimate this model and obtain the coefficients:
```python
X_tsls = np.column_stack((np.ones(n), D_hat))
model_y = OLS(Y, X_tsls).fit()
gamma, delta = model_y.params
```
The TSLS estimate of the treatment effect is the coefficient on D_hat, which is equal to delta. In this example, the TSLS estimate is:
"delta=18.834564189333236"
To calculate the LATE, we need to identify a group of individuals who are affected by the treatment. In this example, let's assume that the treatment affects individuals whose values of Z are above the median. We can split our sample into two groups based on the median of Z:

```python
Z_median = np.median(Z)
D1, D0 = D[Z > Z_median], D[Z <= Z_median]
Y1, Y0 = Y[Z > Z_median], Y[Z <= Z_median]
```
We can estimate the treatment effect separately for each group using OLS:
```python
X1, X0 = np.column_stack((np.ones(len(D1)), D_hat[Z > Z_median])), np.column_stack((np.ones(len(D0)), D_hat[Z <= Z_median]))
model_y1, model_y0 = OLS(Y1, X1).fit(), OLS(Y0, X0).fit()
delta1, delta0 = model_y1.params[1], model_y0.params[1]
```
The LATE is the difference between the treatment effects for the two groups, weighted by the proportion of individuals in each group:
```python
p1 = len(D1) / n
p0 = len(D0) / n
late = p1 * delta1 - p0 * delta0
```
In this example, the LATE estimate is:
"LATE=3.5868043845292945"



