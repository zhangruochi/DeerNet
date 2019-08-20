# SKNet

## Introduction
SKNet is a new type of neural network that is simple in structure but complex in neuron. Each of its neuron is a traditional estimator such as SVM, RF, etc.  

## Fetaures 
We think that such a network has many applicable scenarios.  
- we don't have enough samples to train neural networks. 
- We hope to improve the accuracy of the model by means of emsemble. 
- We hope to learn some new features. 
- We want to save a lot of parameter adjustment time while getting a stable and good model.


## Installation

```python3
pip install sknet
```


## Example

```python
from sknet import Layer,Sequential
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression

layer1 = Layer([
    RandomForestRegressor(random_state = 1),
    RandomForestRegressor(random_state = 2),
    AdaBoostRegressor(random_state = 1),
    AdaBoostRegressor(random_state = 2),
    LinearSVR(random_state = 1),
    LinearSVR(random_state = 2),
])

layer2 = Layer([
    Lasso(random_state = 1),
    Lasso(random_state = 2),
])

layer3 = Layer([
    LogisticRegression(random_state = 1),
])


model = Sequential([layer1,layer2,layer3])
model.fit(X_train,y_train)

predicted = model.predict(X_test)

```


