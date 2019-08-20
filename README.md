# SKNet

## Introduction
SKNet is a new type of neural network that is simple in structure but complex in neuron. Each of its neuron is a traditional estimator such as SVM, RF, etc.  

## Fetaures 
We think that such a network has many applicable scenarios.  
- We don't have enough samples to train neural networks. 
- We hope to improve the accuracy of the model by means of emsemble. 
- We hope to learn some new features. 
- We want to save a lot of parameter adjustment time while getting a stable and good model.


## Installation

```python3
pip install sknet
```


## Example

### Computation Graph

![](./computation_graph.png)

### Code

```python
from sknet.sequential import Layer,Sequential,SKNeuron

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


data = load_breast_cancer()
features = data.data
target = data.target

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)



layer1 = Layer([
    SKNeuron(RandomForestRegressor,params = {"random_state": 0}),
    SKNeuron(GradientBoostingRegressor,params = {"random_state": 0}),
    SKNeuron(AdaBoostRegressor,params = {"random_state": 0}),
    SKNeuron(KNeighborsRegressor),
    SKNeuron(ExtraTreesRegressor,params = {"random_state": 0}),
])

layer2 = Layer([
    SKNeuron(AdaBoostRegressor,params = {"random_state": 0}),
    SKNeuron(LinearSVR,params = {"random_state": 0}),
])

layer3 = Layer([
    SKNeuron(LogisticRegression,params = {"random_state": 0}),
])


model = Sequential([layer1,layer2,layer3],n_splits = 5)
y_pred = model.fit_predict(X_train,y_train, X_test)
print(model.score(y_test,y_pred))


# acc = 0.9736842105263158
```

## Todo
- Two or three level stacking
- multi-processing
- features proxy



