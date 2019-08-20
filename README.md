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
from sknet import Layer,Sequential,SKNeuron
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression

NFOLDS = 5 # set folds for out-of-fold prediction


layer1 = Layer([
    SKNeuron(RandomForestRegressor(),params = {"random_state" = 0}),
    SKNeuron(RandomForestRegressor(),params = {"random_state" = 1}),
    SKNeuron(AdaBoostRegressor(),params = {"random_state" = 0}),
    SKNeuron(AdaBoostRegressor(),params = {"random_state" = 1}),
    SKNeuron(LinearSVR(),params = {"random_state" = 0}),
    SKNeuron(LinearSVR(),params = {"random_state" = 1}),
])

layer2 = Layer([
    SKNeuron(Lasso(),params = {"random_state" = 0}),
    SKNeuron(Lasso(),params = {"random_state" = 1}),
])


layer3 = Layer([
    SKNeuron(LogisticRegression(),params = {"random_state" = 0}),
])


model = Sequential([layer1,layer2,layer3],n_folds = NFOLDS)
model.fit(X_train,y_train)

predicted = model.predict(X_test)
```

## Todo
- Two or three level stacking
- multi-processing
- features proxy



