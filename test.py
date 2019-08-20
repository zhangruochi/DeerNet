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


# 0.9736842105263158