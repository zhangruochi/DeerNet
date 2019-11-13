from deernet.sequential import Layer,Sequential,SKNeuron

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


data = load_breast_cancer()
features = data.data
target = data.target

scale = MinMaxScaler()
features = scale.fit_transform(features)


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


print(X_train.shape)

layer1 = Layer([
    SKNeuron(RandomForestRegressor,params = {"random_state": 0, "n_estimators": 100}),
    SKNeuron(GradientBoostingRegressor,params = {"random_state": 0,"n_estimators": 100}),
    SKNeuron(AdaBoostRegressor,params = {"random_state": 0,"n_estimators": 100}),
    SKNeuron(KNeighborsRegressor,params = {"n_neighbors":5}),
    SKNeuron(KNeighborsClassifier, score_func = "predict_proba"), 
    SKNeuron(ExtraTreesRegressor,params = {"random_state": 0,"n_estimators": 100}),
    SKNeuron(SVC, params = {"random_state": 0,"max_iter":3000,"gamma" :"auto"},  score_func = "decision_function"), 
    SKNeuron(AdaBoostClassifier,params = {"random_state": 0}, score_func = "predict_proba"),
])

layer2 = Layer([
    SKNeuron(AdaBoostRegressor,params = {"random_state": 0}),
    SKNeuron(SVR,params = {"max_iter":3000,"gamma" :"auto"}),
    SKNeuron(LinearRegression,params = {"normalize": True}),
    SKNeuron(Lasso,params = {"random_state": 0}),
    SKNeuron(ElasticNet,params = {"random_state": 0}), 
    SKNeuron(Ridge,params = {"random_state": 0}),

])


# layer3 = Layer([
#     SKNeuron(LogisticRegression,params = {"random_state": 0,"solver":"lbfgs"})
# ])


# model = Sequential([layer1,layer2,layer3],n_splits = 5)
# y_pred = model.fit_predict(X_train,y_train, X_test)
# print(model.score(y_test,y_pred))


layer3 = Layer([
    SKNeuron(LogisticRegression,params = {"random_state": 0,"solver":"lbfgs"}, score_func = "decision_function")
])

model = Sequential([layer1,layer2,layer3],n_splits = 5)
y_pred = model.fit_predict(X_train,y_train, X_test)
auc = roc_auc_score(y_test, y_pred)
print(auc)