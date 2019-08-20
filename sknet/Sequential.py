from sklearn.model_selection import KFold
import numpy as np

class Sequential(object):
    def __init__(self,layer_list, n_splits):
        self.layer_list = layer_list
        self.kf = KFold(n_splits = n_splits)


    def fit_predict(self,x_train, y_train, x_test):
        for layer in self.layer_list:
            layer_train_features = np.empty((x_train.shape[0],layer.get_layer_length()))
            layer_test_features = np.empty((x_test.shape[0],layer.get_layer_length()))
            for i,estimator in enumerate(layer.get_estimator_list()):
                layer_train_features[:,i],layer_test_features[:,i] = self.get_tr_neuron_feature(estimator, x_train, y_train, x_test)
            x_train = layer_train_features
            x_test = layer_test_features

        return x_test
        

    def get_tr_neuron_feature(self,clf, x_train, y_train, x_test):
        neuron_features_train = np.zeros((x_train.shape[0],))     
        neuron_features_test = np.empty((x_test.shape[0],self.kf.get_n_splits()))

        for i,(train_index, test_index) in enumerate(self.kf.split(x_train)):
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]

            clf.train(x_tr, y_tr)

            neuron_features_train[test_index] = clf.predict(x_te)
            neuron_features_test[:,i] = clf.predict(x_test)

        return neuron_features_train, neuron_features_test.mean(axis = 1)

    def score(self,y_true,y_pred):
        y_pred = np.reshape(y_pred,(y_pred.shape[0],))
        return np.sum(y_true == y_pred) / len(y_true)


class Layer(object):
    def __init__(self,estimator_list):
        self.estimator_list = estimator_list

    def get_estimator_list(self):
        return self.estimator_list

    def get_layer_length(self):
        return len(self.estimator_list)

    def __getitem__(self,index):
        return self.estimator_list[index]





class SKNeuron(object):
    def __init__(self, clf, params=None):
        if params:
            self.clf = clf(**params)
        else:
            self.clf = clf()

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)

    def __repr__(self):
        return str(self.clf)
    
    # def feature_importances(self,x,y):
    #     print(self.clf.fit(x,y).feature_importances_)

