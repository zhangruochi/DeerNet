from sklearn.model_selection import KFold
import numpy as np

class Sequential(object):
    def __init__(self,layer_list, n_splits):
        self.layer_list = layer_list
        self.kf = KFold(n_splits= n_splits)


    def fit(self,x_train, y_train):
        for layer in self.layer_list:
            layer_features = np.empty((x_train.shape[0],layer.get_layer_length()))
            for i,estimator in enumerate(layer.get_estimator_list()):
                layer_features[:,i] = self.get_tr_neuron_feature(estimator, x_train, y_train)
            x_train = layer_features
    




    def get_tr_neuron_feature(self,clf, x_train, y_train):
        # print(x_train.shape) 
        # (455, 30)
        neuron_features_train = np.zeros((x_train.shape[0],))
    
        # neuron_test = np.zeros((ntest,))      
        # neuron_test_skf = np.empty((NFOLDS, ntest))

        for train_index, test_index in self.kf.split(x_train):
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]

            clf.train(x_tr, y_tr)

            neuron_features_train[test_index] = clf.predict(x_te)
        #     neuron_test_skf[i, :] = clf.predict(x_test)

        # neuron_test[:] = oof_test_skf.mean(axis=0)
        return neuron_features_train

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
        self.clf = clf(**params)

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

