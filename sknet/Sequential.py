from sklearn.model_selection import KFold

class Sequential(object):
    def __init__(self,layer_list, kf):
        self.layer_list = layer_list
        self.kf = KFold(ntrain, n_folds= NFOLDS, random_state=0)


    def fit(self):
        for layer in self.layer_list:
            for estimator in layer.get_estimator_list():
                neuron_feature = self.get_tr_neuron_feature(et, x_train, y_train, x_test)



    def get_tr_neuron_feature(clf, x_train, y_train, x_test):
        neuron_features_train = np.zeros((ntrain,))
        
        # neuron_test = np.zeros((ntest,))      
        # neuron_test_skf = np.empty((NFOLDS, ntest))

        for i, (train_index, test_index) in enumerate(self.kf):
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]

            clf.train(x_tr, y_tr)

            neuron_features_train[test_index] = clf.predict(x_te)
        #     neuron_test_skf[i, :] = clf.predict(x_test)

        # neuron_test[:] = oof_test_skf.mean(axis=0)
        return neuron_features_train.reshape(-1, 1)

class Layer(object):
    def __init__(self,estimator_list):
        self.estimator_list = estimator_list

    def get_estimator_list(self):
        return self.estimator_list

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

