from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

class Model:
    def __init__(self, model):
        self.model = model


    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)


    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        return predictions



    def get_confusion(self, data, labels):
        predictions = self.predict(data)
        return confusion_matrix(labels, predictions)




    def get_mse(self, data, labels):
        pass

