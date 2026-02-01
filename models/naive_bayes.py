from sklearn.naive_bayes import GaussianNB

def train_model(X_train, y_train):
    """
    Trains Gaussian Naive Bayes classifier
    """
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model
