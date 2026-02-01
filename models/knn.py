from sklearn.neighbors import KNeighborsClassifier

def train_model(X_train, y_train):
    """
    Trains K-Nearest Neighbors classifier
    """
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    return model
