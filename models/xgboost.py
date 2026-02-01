from xgboost import XGBClassifier

def train_model(X_train, y_train):
    """
    Trains XGBoost ensemble classifier
    """
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
