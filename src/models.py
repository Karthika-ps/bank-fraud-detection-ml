from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_logistic_regression(X_train_scaled, y_train, random_state=42):
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state
    )
    model.fit(X_train_scaled, y_train)
    return model

def train_random_forest(X_train, y_train, random_state=42):
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model
