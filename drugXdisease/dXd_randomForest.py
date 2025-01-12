from sklearn.ensemble import RandomForestClassifier

def train_classifier_rf(X_train, y_train):
    """
    Train a Random Forest classifier on the training data.
    """
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train)
    return classifier