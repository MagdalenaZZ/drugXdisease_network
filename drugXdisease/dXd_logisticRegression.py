from sklearn.linear_model import LogisticRegression

def train_classifier_lr(X_train, y_train):
    """
    Train a Logistic Regression classifier on the training data.

    Parameters:
    - X_train: numpy array, training feature matrix.
    - y_train: numpy array, training labels.

    Returns:
    - classifier: trained Logistic Regression model.
    """
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train, y_train)
    return classifier
