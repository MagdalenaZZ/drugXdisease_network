from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def train_classifier_nn(X_train, y_train, input_dim, epochs=10, batch_size=32):
    """
    Train a Neural Network classifier on the training data.

    Parameters:
    - X_train: numpy array, training feature matrix.
    - y_train: numpy array, training labels.
    - input_dim: int, number of input features.
    - epochs: int, number of training epochs.
    - batch_size: int, batch size for training.

    Returns:
    - model: trained Keras model.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))

    # Build the model
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model with class weights
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, class_weight=class_weights, verbose=1)
    return model

