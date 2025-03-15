import logging  # For logging

from sklearn.metrics import accuracy_score  # For evaluating the model accuracy

from train import (rf,  # Import the test data and trained model from train.py
                   x_test, y_test)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def test_model():
    """
    Function to test the trained RandomForestClassifier model on the test dataset.

    It predicts the labels for the test set, calculates the accuracy, and logs the result.
    """
    # Make predictions on the test data using the trained model (rf)
    logging.info("Making predictions on the test set...")
    y_pred = rf.predict(x_test)

    # Evaluate the model using accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Log the accuracy (lazy formatting to resolve W1203)
    logging.info("Model accuracy on test data: %.2f%%", accuracy * 100)

    # Assert if the accuracy is greater than a certain threshold, e.g., 80%
    assert accuracy >= 0.90


test_model()
