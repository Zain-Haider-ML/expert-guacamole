# Import necessary libraries
from train import x_test, y_test, rf  # Import the test data and trained model from train.py
from sklearn.metrics import accuracy_score  # For evaluating the model accuracy
import logging  # For logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Make predictions on the test data using the trained model (rf)
logging.info("Making predictions on the test set...")
y_pred = rf.predict(x_test)

# Evaluate the model using accuracy
accuracy = accuracy_score(y_test, y_pred)

# Log the accuracy
logging.info(f"Model accuracy on test data: {accuracy * 100:.2f}%")