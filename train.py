import joblib

import logging

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Function to load the Wine dataset
def load_dataset():
    """
    Loads the Wine dataset using sklearn's datasets modules.

    Returns:
        X (numpy array): Feature matrix containing the attributes of the wines.
        Y (numpy array): Target array containing the class labels for the wines.
    """
    # Load the Wine dataset
    logging.info("Loading Wine dataset...")
    ds = datasets.load_wine()

    # Access the features (X) and target (Y)
    X = ds.data  # Feature matrix (attributes of the wines)
    Y = ds.target  # Target variable (labels representing wine classes)

    return X, Y


# Function to split the dataset into training and testing sets
def split_data(x, y, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.

    Parameters:
        x (numpy array): Feature matrix.
        y (numpy array): Target labels.
        test_size (float): Proportion of the dataset to be used for testing. Default is 0.2 (20%).
        random_state (int): Random seed for reproducibility.

    Returns:
        x_train, x_test, y_train, y_test (numpy arrays): Split datasets for training and testing.
    """
    logging.info("Splitting data into training and test sets...")
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=test_size, random_state=random_state)

    return xtrain, xtest, ytrain, ytest


# Load dataset
features, labels = load_dataset()

# Split the data into training and test sets
x_train, x_test, y_train, y_test = split_data(features, labels)

# Initialize the RandomForestClassifier model
rf = RandomForestClassifier(random_state=42)

# Train the Random Forest model on the training data
logging.info("Training RandomForestClassifier model...")
rf.fit(x_train, y_train)

# Log the shapes of the split datasets (lazy formatting to resolve W1203)
logging.info("Training data shape: %s, Test data shape: %s", x_train.shape, x_test.shape)
logging.info("Training target shape: %s, Test target shape: %s", y_train.shape, y_test.shape)

# Save the trained model
joblib.dump(rf, "model.pkl")
logging.info("Model saved as model.pkl")

# Save test data for API validation
joblib.dump((x_test, y_test), "test_data.pkl")
logging.info("Test data saved as test_data.pkl")

# The trained model and data are now ready for reuse in other files (e.g., test.py).
