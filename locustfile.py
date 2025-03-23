from locust import HttpUser, task, between
import random
import joblib

# Load test data
x_test, _ = joblib.load("test_data.pkl")

class LoadTestUser(HttpUser):
    wait_time = between(1, 5)  # Simulate user behavior

    @task
    def test_prediction(self):
        random_sample = random.choice(x_test).tolist()  # Pick a random test sample
        self.client.post("/predict", json={"features": random_sample})  # Send request

    @task
    def test_health(self):
        self.client.get("/health")  # Check API health