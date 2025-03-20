# Use the official Python image as a base image
FROM python:3.13

# Set the working directory inside the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Set the command to run your app (make sure you replace with your actual Python file)
# CMD ["python", "train.py"]
CMD ["python", "test.py"]