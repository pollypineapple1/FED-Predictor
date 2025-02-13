# Start with a base image that contains Python
FROM python:3.10-slim

# Install necessary dependencies for Python packages
RUN apt-get update && apt-get install -y curl build-essential

# Install pip and upgrade it
RUN pip install --upgrade pip

# Set the working directory in the container
WORKDIR /prod

# Copy only the requirements file first (to leverage Docker caching)
COPY requirements_docker.txt /prod/requirements_docker.txt

# Install dependencies (with preference for pre-built wheels)
RUN pip install --no-cache-dir --prefer-binary -r /prod/requirements_docker.txt

# Copy the rest of the application code
COPY . /prod
COPY FED-predictor/api/checkpoints/best_model.pth /prod/FED-predictor/api/checkpoints/

# Copy the .env file (be sure to use this only if needed, sensitive info should be handled carefully)
COPY .env /prod/.env

# Expose the port your app will run on
ENV PORT=8000

# explicitly set the PYTHONPATH in dockerfile to help finding modules
ENV PYTHONPATH="/prod/FED-predictor"

# Command to run the app (Adjust according to your app's structure)
CMD ["sh", "-c", "uvicorn FED-predictor.api.fast:app --host 0.0.0.0 --port ${PORT}"]