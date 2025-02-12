# Start with a base image that contains Python
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /prod

# Copy only the requirements file first (to leverage Docker caching)
COPY requirements.txt /prod/requirements.txt

# Install dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r /prod/requirements.txt

# Copy the rest of the application code
COPY . /prod
# Copy the .env file (be sure to use this only if needed, sensitive info should be handled carefully)
COPY .env /prod/.env

# Expose the port your app will run on
ENV PORT=8000

# Command to run the app (Adjust according to your app's structure)
CMD ["uvicorn", "FED-predictor.api.fast:app", "--host", "0.0.0.0", "--port", "$PORT"]