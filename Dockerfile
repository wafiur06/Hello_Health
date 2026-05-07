# Use a slim Python image to keep size down, but not Alpine (Alpine struggles with heavy ML libs)
FROM python:3.10-slim

# Install system dependencies for audio and ML
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
# Note: This might take a while because of Torch and TensorFlow
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port Flask runs on
EXPOSE 5000

# Command to run your Flask app
CMD ["python", "app.py"]
