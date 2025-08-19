FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Start FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "$PORT"]
