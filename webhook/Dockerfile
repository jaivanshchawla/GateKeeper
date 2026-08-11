FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY webhook/ ./webhook/

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "webhook/app.py"]