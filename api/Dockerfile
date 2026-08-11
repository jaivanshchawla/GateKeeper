FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (libgomp for LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY api/ ./api/
COPY ml/ ./ml/

# Expose port
EXPOSE 8000

# Run the application (uvicorn.main:app so it resolves api/main.py)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]