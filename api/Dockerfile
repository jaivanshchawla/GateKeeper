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
COPY models/ ./models/

# Entrypoint reads $PORT from Render (default 10000) or falls back to 8000 locally
RUN chmod +x /app/api/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/api/entrypoint.sh"]