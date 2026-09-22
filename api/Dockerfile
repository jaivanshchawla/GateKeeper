FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (libgomp for LightGBM, git for file-history endpoints)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 git && rm -rf /var/lib/apt/lists/*

# Copy slim requirements and install only serving dependencies
COPY api/requirements-slim.txt ./requirements-slim.txt
RUN pip install --no-cache-dir -r requirements-slim.txt

# Copy source code
COPY api/ ./api/
COPY ml/ ./ml/
COPY models/ ./models/
# Dashboard static build served by FastAPI (W7.3: single backend)
COPY dashboard/dist/ ./dashboard/dist/

# Entrypoint reads $PORT from Render (default 10000) or falls back to 8000 locally
RUN chmod +x /app/api/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/api/entrypoint.sh"]