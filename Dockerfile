FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy Poetry files and README
COPY pyproject.toml poetry.lock* README.md ./

# Copy application code
COPY mechafil_server/ ./mechafil_server/

# Configure Poetry: Don't create virtual environment, install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --only=main --no-interaction --no-ansi

# Create shared cache directory (will be mounted as volume)
RUN mkdir -p /data/shared-cache

# Expose port
EXPOSE 8000

# Run the server
CMD ["python", "-m", "mechafil_server.main"]