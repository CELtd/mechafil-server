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

# Copy shared code
COPY shared/ ./shared/

# Copy cache updater service code
COPY services/cache_updater/ ./services/cache_updater/

# Configure Poetry: Don't create virtual environment, install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --only=main --no-interaction --no-ansi

# Create shared cache directory (will be mounted as volume)
RUN mkdir -p /data/shared-cache

# Lambda-like execution: run once and exit
CMD ["python", "-m", "services.cache_updater.main", "--once"]