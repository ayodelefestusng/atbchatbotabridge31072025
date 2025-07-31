# ---------- Stage 1: Build Dependencies ----------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies for psycopg (and other packages that might need C extensions)
# libpq-dev provides the necessary headers for psycopg to compile against PostgreSQL client libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install pip packages with retry and timeout resilience
RUN pip install --upgrade pip \
    && pip install --default-timeout=100 --retries=5 --no-cache-dir -r requirements.txt

# --- ADDED: Download and install the spaCy language model ---
# This ensures the model is present in the /usr/local path and copied to the final image
RUN python -m spacy download en_core_web_sm

# ---------- Stage 2: Production Runtime Image ----------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install runtime dependencies for psycopg AND Java for LanguageTool
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    openjdk-17-jre-headless \ 
    && apt-get clean && rm -rf /var/lib/apt/lists/* 

# Create appuser before chown
RUN useradd -m appuser

WORKDIR /app

# Copy installed packages from builder stage (this now includes the spaCy model)
COPY --from=builder /usr/local /usr/local

# Copy application source code
COPY . .

# Collect static files and prepare dirs
RUN mkdir -p /app/staticfiles \
    && python manage.py collectstatic --noinput

# Set ownership to non-root user
RUN chown -R appuser /app

# Switch to non-root user for safety
USER appuser

# Final command: run Gunicorn server with a timeout (e.g., 120 seconds)
CMD ["gunicorn", "myproject.wsgi:application", "--bind", "0.0.0.0:8000", "--timeout", "120"]
