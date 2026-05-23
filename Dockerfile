# Portofino RiskView – Hugging Face Docker Space
# Docker Spaces expose the application through port 7860 by default.

FROM python:3.11-slim

# Runtime defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    PIP_NO_CACHE_DIR=1

# System packages required for common Python ML dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Docker Spaces run with user ID 1000.
# Creating the same user avoids file-permission issues.
RUN useradd -m -u 1000 user

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install Python dependencies first for better Docker layer caching
COPY --chown=user requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# Copy the complete project
COPY --chown=user . .

# Hugging Face Space port
EXPOSE 7860

# Start the existing FastAPI/Uvicorn app exactly as locally
CMD ["python", "app.py"]
