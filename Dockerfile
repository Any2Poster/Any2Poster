FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Playwright + Chromium
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt .
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e ".[render]"

# Install Playwright Chromium browser
RUN playwright install chromium
RUN playwright install-deps chromium

# Copy source code
COPY any2poster/ ./any2poster/
COPY config/ ./config/
COPY .env.example .env.example

# Create volume mount points
VOLUME ["/input", "/output"]

ENTRYPOINT ["any2poster"]
CMD ["--help"]
