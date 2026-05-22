FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy bot source
COPY main_rolling.py .

# Railway injects PORT; default to 8080 if not set
ENV PORT=8080
ENV PERSIST_DIR=/app/data

# Ensure the data directory exists (Volume will mount here)
RUN mkdir -p /app/data

EXPOSE 8080

CMD ["python", "main_rolling.py"]
